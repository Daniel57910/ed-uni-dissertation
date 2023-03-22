import argparse
import glob
import os
import pdb
import pprint as pp
from datetime import datetime

import boto3
import torch
import tqdm
from pprint import pformat

if torch.cuda.is_available():
    import cupy as np
    import numpy
    import cudf as pd
    import dask_cudf as dd
    from cuml.preprocessing import MinMaxScaler
    import GPUtil
else:
    import numpy as np
    import numpy
    import pandas as pd
    import dask.dataframe as dd
    from sklearn.preprocessing import MinMaxScaler

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=4)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=200)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

torch.set_printoptions(linewidth=400, precision=4, sci_mode=False)

from constant import (
    INITIAL_LOAD_COLUMNS,
    SCALED_COLS,
    GENERATED_COLS,
    ENCODED_COLS
)
def _encode_countries(x):
        if x == 'Finland':
            return 1
        elif x == 'United States':
            return 2
        elif x == 'China':
            return 3
        else:
            return 4

def get_logger():
    logger = logging.getLogger(__name__)
    return logger



def join_for_encodings(df):


    project_id_value_counts = df['project_id'].value_counts().reset_index().rename(columns={'index': 'project_id', 'project_id': 'project_count'})
    df = df.merge(project_id_value_counts, on='project_id', how='left')

    if torch.cuda.is_available():
        df = df.compute().to_pandas()
    else:
        df = df.compute()

    df['country'] = df['country'].apply(_encode_countries)

    if torch.cuda.is_available():
        df = pd.from_pandas(df)

    df['date_time'] = pd.to_datetime(df['date_time'])
    total_events = df['user_id'].value_counts().reset_index().rename(columns={'index': 'user_id', 'user_id': 'total_events'})

    df = df.merge(total_events, on='user_id', how='left')
    df['cum_events'] = df[['user_id', 'task_within_session_count']].groupby('user_id').cumcount() + 1
    df['cum_projects'] = (df.groupby('user_id')['project_id'].transform(lambda x: pd.CategoricalIndex(x).codes) + 1).astype('int32')
    df['cum_time'] = df.groupby('user_id')['time_diff_seconds'].cumsum()
    df['cum_time_within_session'] = df.groupby(['user_id', '30_minute_session_count'])['time_diff_seconds'].cumsum()
    df['av_time_across_clicks'] = df.groupby('user_id')['time_diff_seconds'].rolling(1000, min_periods=1).mean().reset_index()['time_diff_seconds']
    df['av_time_across_clicks_session'] = df.groupby(['user_id', '30_minute_session_count'])['time_diff_seconds'].rolling(10, min_periods=1).mean().reset_index()['time_diff_seconds']


    av_num_events_within_session = df.groupby(['user_id', '30_minute_session_count'])['task_within_session_count'].max().reset_index()
    av_num_events_within_session['rolling_average_tasks_within_session'] = av_num_events_within_session.groupby('user_id')['task_within_session_count'].rolling(10, min_periods=1).mean().reset_index()['task_within_session_count']

    av_time_within_session = df.groupby(['user_id', '30_minute_session_count']).apply(lambda x: x['date_time'].max() - x['date_time'].min()).reset_index().rename(columns={0: 'av_time_within_session'})
    av_time_within_session['av_time_within_session'] = av_time_within_session['av_time_within_session'].apply(lambda x: x.total_seconds())
    av_time_within_session['rolling_av_time_within_session'] = av_time_within_session.groupby('user_id')['av_time_within_session'].rolling(10, min_periods=1).mean().reset_index()['av_time_within_session']


    session_start_time = df.groupby(['user_id', '30_minute_session_count'])['date_time'].min().reset_index().rename(columns={'date_time': 'session_start_time'})
    session_end_time = df.groupby(['user_id', '30_minute_session_count'])['date_time'].max().reset_index().rename(columns={'date_time': 'session_end_time'})

    session_meta = session_start_time.merge(session_end_time, on=['user_id', '30_minute_session_count'], how='left')
    session_meta['previous_end_time'] = session_meta.groupby(['user_id'])['session_end_time'].shift()
    session_meta['previous_end_time'] = session_meta['previous_end_time'].fillna(0)
    session_meta['time_between_sessions'] = session_meta.apply(lambda x: 0 if x['previous_end_time'] == 0 else (x['session_start_time'] - x['previous_end_time']), axis=1)
    session_meta['time_between_sessions'] = session_meta['time_between_sessions'].apply(lambda x: 0 if x == 0 else x.seconds)
    session_meta['rolling_time_between_sessions'] = session_meta.groupby('user_id')['time_between_sessions'].rolling(5,  min_periods=1).mean().reset_index()['time_between_sessions']

    df = df.merge(av_num_events_within_session[['user_id', '30_minute_session_count', 'rolling_average_tasks_within_session']], on=['user_id', '30_minute_session_count'], how='left')
    df = df.merge(av_time_within_session[['user_id', '30_minute_session_count', 'rolling_av_time_within_session']], on=['user_id', '30_minute_session_count'], how='left')
    df = df.merge(session_meta[['user_id', '30_minute_session_count', 'rolling_time_between_sessions']], on=['user_id', '30_minute_session_count'], how='left')
    user_id_hash = pd.DataFrame(df['user_id'].unique()).reset_index().rename(columns={'index': 'user_id_hash', 0: 'user_id'})
    project_id_hash = pd.DataFrame(df['project_id'].unique()).reset_index().rename(columns={'index': 'project_id_hash', 0: 'project_id'})

    user_id_hash['user_id_hash'] = user_id_hash['user_id_hash'] + 1
    project_id_hash['project_id_hash'] = project_id_hash['project_id_hash'] + 1

    df = df.merge(user_id_hash, on='user_id', how='left')
    df = df.merge(project_id_hash, on='project_id', how='left')

    df = df.drop(columns=['user_id', 'project_id'])
    df = df.rename(columns={'user_id_hash': 'user_id', 'project_id_hash': 'project_id'})
    return df

def prepare_for_sessionization(data_paths: list, scaler: MinMaxScaler):

    df = dd.read_csv(data_paths, usecols=INITIAL_LOAD_COLUMNS)

    df = join_for_encodings(df)
    df['timestamp_raw'] = df['date_time'].astype('int64') // 10**9
    df = df.sort_values(by='date_time')

    df['timestamp'] = df['timestamp_raw']
    print(f'Loaded data: shape = {df.shape}, min_date, max_date: {df.date_time.min()}, {df.date_time.max()}')
    print(f'Data label true: {df.label.value_counts() / len(df)}')
    df = df[SCALED_COLS + GENERATED_COLS + ENCODED_COLS + ['timestamp_raw', 'label', 'total_events']]
    df[SCALED_COLS + GENERATED_COLS] = scaler.fit_transform(df[SCALED_COLS + GENERATED_COLS].values)
    return df.astype('float32')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_index', type=int, default=10)
    parser.add_argument('--input_path', type=str, default='../datasets/frequency_encoded_data')
    parser.add_argument('--output_path', type=str, default='torch_ready_data')
    parser.add_argument('--data_subset', type=int, default=60, help='Number of files to read from input path')
    return parser.parse_args()


def encode_counts(df):
    
    user_count, project_count, country_count = (
        df['user_id'].value_counts().reset_index().rename(columns={'user_id': 'user_count', 'index': 'user_id'}),
        df['project_id'].value_counts().reset_index().rename(columns={'project_id': 'project_count', 'index': 'project_id'}),
        df['country'].value_counts().reset_index().rename(columns={'country': 'country_count', 'index': 'country'})
    )
   
    user_count['user_id_hash'] = user_count.index.values + 1
    project_count['project_id_hash'] = project_count.index.values + 1
    country_count['country_hash'] = country_count.index.values + 1

    df = df.merge(user_count, on='user_id')
    df = df.merge(project_count, on='project_id')
    df = df.merge(country_count, on='country')
    return df
   
def time_encodings(df):
    """
    Timestamp raw encoded in units of seconds
    """
    df['date_time'] = dd.to_datetime(df['date_time'])
    df['timestamp_raw'] = df['date_time'].astype('int64') // 10**9
    return df

def rolling_window_session_10(df, logger):
    logger.info('Calculating rolling session time averages')
    df = df.reset_index()
    df['row_count'] = df.index.values
    rolling_10 = df.set_index('row_count') \
        .groupby(['user_id', 'session_30']) \
        .rolling(10, min_periods=1)['delta_last_event'].mean() \
        .reset_index().rename(columns={'delta_last_event': 'rolling_10'}) \
        .sort_values(by='row_count')

    logger.info('Rolling averages calculated: joining to df')
    df = df.set_index('row_count').join(rolling_10[['row_count', 'rolling_10']].set_index('row_count'))
    logger.info('Rolling averages joined to df')
    df = df.sort_values(by='date_time')
    return df


def expanding_session_time_delta(df, logger):
    logger.info('Calculating expanding session time averages')
    df = df.reset_index()
    df['row_count'] = df.index.values
    expanding_window = df.set_index('row_count') \
        .groupby(['user_id', 'session_30']) \
        .expanding(min_periods=1)['delta_last_event'].mean() \
        .reset_index().rename(columns={'delta_last_event': 'expanding_click_average'}) \
        .sort_values(by='row_count')
    
    logger.info('Expanding averages calculated: joining to df')
    df = df.set_index('row_count').join(expanding_window[['row_count', 'expanding_click_average']].set_index('row_count'))
    logger.info('Expanding averages joined to df')
    df = df.sort_values(by='date_time')
    return df

def intra_session_stats(df, logger):
    
    logger.info('Bringing df to memory')
    df = df.compute() 
    df = df.sort_values(by=['date_time', 'user_id'])
    
    df = df.drop_duplicates(subset=['user_id', 'date_time'], keep='first')
    logger.info('Calculating cum_event_count')
    df['cum_session_event_count'] = df.groupby(['user_id', 'session_30'])['date_time'].cumcount() + 1
    logger.info('Cum_event_count calculated: calculating delta_last_event')
    df['delta_last_event'] = df.groupby(['user_id', 'session_30'])['date_time'].diff()
    df['delta_last_event'] = df['delta_last_event'].dt.total_seconds()
    df['delta_last_event'] = df['delta_last_event'].fillna(0)
    df['cum_session_time_seconds'] = df.groupby(['user_id', 'session_30'])['delta_last_event'].cumsum()
    logger.info('Beginning rolling window 10 calculation')
    df = rolling_window_session_10(df, logger)
    logger.info('Rolling window 10 calculation complete: beginning expanding window calculation')
    df = expanding_session_time_delta(df, logger)
    logger.info('Expanding window calculation complete: returning to dask')
    return df

def main(args):
    #
    torch.set_printoptions(sci_mode=False)
    torch.set_printoptions(precision=4)

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    logger =  get_logger()
    logger.info(f'Running feature calculation with args')
    logger.info(pformat(args.__dict__))

    files = glob.glob(f'{args.input_path}/*.parquet')
    files = list(sorted(files))
    logger.info(f'Found {len(files)} files in {args.input_path}: {files[:args.data_subset]}')
    df = dd.read_parquet(files[:args.data_subset], usecols=INITIAL_LOAD_COLUMNS)
    logger.info(f'Loaded data: shape = {df.shape}, min_date, max_date: {df.date_time.min()}, {df.date_time.max()}')
    df = df[INITIAL_LOAD_COLUMNS]
    df['date_time'] = dd.to_datetime(df['date_time'])
    logger.info(f'Sorting data by date_time')
    df = df.sort_values(by='date_time')
    logger.info('Finished sorting data: encoding value counts')
    df = encode_counts(df)
    logger.info('Finished encoding value counts: encoding time features')
    df = time_encodings(df) 
    
    logger.info('Time encodings complete: encoding categorical features')
    
    df['country'] = df['country'].astype('category')
    df['project_id'] = df['project_id'].astype('category')
    df['user_id'] = df['user_id'].astype('category')
    
    
    df = intra_session_stats(df, logger)
    sample_df = df[df['user_id'] == 2371513].sort_values(by='date_time')
    print(sample_df[['user_id', 'session_30', 'date_time', 'delta_last_event', 'rolling_10', 'expanding_click_average']].head(100))
    


    # df = prepare_for_sessionization(files, MinMaxScaler())
    # print('Dataframe columns:')
    # pp.pprint(list(df.columns))

    # for seq in args.seq_list:
    #     output_path = os.path.join(
    #         args.output_path,
    #         f'files_used_{args.data_subset}',
    #     )
    #     sessionize = SessionizeData(df, seq, output_path, save_s3=args.save_s3)
    #     sessionize.generate_sequence()
    #     print(f'Finished writing {seq} to disk')
    # print(f'Exiting application...')




class Arguments:
    def __init__(self):
        self.input_path = 'datasets/labelled_session_count_data/'
        self.output_path = 'datasets/calculated_features/'
        self.data_subset = 5


if __name__ == "__main__":
    args = Arguments()
    main(args)
