import argparse
import glob
import os

from pprint import pformat
USE_GPU = False


import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=200)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from constant import (
    INITIAL_LOAD_COLUMNS
)
def get_logger():
    logger = logging.getLogger(__name__)
    return logger

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
    
    df = df.drop(columns=['user_id', 'project_id', 'country'])
    df = df.rename(columns={'user_id_hash': 'user_id', 'project_id_hash': 'project_id', 'country_hash': 'country'})
    return df
   
def time_encodings(df):
    """
    Timestamp raw encoded in units of seconds
    """
    df['date_time'] = pd.to_datetime(df['date_time'])
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
        .rolling(1000000, min_periods=1)['delta_last_event'].mean() \
        .reset_index().rename(columns={'delta_last_event': 'expanding_click_average'}) \
        .sort_values(by='row_count')
    
    logger.info('Expanding averages calculated: joining to df')
    df = df.set_index('row_count').join(expanding_window[['row_count', 'expanding_click_average']].set_index('row_count'))
    logger.info('Expanding averages joined to df')
    df = df.sort_values(by='date_time')
    return df

def intra_session_stats(df, logger):
    
    logger.info('Sorting by date_time and user_id')
    df = df.sort_values(by=['date_time', 'user_id'])
    
    df = df.drop_duplicates(subset=['user_id', 'date_time'], keep='first')
    logger.info('Calculating cum_event_count')
    df['cum_session_event_count'] = df.groupby(['user_id', 'session_30'])['date_time'].cumcount() + 1
    logger.info('Cum_event_count calculated: calculating delta_last_event')
    df['delta_last_event'] = df.groupby(['user_id', 'session_30'])['date_time'].diff()

    df = df.sort_values(by=['date_time', 'user_id'])
    # df = df.to_pandas().sort_values(by=['date_time', 'user_id'])
    df['delta_last_event'] = df['delta_last_event'].dt.total_seconds()
    df['delta_last_event'] = df['delta_last_event'].fillna(0)
    # df = pd.from_pandas(df)
    df = df.sort_values(by=['date_time', 'user_id'])
    df['cum_session_time_minutes'] = df.groupby(['user_id', 'session_30'])['delta_last_event'].cumsum()
    df['cum_session_time_minutes'] = df['cum_session_time_minutes'] / 60
    logger.info('Beginning rolling window 10 calculation')
    logger.info('Rolling window 10 calculation complete: beginning expanding window calculation')
    logger.info('Expanding window calculation complete: returning to dask')
    return df

import pdb
def running_user_stats(df, logger):
    logger.info('Calculating cumulative platform time')
    df['cum_platform_time_minutes'] = df.groupby(['user_id'])['delta_last_event'].cumsum()
    df['cum_platform_time_minutes'] = df['cum_platform_time_minutes'] / 60
    logger.info('Calculating cumulative platform events')
    df['cum_platform_events'] = df.groupby(['user_id'])['delta_last_event'].cumcount() + 1
    logger.info('Calculated cumulative platform events: calculating running unique projects')
    
    logger.info('Using GPU: converting to pandas')
    logger.info('Calculating running unique projects: shifting projects to find unique')
    
    df['previous_user_project'] = df.groupby('user_id')['project_id'].shift(1)
    logger.info('Calculating running unique projects: calculating unique projects')
    df['previous_user_project'] = df[['project_id', 'previous_user_project']].apply(
        lambda x: x['project_id'] if pd.isna(x['previous_user_project']) else x['previous_user_project'], 
        axis=1)
    
    df['previous_user_project'] = df['previous_user_project'].astype(int)
    df['project_change'] = df[['project_id', 'previous_user_project']].apply(lambda x: 1 if x[0] != x[1]  else 0, axis=1)
    
    df['cum_projects'] = df.groupby('user_id')['project_change'].cumsum()
    df['cum_projects'] = df['cum_projects'] + 1
    

    
    logger.info('Calculated running unique projects: calculating average event time delta')
    df = df.reset_index()
    df['row_count'] = df.index.values
    
    average_event_time = df.set_index('row_count') \
        .groupby('user_id') \
        .rolling(10000000, min_periods=1)['delta_last_event'].mean() \
        .reset_index().rename(columns={'delta_last_event': 'average_event_time'}) \
        .sort_values(by='row_count')
    df = df.set_index('row_count').join(average_event_time[['row_count', 'average_event_time']].set_index('row_count'))
    logger.info('Calculated average event time delta')
    return df

def expanding_session_time(df, session_inflection_times, logger):
 
    logger.info('Session inflection times calculated: calculating expanding session time')
    # session_inflection_times = session_inflection_times.to_pandas()
    session_inflection_times['session_time_mins'] = (session_inflection_times['date_time_max'] - session_inflection_times['date_time_min']).dt.total_seconds() / 60
   

    session_inflection_times['expanding_session_time_minutes'] = session_inflection_times.set_index(['date_time_min', 'session_30']) \
        .groupby(['user_id'])['session_time_mins'] \
        .rolling(10000000, min_periods=1, closed='left') \
        .sum() \
        .reset_index() \
        .rename(columns={'session_time_mins': 'expanding_session_time_minutes'})['expanding_session_time_minutes']
    


    session_inflection_times['session_30_divisor'] = session_inflection_times['session_30'].apply(lambda x: max(1, x-1))
    session_inflection_times['expanding_session_time_minutes'] = (session_inflection_times['expanding_session_time_minutes'] / session_inflection_times['session_30_divisor']).fillna(0)
    
    
    # # session_inflection_times = pd.from_pandas(session_inflection_times)
    
    logger.info('Expanding session time calculated: joining to df')
  
    df = pd.merge(df, session_inflection_times[['user_id', 'session_30',  'expanding_session_time_minutes', 'session_time_mins']], on=['user_id', 'session_30'], how='left')
    df['expanding_session_time_minutes'] = df['expanding_session_time_minutes'].fillna(0)
    logger.info('Expanding session time joined to df')
    return df

def time_between_sessions(df, session_min_max, logger):

    session_min_max = session_min_max.sort_values(by=['date_time_min', 'user_id'])
    session_min_max['previous_session_end'] = session_min_max.groupby(['user_id'])['date_time_max'].shift(1)
    
    session_min_max['previous_session_exists'] = session_min_max['previous_session_end'].notnull()

    session_min_max['previous_session_end'] = session_min_max[['previous_session_exists', 'previous_session_end', 'date_time_min']] \
        .apply(lambda x: x['date_time_min'] if not x['previous_session_exists'] else x['previous_session_end'], axis=1)

    # session_min_max = session_min_max.to_pandas()
    logger.info('Calculating time between sessions minutes on cpu')
    session_min_max['time_between_sessions_minutes'] = (session_min_max['date_time_min'] - session_min_max['previous_session_end']).dt.total_seconds() / 60
  
    session_min_max = session_min_max.drop(columns=['previous_session_exists'])
    session_min_max = session_min_max.set_index(['date_time_min', 'session_30']) \
        .groupby('user_id')['time_between_sessions_minutes'] \
        .rolling(10000000, min_periods=1, closed='right') \
        .sum() \
        .reset_index() \
        .rename(columns={'time_between_sessions_minutes': 'expanding_session_delta_minutes'}) \
        
    session_min_max['expanding_session_delta_minutes'] = (session_min_max['expanding_session_delta_minutes'] / (session_min_max['session_30'] - 1)).fillna(0)

    sample = session_min_max[session_min_max['session_30'] > 4]['user_id'].unique()[0]
    sample_df = session_min_max[session_min_max['user_id'] == sample]
    # session_min_max = pd.from_pandas(session_min_max) 
    df = pd.merge(df, session_min_max[['user_id', 'session_30', 'expanding_session_delta_minutes']], on=['user_id', 'session_30'], how='left') 
    df['expanding_session_delta_minutes'] = df['expanding_session_delta_minutes'].fillna(0)
       
    logger.info('Time between sessions joined to df')
    df = df.sort_values(by='date_time')
    
    logger.info('DF resorted by date_time')
    return df

def assign_metadata(df, logger):
    logger.info(f'Obtaining global session time and user events')
    global_events_user = df.groupby('user_id')['cum_platform_events'].max().reset_index().rename(columns={'cum_platform_events': 'global_events_user'})
    global_session_time = df.groupby('user_id')['cum_platform_time_minutes'].max().reset_index().rename(columns={'cum_platform_time_minutes': 'global_session_time_minutes'})
    
    metadata = global_events_user.set_index('user_id').join(global_session_time.set_index('user_id'))
     
    logger.info('Joining metadata to df')
    df = pd.merge(df, metadata, on='user_id', how='left')
    return df
    
def main(args):
    #

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    logger =  get_logger()
    logger.info(f'Running feature calculation with args')
    logger.info(pformat(args.__dict__))

    files = glob.glob(f'{args.input_path}/*.parquet')
    files = list(sorted(files))
    logger.info(f'Found {len(files)} files in {args.input_path}')
    logger.info(pformat(f'Loading data from'))
    logger.info(pformat(files[:args.data_subset]))
    df = pd.read_parquet(files[:args.data_subset], columns=INITIAL_LOAD_COLUMNS)
    logger.info(f'Loaded data: shape = {df.shape}, min_date, max_date: {df.date_time.min()}, {df.date_time.max()}')
    df['date_time'] = pd.to_datetime(df['date_time'])
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
    
    
    logger.info('Categorical features encoded: calculating intra-session stats')
    df = intra_session_stats(df, logger)
    logger.info('Beginning rolling window 10 calculation')
    
    df = rolling_window_session_10(df, logger)
    logger.info('Rolling window 10 calculation complete: beginning expanding window calculation')
    
    df = expanding_session_time_delta(df, logger)
    logger.info('Expanding window calculation complete: returning to dask')
    
    logger.info(f'Calculating running user stats')
    # df = running_user_stats(df, logger)
    
    logger.info('Calculating between session stats')
   
    session_inflection_times = df.groupby(['user_id', 'session_30']).agg({'date_time': ['min', 'max']}).reset_index()
    session_inflection_times.columns = session_inflection_times.columns.map('_'.join).str.strip()

    session_inflection_times = session_inflection_times.rename(columns={'user_id_': 'user_id', 'session_30_': 'session_30'})
    logger.info(f'session_inflection_times columns: {session_inflection_times.columns}')
    # return 
    session_inflection_times = session_inflection_times.sort_values(by=['date_time_min', 'user_id'])
    
    logger.info('Session inflection times calculated: columns')
    logger.info(pformat(session_inflection_times.columns))
    logger.info('Calculating time within session')
    df = expanding_session_time(df, session_inflection_times.copy(), logger)
    return
    logger.info('Calculating time between sessions')
    print(session_inflection_times.columns)
    df = time_between_sessions(df, session_inflection_times.copy(), logger)
    df['session_30_raw'] = df['session_30']

    logger.info('Assigning metadata')
    df = assign_metadata(df, logger)
    logger.info('Metadata assigned: dropping columns')
    
    logger.info('Returning df to dask for writing to disk')
       
    output_path = os.path.join(args.output_path, f'files_used_{args.data_subset}')
    logger.info(f'Writing to {output_path}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    df = df.rename(columns={'label_30': 'session_terminates_30_minutes'})
    for col in df.columns:
        if df[col].dtype == 'category':
            logger.info(f'Converting {col} to int')
            df[col] = df[col].astype(int)
   
    df = dd.from_pandas(df, npartitions=args.data_subset)
    
    logger.info(f'df converted to dask: shape -> {df.shape}')
    logger.info(f'Final out columns:')
    logger.info(pformat(df.columns))
    df = df.sort_values(by='date_time').reset_index(drop=True).to_parquet(output_path)

    logger.info('Finished writing to disk')

class Arguments:
    def __init__(self):
        self.input_path = 'datasets/labelled_session_count_data/'
        self.output_path = 'datasets/calculated_features/'
        self.data_subset = 1 


if __name__ == "__main__":
    args = Arguments()
    main(args)
