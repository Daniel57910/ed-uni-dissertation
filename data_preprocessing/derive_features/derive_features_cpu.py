im port argparse
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

def encode_counts(df, logger):

    logger.info('Encoding country counts')
    country_count = df['country'].value_counts().reset_index(name='country_count').rename(columns={'index': 'country'})
 
    logger.info('Encoding counts complete: joining users to df')
    df = df.merge(country_count, on='country', how='left')
    return df
   
def time_encodings(df):
    """
    Timestamp raw encoded in units of seconds
    """
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['timestamp_raw'] = df['date_time'].astype(np.int64) // 10 ** 9
    df['date_hour'] = df['date_time'].dt.hour + df['date_time'].dt.minute / 60
    
    df['date_hour_sin'] = np.sin(2 * np.pi * df['date_hour'] / 24)
    df['date_hour_cos'] = np.cos(2 * np.pi * df['date_hour'] / 24)
    

    return df



def rolling_time_between_events_session(df, logger):
    logger.info('Calculating expanding session time averages')
    df = df.reset_index()
    df['row_count'] = df.index.values
    expanding_window = df.set_index('row_count') \
        .sort_values(by=['date_time']) \
        .groupby(['user_id', 'session_30_count']) \
        .rolling(10, min_periods=1)['delta_last_event'].mean() \
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
    df['cum_session_event_count'] = df.groupby(['user_id', 'session_30_count'])['date_time'].cumcount() + 1
    logger.info('Cum_event_count calculated: calculating delta_last_event on cpu')
    df['delta_last_event'] = df.groupby(['user_id', 'session_30_count'])['date_time'].diff()

    df = df.sort_values(by=['date_time', 'user_id'])
    # df = df.to_pandas(
    df['delta_last_event'] = df['delta_last_event'].dt.total_seconds()
    df['delta_last_event'] = df['delta_last_event'].fillna(0)
    # df = pd.from_pandas(df)
    df = df.sort_values(by=['date_time', 'user_id'])
    df['cum_session_time_minutes'] = df.groupby(['user_id', 'session_30_count'])['delta_last_event'].cumsum()
    df['cum_session_time_minutes'] = df['cum_session_time_minutes'] / 60
    logger.info('Beginning rolling window 10 calculation')
    logger.info('Rolling window 10 calculation complete: beginning expanding window calculation')
    logger.info('Expanding window calculation complete: returning to dask')
    return df

def running_user_stats(df, logger):
    logger.info('Calculating cumulative platform time')
    df = df.sort_values(by=['date_time'])
    df['cum_platform_time_minutes'] = df.groupby(['user_id'])['delta_last_event'].cumsum()
    df['cum_platform_events'] = df.groupby(['user_id']).cumcount() + 1
    df['cum_platform_time_minutes'] = df['cum_platform_time_minutes'] / 60
    logger.info('Calculating cumulative platform events')
    logger.info('Calculated cumulative platform events: calculating running unique projects')
    
    logger.info('Using GPU: converting to pandas')
    logger.info('Calculating running unique projects: shifting projects to find unique')
    
    df['project_id'] = df['project_id'].astype(int)
    df['user_id'] = df['user_id'].astype(int)
    df['previous_user_project'] = df.groupby('user_id')['project_id'].shift(1)
    df['previous_project_exists'] = df['previous_user_project'].notna()
    
    df['previous_user_project'] = df[['previous_user_project', 'previous_project_exists', 'project_id']].apply(
        lambda x: x['previous_user_project'] if x['previous_project_exists'] else x['project_id'], axis=1)
    logger.info('Calculating running unique projects: calculating unique projects')
    
    df['previous_user_project'] = df['previous_user_project'].astype(int)
    df['project_change'] = df['project_id'] != df['previous_user_project']
    
    df['cum_projects'] = df.groupby('user_id')['project_change'].cumsum() + 1
    
   
    df = df.drop(columns=['previous_user_project', 'previous_project_exists', 'project_change'])
    logger.info('Calculated running unique projects: calculating average event time delta')
    df = df.reset_index()
    df['row_count'] = df.index.values
    
    average_event_time = df.set_index('row_count') \
        .sort_values(by=['date_time']) \
        .groupby('user_id') \
        .rolling(1000, min_periods=1)['delta_last_event'].mean() \
        .reset_index().rename(columns={'delta_last_event': 'average_event_time'}) \
        .sort_values(by='row_count')
    df = df.set_index('row_count').join(average_event_time[['row_count', 'average_event_time']].set_index('row_count'))
    logger.info('Calculated average event time delta')
    return df


def time_from_previous_session_minutes(session_inflection_times, logger):
    session_inflection_times = session_inflection_times.sort_values(by=['session_30_count', 'user_id'])
    
    session_inflection_times['previous_session_end'] = session_inflection_times.groupby(['user_id'])['date_time_max'].shift(1)
    session_inflection_times['time_between_session_minutes'] = (session_inflection_times['date_time_min'] - session_inflection_times['previous_session_end']).dt.total_seconds() / 60
    session_inflection_times['time_between_session_minutes'] = session_inflection_times['time_between_session_minutes'].fillna(0)
    return session_inflection_times[['user_id', 'session_30_count', 'time_between_session_minutes', 'date_time_min', 'date_time_max']]

def rolling_average_session_statistics(df, session_inflection_times, logger):
 
    logger.info('Session inflection times calculated: calculating expanding session time')
    average_session_minutes = session_inflection_times.sort_values(by=['session_30_count', 'user_id']) \
    .set_index(['session_30_count', 'date_time_min', 'date_time_max']) \
    .groupby(['user_id']) \
    ['session_time_minutes'] \
    .rolling(10, min_periods=1, closed='left') \
    .mean() \
    .reset_index() \
    .rename(columns={'session_time_minutes': 'rolling_session_time'})
   
    average_session_minutes['rolling_session_time'] = average_session_minutes['rolling_session_time'].fillna(0)
    logger.info('Calculating average events per session')
    average_events_session = session_inflection_times.sort_values(by=['session_30_count', 'user_id']) \
        .set_index(['session_30_count', 'date_time_min', 'date_time_max']) \
        .groupby(['user_id']) \
        ['session_event_count'] \
        .rolling(10, min_periods=1, closed='left') \
        .mean() \
        .reset_index() \
        .rename(columns={'session_event_count': 'rolling_session_events'})
    
    average_events_session['rolling_session_events'] = average_events_session['rolling_session_events'].fillna(0)
    
    logger.info('Calculating time from previous session')
    time_between_session = time_from_previous_session_minutes(session_inflection_times, logger)
    
    time_between_session = time_between_session.sort_values(by=['session_30_count', 'user_id']) \
        .set_index(['session_30_count']) \
        .groupby(['user_id']) \
        ['time_between_session_minutes'] \
        .rolling(5, min_periods=1) \
        .mean() \
        .reset_index() \
        .rename(columns={'time_between_session_minutes': 'rolling_session_gap'})

    logger.info('Joining dataframes')
   
    session_stats = pd.merge(average_events_session, average_session_minutes, on=['user_id', 'session_30_count']) 
    session_stats = pd.merge(session_stats, session_inflection_times, on=['user_id', 'session_30_count'])
    session_stats = pd.merge(session_stats, time_between_session, on=['user_id', 'session_30_count'])
    # session_stats = pd.from_pandas(session_stats)
    
    df = pd.merge(df, session_stats[['user_id', 'session_30_count', 'rolling_session_time', 'rolling_session_events', 'session_event_count', 'session_time_minutes', 'rolling_session_gap']], on=['user_id', 'session_30_count'])
    logger.info('Dataframes joined::returning')
    return df


def assign_metadata(df, logger):
    logger.info(f'Obtaining global session time and user events')
    global_session_time = df.groupby('user_id')['cum_platform_time_minutes'].max().reset_index().rename(columns={'cum_platform_time_minutes': 'global_session_time_minutes'})
    user_count = df['user_id'].value_counts().reset_index(name='global_events_user').rename(columns={'index': 'user_id'})
    
    logger.info('Joining session_time to df')
    df = pd.merge(df, global_session_time, on='user_id', how='left')
    logger.info('Joining user_count to df')
    df = pd.merge(df, user_count, on='user_id', how='left')
   
    df['cum_session_event_raw'] = df['cum_session_event_count']
    df['cum_platform_event_raw'] = df['cum_platform_events']
    df['session_30_count_raw'] = df['session_30_count']
    logger.info('Assigning date_metadata')
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    return df
    
    
def hash_user_id(df):
    
    user_id = df[['user_id']].drop_duplicates().reset_index(names='user_id_hash')
    df = pd.merge(df, user_id[['user_id_hash', 'user_id']], on='user_id')
    df = df.drop(columns=['user_id'])
    df = df.rename(columns={'user_id_hash': 'user_id'})
    df = df.sort_values(by=['date_time']).reset_index(drop=True)
    return df

def generate_summary_session_stats(df, logger):
    logger.info('Generating session statistics')
    session_inflection_statistics = df.groupby(['user_id', 'session_30_count']).agg({'date_time': ['min', 'max', 'count']}).reset_index()
    session_inflection_statistics.columns = ['user_id', 'session_30_count', 'date_time_min', 'date_time_max', 'session_event_count']
    # session_inflection_statistics = session_inflection_statistics.to_pandas()
    session_inflection_statistics['session_time_minutes'] = (session_inflection_statistics['date_time_max'] - session_inflection_statistics['date_time_min']).dt.total_seconds() / 60
    return session_inflection_statistics

 
def _pretty_print_columns(df):
    for col in df.columns:
        print(f'    "{col}"')
def main(args):
    #

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    logger =  get_logger()
    logger.info(f'Running feature calculation with args')
    logger.info(pformat(args.__dict__))


    
    logger.info(f'Loading data from {args.input_path}')
    df = pd.read_parquet(args.input_path, columns=INITIAL_LOAD_COLUMNS)
    df = hash_user_id(df)
    logger.info(f'Loaded data: shape = {df.shape}, min_date, max_date: {df.date_time.min()}, {df.date_time.max()}')
    label_count = df[df['session_terminates_30_minutes'] == True].shape[0] / df.shape[0]
    logger.info(f'Perc ending in 30 minutes: {label_count}')
    df['date_time'] = pd.to_datetime(df['date_time'])
    logger.info(f'Sorting data by date_time')
    df = df.sort_values(by='date_time')
    logger.info('Finished sorting data: encoding value counts')
    df = encode_counts(df, logger)
    logger.info('Finished encoding value counts: encoding time features')
    df = time_encodings(df) 
   
    logger.info('Time encodings complete: encoding categorical features')
    
    
    logger.info('Categorical features encoded: calculating intra-session stats')
    df = intra_session_stats(df, logger)
    logger.info('Beginning rolling window 10 calculation')
    
    df = rolling_time_between_events_session(df, logger)
    logger.info('Rolling window 10 calculation complete: beginning expanding window calculation')
    
    logger.info(f'Calculating running user stats')
    df = running_user_stats(df, logger)
    logger.info('Calculating between session stats')
   

    session_inflection_times = generate_summary_session_stats(df, logger)
    logger.info('Session inflection times calculated: columns')
    logger.info(pformat(session_inflection_times.columns))
    df = rolling_average_session_statistics(df, session_inflection_times, logger)
    logger.info('Time within session and average session clicks calculated:: calculating time between session')
    df['session_30_raw'] = df['session_30_count']

    logger.info('Assigning metadata')
    df = assign_metadata(df, logger)
    logger.info('Metadata assigned: dropping columns')
    
    logger.info('Returning df to dask for writing to disk')
       
    output_path = os.path.join(args.output_path, f'files_used_{args.data_subset}')
    logger.info(f'Writing to {output_path}')
    
    logger.info(f'df converted to dask: shape -> {df.shape}')
    logger.info(f'Final out columns:')
    _pretty_print_columns(df)

    df = df.sort_values(by='date_time').reset_index(drop=True).to_parquet(output_path)

    logger.info('Finished writing to disk')


class Arguments:
    def __init__(self):
        self.input_path = 'datasets/labelled_session_count_data/files_used_2'
        self.output_path = 'datasets/calculated_features/'
        self.data_subset = 2


if __name__ == "__main__":
    args = Arguments()
    main(args)