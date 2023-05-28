import pandas as pd
import logging

global logger
logger = logging.getLogger('rl_results_eval')
from functools import reduce

def remove_events_in_minute_window(df):
    df['second_window'] = df['second'] // 10
    df = df.drop_duplicates(
        subset=['user_id', 'session_30_raw', 'year', 'month', 'day', 'hour', 'minute'],
        keep='last'
    ).reset_index(drop=True)

    return df


def convolve_delta_events(df, window):
    df['convolved_delta_event'] = (
        df.set_index('date_time').groupby(by=['user_id', 'session_30_raw'], group_keys=False) \
            .rolling(f'{window}T', min_periods=1)['delta_last_event'] \
            .mean()
            .reset_index(name='convolved_event_delta')['convolved_event_delta']
    )

    df['delta_last_event'] = df['convolved_delta_event']

    return df

def generate_metadata(dataset):
    
    session_size = dataset.groupby(['user_id', 'session_30_raw'])['size_of_session'].max().reset_index(name='session_size')
    session_minutes = dataset.groupby(['user_id', 'session_30_raw'])['cum_session_time_raw'].max().reset_index(name='session_minutes')
    
    sim_minutes = dataset.groupby(['user_id', 'session_30_raw'])['cum_session_time_raw'].quantile(.7, interpolation='nearest').reset_index(name='sim_minutes')
    sim_size = dataset.groupby(['user_id', 'session_30_raw'])['cum_session_event_raw'].quantile(.7, interpolation='nearest').reset_index(name='sim_size')
    
    
    sessions = [session_size, session_minutes, sim_minutes, sim_size]
    sessions = reduce(lambda left, right: pd.merge(left, right, on=['user_id', 'session_30_raw']), sessions)
    dataset = pd.merge(dataset, sessions, on=['user_id', 'session_30_raw'])
    dataset['reward'] = dataset['cum_session_time_raw']
    return dataset



def setup_data_at_window(df, window):
    df['date_time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']], errors='coerce')
    size_of_session = df.groupby(['user_id', 'session_30_raw']).size().reset_index(name='size_of_session')
    df = pd.merge(df, size_of_session, on=['user_id', 'session_30_raw'])
    df['cum_session_event_raw'] = df.groupby(['user_id', 'session_30_raw'])['date_time'].cumcount() + 1
    
    logger.info(f'Convolution over {window} minute window')
    df = convolve_delta_events(df, window)
    logger.info(f'Convolving over {window} minute window complete: generating metadata')
    df = generate_metadata(df) 
    logger.info(f'Metadata generated: selecting events only at {window} minute intervals')
    df = df[df['minute'] % window == 0]
    logger.info(f'Data read: {df.shape[0]} rows, {df.shape[1]} columns, dropping events within 2 minute window')
    df = remove_events_in_minute_window(df)
    df = df.reset_index(drop=True)
    
    logger.info(f'Number of events after dropping events within 2 minute window: {df.shape[0]}')
    
    df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'second_window'])