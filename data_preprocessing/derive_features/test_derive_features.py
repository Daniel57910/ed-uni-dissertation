from derive_features_cpu import encode_counts
import pytest
import glob
import dask.dataframe as dd
from constant import INITIAL_LOAD_COLUMNS
from typing import Any
import logging
logging.basicConfig(level=logging.INFO)
import random 
import pandas as pd
from derive_features_cpu import (
    time_encodings, encode_counts, running_user_stats, intra_session_stats,
    rolling_window_session_10, expanding_session_time_delta, expanding_session_time, 
    time_between_sessions
)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
import numpy as np
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def load_entry_df():
    files = glob.glob(f'datasets/labelled_session_count_data/*.parquet')
    files = list(sorted(files))
    df = dd.read_parquet(files[:2], usecols=INITIAL_LOAD_COLUMNS)
    df = df[INITIAL_LOAD_COLUMNS]
    df['date_time'] = dd.to_datetime(df['date_time'])
    df = df.sort_values(by='date_time')
    return df


def test_encode_counts(load_entry_df: Any):
    df = encode_counts(load_entry_df)
    df = df.compute()
    random_sample = df.iloc[random.randint(0, len(df))]
    
    user_count, project_count, country_count = random_sample['user_count'], random_sample['project_count'], random_sample['country_count']
    assert user_count == df[df['user_id'] == random_sample['user_id']].shape[0]
    assert project_count == df[df['project_id'] == random_sample['project_id']].shape[0]
    assert country_count == df[df['country'] == random_sample['country']].shape[0]     

def test_time_encodings(load_entry_df: Any):
    df = time_encodings(load_entry_df)
    df = df.compute()
    random_sample = df.iloc[random.randint(0, len(df))]
    
    date_time, timestamp_raw = random_sample['date_time'], random_sample['timestamp_raw']
    assert date_time == pd.to_datetime(timestamp_raw, unit='s')

def test_within_session_stats(load_entry_df: Any):
    encoded_df = encode_counts(load_entry_df)
    encoded_df = time_encodings(encoded_df)
    
    df = intra_session_stats(encoded_df, logger)
    
    df = rolling_window_session_10(df, logger)
    
    df = expanding_session_time_delta(df, logger)
    
    sample = df.iloc[random.randint(0, len(df))]
    
    sample_session = df[(
        df['user_id_hash'] == sample['user_id_hash']) &
        (df['session_30'] == sample['session_30'])]

    sample_session = sample_session.sort_values(by='date_time')
   
    assert sample_session['cum_session_time_seconds'].max() == \
        (sample_session['date_time'].max() - sample_session['date_time'].min()).total_seconds()
    
    
    assert sample_session['cum_session_event_count'].max() == sample_session.shape[0]
   
    sample_last = sample_session.iloc[-1]['rolling_10']
    rolling_last = sample_session.iloc[-10:]['delta_last_event'].mean()
    
    expanding_last = sample_session.iloc[-1]['expanding_click_average']
    average_delta = sample_session['delta_last_event'].mean()
    
   
    assert abs(sample_last - rolling_last) < 0.1
    assert abs(expanding_last - average_delta) < 0.1

def test_running_user_stats(load_entry_df: Any):
    encoded_df = encode_counts(load_entry_df)
    encoded_df = time_encodings(encoded_df)
    encoded_df = intra_session_stats(encoded_df, logger)
    df = running_user_stats(encoded_df, logger)
    
    sample_user = df.iloc[random.randint(0, len(df))]['user_id_hash']
    sample_df = df[(df['user_id_hash'] == sample_user)]
    
    assert sample_df.iloc[-1]['cum_platform_events'] == sample_df.shape[0]
    assert sample_df.iloc[-1]['cum_platform_time'] == sample_df['delta_last_event'].sum()
    assert sample_df.iloc[-1]['cum_projects'] == sample_df['project_id_hash'].nunique()
    assert sample_df.iloc[-1]['average_event_time'] == sample_df['delta_last_event'].mean()

def test_session_stats(load_entry_df):
    encoded_df = encode_counts(load_entry_df)
    encoded_df = time_encodings(encoded_df)
    encoded_df = intra_session_stats(encoded_df, logger)
    session_inflection_times = encoded_df \
        .groupby(['user_id', 'session_30']) \
        .agg({'date_time': ['min', 'max']}) \
        .reset_index() \
        .rename(columns={'min': 'session_start', 'max': 'session_end'})
        
    session_inflection_times.columns = session_inflection_times.columns.map('_'.join).str.strip()
    session_inflection_times = session_inflection_times.rename(columns={'user_id_': 'user_id', 'session_30_': 'session_30'})

    encoded_df = expanding_session_time(encoded_df, session_inflection_times, logger)
    
    user_more_one_session = \
        encoded_df[encoded_df['session_30'] > 2]['user_id'].unique()
    
    random_user = user_more_one_session[random.randint(0, len(user_more_one_session))]
    
    sample_df = encoded_df[encoded_df['user_id'] == random_user]
    first_session = sample_df[sample_df['session_30'] == 1]

    assert first_session['expanding_session_time_minutes'].max() == 0
    sample_df = sample_df[['user_id', 'session_30', 'expanding_session_time_minutes', 'time_in_session_minutes']].drop_duplicates()
    last_session = sample_df.iloc[-1]['expanding_session_time_minutes']
    assert last_session  == sample_df['time_in_session_minutes'].iloc[:-1].sum() / (sample_df.shape[0] - 1)
     

def test_time_between_sessions(load_entry_df): 
    encoded_df = encode_counts(load_entry_df)
    encoded_df = time_encodings(encoded_df)
    encoded_df = intra_session_stats(encoded_df, logger)
    session_inflection_times = encoded_df \
        .groupby(['user_id', 'session_30']) \
        .agg({'date_time': ['min', 'max']}) \
        .reset_index() \
        .rename(columns={'min': 'session_start', 'max': 'session_end'})
        
    session_inflection_times.columns = session_inflection_times.columns.map('_'.join).str.strip()
    session_inflection_times = session_inflection_times.rename(columns={'user_id_': 'user_id', 'session_30_': 'session_30'})
    
    df = time_between_sessions(encoded_df, session_inflection_times, logger)
    
    first_session, n_session = df[df['session_30'] == 1], df[df['session_30'] > 1]
    
    np.testing.assert_equal(first_session['expanding_session_delta_minutes'].values, np.zeros(first_session.shape[0]))

    
    
    random_sample = n_session[n_session['session_30'] == 2].iloc[random.randint(0, n_session[n_session['session_30'] == 2].shape[0])]
    random_sample_user, session_delta = random_sample['user_id'], random_sample['expanding_session_delta_minutes']
    
    min_ev_ts = df[(df['user_id'] == random_sample_user) & (df['session_30'] == 1)]['date_time'].max()
    max_ev_ts = df[(df['user_id'] == random_sample_user) & (df['session_30'] == 2)]['date_time'].min()
    
    assert (max_ev_ts - min_ev_ts).total_seconds() / 60 == session_delta

    
