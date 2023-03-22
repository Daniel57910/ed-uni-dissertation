from derive_features_cpu import encode_counts
import pytest
import glob
import dask.dataframe as dd
from constant import INITIAL_LOAD_COLUMNS
import logging
logging.basicConfig(level=logging.INFO)
import random 
import pandas as pd
from derive_features_cpu import (
    time_encodings, encode_counts, intra_session_stats
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


def test_encode_counts(load_entry_df):
    df = encode_counts(load_entry_df)
    df = df.compute()
    random_sample = df.iloc[random.randint(0, len(df))]
    
    user_count, project_count, country_count = random_sample['user_count'], random_sample['project_count'], random_sample['country_count']
    assert user_count == df[df['user_id'] == random_sample['user_id']].shape[0]
    assert project_count == df[df['project_id'] == random_sample['project_id']].shape[0]
    assert country_count == df[df['country'] == random_sample['country']].shape[0]     

def test_time_encodings(load_entry_df):
    df = time_encodings(load_entry_df)
    df = df.compute()
    random_sample = df.iloc[random.randint(0, len(df))]
    
    date_time, timestamp_raw = random_sample['date_time'], random_sample['timestamp_raw']
    assert date_time == pd.to_datetime(timestamp_raw, unit='s')

def test_within_session_stats(load_entry_df):
    encoded_df = encode_counts(load_entry_df)
    encoded_df = time_encodings(encoded_df)
    
    df = intra_session_stats(encoded_df, logger)
    
    sample = df.iloc[random.randint(0, len(df))]
    
    sample_session = df[(
        df['user_id_hash'] == sample['user_id_hash']) &
        (df['session_30'] == sample['session_30'])]

    sample_session = sample_session.sort_values(by='date_time')
   
    assert sample_session['cum_session_time_seconds'].max() == \
        (sample_session['date_time'].max() - sample_session['date_time'].min()).total_seconds()
    
    
    assert sample_session['cum_session_event_count'].max() == sample_session.shape[0]
    print("\n\n")
    print(sample_session[['date_time', 'delta_last_event', 'rolling_10']].head(20))
   
    sample_last = sample_session.iloc[-1]['rolling_10']
    rolling_last = sample_session.iloc[-10:-1]['delta_last_event'].mean()
 
    """
    Allow 10 second time window range
    """
    assert abs(sample_last - rolling_last) < 10
    