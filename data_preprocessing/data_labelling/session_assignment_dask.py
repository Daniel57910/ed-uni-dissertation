import argparse
import glob
import os
import pdb
import shutil
from datetime import datetime
from random import sample

import numba
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
LABEL_COLS = ['5_within_5_mins', 'within_5']
EXAMPLE_COLS = ['user_id', 'date_time', 'project_id']
CALENDAR_COLS = ['year', 'month', 'day', 'hour', 'minute', 'second']
META_CALENDAR_COLS = ['is_weekend', 'day_of_week']
EVAL_COLUMNS_5_MINS = ['user_id', 'date_time', 'previous_date_time', 'time_diff_seconds', '5_minute_contiguous_session', '5_minute_session_count']
EVAL_COLUMNS_30_MINS = ['user_id', 'date_time', 'previous_date_time', 'time_diff_seconds', '30_minute_contiguous_session', '30_minute_session_count']

COLS_TO_USE = [
    'project_id',
    'workflow_id',
    'subjects_ids',
    'user_id',
    'country',
    'timestamp'
] + CALENDAR_COLS + META_CALENDAR_COLS

PATH_CONSTRAINTS = (200, 210)
TOP_10_USERS = [870816.0, 2482328.0, 2609.0, 2324166.0, 695707.0, 2321181.0, 2404151.0, 2316079.0, 2380660.0, 2404871.0]
SAMPLE_USER = 2324166.0

SAMPLE_USER_SECOND = 2433267
DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def get_inflection_points_5(subset):
    return subset[subset['5_minute_contiguous_session'] == False].index.values

def get_inflection_points_30(subset):
    return subset[subset['30_minute_contiguous_session'] == False].index.values

# numba njit disable gilobal interpreter lock
@numba.njit(nogil=True)
def label_session_windows(index_for_row, inflection_for_user):
    return 1 + np.searchsorted(inflection_for_user, index_for_row, side='right')

def outer_apply_inflection_points(inflections):
    def inner_find_session(row):
        inflection_for_user = inflections[row['user_id']]
        index_for_row = row['row_count']
        return 1 + np.searchsorted(inflection_for_user, index_for_row, side='right')

    return inner_find_session

def reassign_indexes(df):
    df['row_count'] = df.index
    return df

def main():

    current_time = datetime.now()
    print(f"Current Time = {current_time}")

    client = Client()
    print('Starting Dask Client')
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='input_data')
    parser.add_argument('--write_path', type=str, default='contiguous_session_data_optimized')

    args = parser.parse_args()

    read_path, write_path = args.read_path, args.write_path

    paths = list(glob.iglob(f"{read_path}/*.csv"))

    print(f'Read path: {read_path}, Write path: {write_path}')

    print(f'Found {len(paths)} files to process')

    print('Clearing write path')
    shutil.rmtree(write_path, ignore_errors=True)

    paths = sorted(paths)
    print(f'Reading from: {paths[0]} -> {paths[-1]}')
    df = dd.read_csv(paths, usecols=COLS_TO_USE)

    df = df.repartition(npartitions=20)
    df = df.persist()

    df['date_time'] = dd.to_datetime(df[CALENDAR_COLS])
    df = df.sort_values(by='timestamp')
    print(f'Number of rows prior to assignment: {len(df)}')
    """
    convert to pandas due to Exception: "ValueError('cannot reindex on an axis with duplicate labels')"
    batching may be required after
    """
    print(f'Backcalculating timestamps')

    """
    Perform row-wise groupy by user_id
    """
    df = df.groupby('user_id').apply(lambda x: x.assign(previous_date_time=x['date_time'].shift(1)))
    df = df.dropna()
    df = df.sort_values(by='timestamp')

    print('Pushing df into memory: resetting index and row count')
    df = df.compute().reset_index()

    df['row_count'] = df.index

    print('Converting df back to dask')
    df = dd.from_pandas(df, npartitions=60)

    print('DF converted back to dask')
    print(f'Number of rows after assignment: {len(df)}')
    print(df.head(10))

    print('Previous HIT identified: calculating time difference')
    df['time_diff_seconds'] = (df['date_time'] - df['previous_date_time']).apply(lambda x: x.total_seconds(), meta=('time_diff_seconds', 'float64'))

    df['5_minute_contiguous_session'] = df['time_diff_seconds'] < 5 * 60
    df['30_minute_contiguous_session'] = df['time_diff_seconds'] < 30 * 60

    print('Converting back to dask. reindexing...')
    """
    Approx size is 6gb: repartition to 100mb chunks
    """

    df = df.set_index("row_count", npartitions=60, drop=False)

    print('DF in dask format')
    print(df[['user_id', 'date_time', 'previous_date_time', 'time_diff_seconds', '5_minute_contiguous_session', '30_minute_contiguous_session', 'row_count']].head(10))

    print('Calculating 5 minute contiguous sessions')
    inflections_5 = df.groupby('user_id').apply(get_inflection_points_5, meta=(None, pd.Series(dtype='int64'))).compute().to_dict()

    print('Calculating 30 minute contiguous sessions')
    inflections_30 = df.groupby('user_id').apply(get_inflection_points_30, meta=(None, pd.Series(dtype='int64'))).compute().to_dict()

    print(f'Checking sample user: {SAMPLE_USER}')
    print('Inflections 5', inflections_5[SAMPLE_USER][0:10])
    print('Inflections 30', inflections_30[SAMPLE_USER][0:10])

    print('Applying inflection points to 5 minute sessions and 30 minute sessions')
    df['5_minute_session_count'] = df.apply(lambda x: label_session_windows(x['row_count'], inflections_5[x['user_id']]), axis=1, meta=(None, 'int64'))
    df['30_minute_session_count'] = df.apply(lambda x: label_session_windows(x['row_count'], inflections_30[x['user_id']]), axis=1, meta=(None, 'int64'))

    print(f'Writing 5 minute sessions to {write_path}/')

    df.drop(columns=['row_count']).to_csv(f'{write_path}/contiguous-session-*.csv')
    print(f'Application finished')
    end_time = datetime.now()

    total_time = end_time - current_time
    print(f"Total time: {total_time}")

def evaluate_sessions(sample_df, eval_column):
    sessions = sample_df[eval_column].unique()
    print(sessions)
    for session in sessions[0:3]:
        print(f'Session {session}')
        max_index = sample_df[sample_df[eval_column] == session].index.max()
        print(sample_df.iloc[max_index -2:max_index+3].values)

def evaluate_sessionized_data():

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='contiguous_session_data_optimized')

    args = parser.parse_args()

    read_path = args.read_path

    """
    Glob does not maintain ordering; require to order files to ensure timestamp maintained
    """
    files = list(glob.iglob(f"{read_path}/*.csv"))

    files = sorted(files)

    df = dd.read_csv(
        files,
        usecols=['user_id', 'date_time', 'previous_date_time', 'time_diff_seconds', '5_minute_contiguous_session', '5_minute_session_count', '30_minute_contiguous_session', '30_minute_session_count']
    )

    sample_df = df[df['user_id'] == SAMPLE_USER].compute()
    sample_df = sample_df.reset_index()
    sample_df = sample_df[['date_time', 'previous_date_time', 'time_diff_seconds', '5_minute_contiguous_session', '5_minute_session_count', '30_minute_contiguous_session', '30_minute_session_count']]

    sample_df.reset_index(inplace=True)

    evaluate_sessions(sample_df[[col for col in sample_df.columns if '30' not in col]], '5_minute_session_count')
    evaluate_sessions(sample_df[[col for col in sample_df.columns if '5' not in col]], '30_minute_session_count')

if __name__ == '__main__':

    main()
    evaluate_sessionized_data()
