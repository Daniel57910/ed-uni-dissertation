import argparse
import glob
import os
import pdb
from random import sample

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
from dask.distributed import LocalCluster
from numba import jit

LABEL_COLS = ['5_within_5_mins', 'within_5']
EXAMPLE_COLS = ['user_id', 'date_time', 'project_id']
CALENDAR_COLS = ['year', 'month', 'day', 'hour', 'minute', 'second']
EVAL_COLUMNS_5_MINS = ['user_id', 'date_time', 'previous_date_time', 'time_diff_seconds', '5_minute_contiguous_session', '5_minute_session_count']
EVAL_COLUMNS_30_MINS = ['user_id', 'date_time', 'previous_date_time', 'time_diff_seconds', '30_minute_contiguous_session', '30_minute_session_count']
PATH_CONSTRAINTS = (260, 270)
TOP_10_USERS = [870816.0, 2482328.0, 2609.0, 2324166.0, 695707.0, 2321181.0, 2404151.0, 2316079.0, 2380660.0, 2404871.0]
SAMPLE_USER = 2324166.0

def outer_apply_inflection_points(inflections):
    def inner_find_session(row):
        inflection_for_user = inflections[row['user_id']]
        index_for_row = row['row_count']
        return 1 + np.searchsorted(inflection_for_user, index_for_row, side='right')

    return inner_find_session

def get_inflection_points_5(subset):
    return subset[subset['5_minute_contiguous_session'] == False].index.values

def get_inflection_points_30(subset):
    return subset[subset['30_minute_contiguous_session'] == False].index.values

def sessionize_hit_entries():
    paths = list(glob.iglob("datasets/encoded_time_data/*.csv"))

    sample_paths = paths[PATH_CONSTRAINTS[0]:PATH_CONSTRAINTS[1]]
    df = pd.concat([pd.read_csv(path) for path in sample_paths])

    df['date_time'] = pd.to_datetime(df[CALENDAR_COLS])
    df = df[EXAMPLE_COLS]
    df['previous_date_time'] = df.groupby('user_id')['date_time'].shift(1)
    df['time_diff_seconds'] = (df['date_time'] - df['previous_date_time']).apply(lambda x: x.total_seconds())

    print(f'Number of rows: {len(df)}')
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    print(f'Dropped NA rows. Number of rows: {len(df)}')

    """
    Dropping returns >99% of the data. This is because the first entry for each user is dropped.
    """

    df['5_minute_contiguous_session'] = df['time_diff_seconds'] < 5 * 60
    df['30_minute_contiguous_session'] = df['time_diff_seconds'] < 30 * 60
    df['row_count'] = df.apply(lambda x: x.name, axis=1)

    # get top 10 users by count
    df = df[df['user_id'].isin(TOP_10_USERS)]
    inflections_5 = df.groupby('user_id').apply(get_inflection_points_5)
    inflections_30 = df.groupby('user_id').apply(get_inflection_points_30)

    print(f'Checking sample user: {SAMPLE_USER}')
    print('Inflections 5', inflections_5[SAMPLE_USER])
    print('Inflections 30', inflections_30[SAMPLE_USER])

    inflections_5_function = outer_apply_inflection_points(inflections_5)
    inflections_30_function = outer_apply_inflection_points(inflections_30)
    df['5_minute_session_count'] = df.apply(inflections_5_function, axis=1)
    df['30_minute_session_count'] = df.apply(inflections_30_function, axis=1)
    df.drop(columns=['row_count']).to_csv('datasets/sample_user_sessionized.csv', index=False)

def evaluate_sessions(sample_df, eval_column):
    sessions = sample_df[eval_column].unique()
    print(sessions)
    for session in sessions[0:3]:
        print(f'Session {session}')
        max_index = sample_df[sample_df[eval_column] == session].index.max()
        print(sample_df.iloc[max_index -2:max_index+3].values)

def evaluate_sessionized_data():
    df = pd.read_csv('datasets/sample_user_sessionized.csv')
    sample_df = df[df['user_id'] == SAMPLE_USER]
    sample_df = sample_df.reset_index()
    sample_df = sample_df[['date_time', 'previous_date_time', 'time_diff_seconds', '5_minute_contiguous_session', '5_minute_session_count', '30_minute_contiguous_session', '30_minute_session_count']]
    # evaluate_sessions(sample_df[[col for col in sample_df.columns if '30' not in col]], '5_minute_session_count')
    evaluate_sessions(sample_df[[col for col in sample_df.columns if '5' not in col]], '30_minute_session_count')

def main():

    # sessionize_hit_entries()
    evaluate_sessionized_data()



main()
