import argparse
import glob
import os
import pdb
import shutil
from random import sample

import numba
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask import delayed
from dask.distributed import Client
from dask.distributed import LocalCluster
from numba import jit
from tqdm import tqdm
SESSION_MINUTES_EXTENSION = 30
SAMPLE_USER_ID = 2371513

REQUIRED_COLUMNS = [
    "user_id",
    "project_id",
    "workflow_id",
    "country",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "day_of_week",
    "date_time",
    "previous_date_time",
    "time_diff_seconds",
    "5_minute_session_count",
    "30_minute_session_count",
]

def evaluate_session(df):
    summary_df = df[df['user_id'] == 2371513.0].compute().reset_index()
    print(summary_df[['user_id', '30_minute_session_count', 'date_time', 'time_until_end_of_session', 'label']].head(50))



def main():

    #argparse arguments read and write path. return args create vars read and write
    client = Client()
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', '-r', type=str, default='contiguous_session_data_optimized/')
    parser.add_argument('--write_path', '-w', type=str, default='contiguous_session_data_labelled/')
    parser.add_argument('--eval',  '-e', action='store_true', default=True)

    args = parser.parse_args()

    read_path = args.read_path
    write_path = args.write_path

    #read in data from read_path
    paths = list(glob.iglob(read_path + "*.csv"))
    print(f'Read path: {read_path}, Write path: {write_path}')
    print(f'Found {len(paths)} files to process')

    #clear write path
    print('Clearing write path')
    shutil.rmtree(write_path, ignore_errors=True)

    #sort paths
    paths = sorted(paths)

    print(f'Reading from: {paths[0]} -> {paths[-1]}')

    #read in data
    df = dd.read_csv(paths, usecols=REQUIRED_COLUMNS)
    df = df.persist()
    df['date_time'] = dd.to_datetime(df['date_time'])
    print(f'Number of rows prior to assignment: {len(df)}')
    # identify max timestamp for each 30 minute session count

    print('Identifying max timestamp for each 30 minute work session')
    max_session_count = df.groupby(by=['user_id', '30_minute_session_count'])['date_time'].max().compute().reset_index().rename(
        columns={
            'date_time': 'max_session_time'
        }
    )

    # merge max timestamp with original data
    print('Merging max timestamp with original data')
    df = df.merge(max_session_count, on=['user_id', '30_minute_session_count'], how='left')

    print('Labeling sessions')
    df['time_until_end_of_session'] = df['max_session_time'] - df['date_time']
    df['label'] = df['time_until_end_of_session'].apply(lambda x: x.total_seconds() > SESSION_MINUTES_EXTENSION * 60 , meta=('label', 'bool'))
    print(df.head(10))

    if args.eval:
        print('Evaluating session')
        evaluate_session(df)

    # # write to disk
    print(f'Writing to disk {write_path}')

    df.to_csv(f'{write_path}/labelled_session_data-*.csv')


if __name__ == "__main__":
    main()
