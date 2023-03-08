import argparse
import glob
import os
import pdb
import shutil
from random import sample

import numba
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
# from dask import dataframe as dd
# from dask.distributed import Client, LocalCluster
SESSION_MINUTES_EXTENSION = 30

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


def outer_session(sessions):

    @numba.jit(nogil= True, forceobj=True)
    def assign_session_label(user_id, session_count, hit_entry_time, sessions):
        session_inflection = sessions[(sessions[:,0] == user_id) & (sessions[:,1] == session_count)][:,-1][0]
        return session_inflection - hit_entry_time

    return assign_session_label

def main():

    #argparse arguments read and write path. return args create vars read and write

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/contiguous_session_sample/')
    parser.add_argument('--write_path', type=str, default='datasets/contiguous_session_labelled/')
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

    paths = paths[0:2]
    print(f'Reading from: {paths[0]} -> {paths[-1]}')

    #read in data
    df_subset = [pd.read_csv(path, usecols=REQUIRED_COLUMNS, parse_dates=True, infer_datetime_format=True) for path in paths]
    df = pd.concat(df_subset, ignore_index=True)
    print(f'Number of rows prior to assignment: {len(df)}')
    df['date_time'] = pd.to_datetime(df['date_time'])

    """
    total seconds from date_time
    """
    # identify max timestamp for each 30 minute session count

    max_session_count = df.groupby(by=['user_id', '30_minute_session_count'])['date_time'].max().reset_index().to_numpy()
    print(f'Total number of sessions: {max_session_count.shape[0]}')

    tqdm.pandas()

    """
    assign session labels
    """

    df = df[df['user_id'] == 2371513.0]

    df = df.reset_index()

    assign_session_label = outer_session(max_session_count)

    df['time_until_end_of_session'] = df.progress_apply(
        lambda x:
        assign_session_label(x['user_id'], x['30_minute_session_count'], x['date_time'], max_session_count), axis=1
    )

    df['label'] = df['time_until_end_of_session'].dt.total_seconds() > SESSION_MINUTES_EXTENSION * 60
    print(df[['user_id', '30_minute_session_count', 'date_time', 'time_until_end_of_session', 'label']].head(50))







if __name__ == "__main__":
    main()
