import argparse
import glob
import os
import pdb

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
from dask.distributed import LocalCluster

SAMPLE_USER =  932809.0
REQUIRED_COLS = ['user_id', '30_minute_session_count', 'task_within_session_count', 'date_time']

def pandas():

    files = glob.glob('contiguous_session_data_count/*.csv')

    files = list(sorted(files))

    df = pd.read_csv(files[0], usecols=REQUIRED_COLS)

    df['date_time'] = pd.to_datetime(df['date_time'])

    inflections_min = df.groupby(['user_id', '30_minute_session_count'])[['date_time']].min().reset_index()
    inflections_max = df.groupby(['user_id', '30_minute_session_count'])[['date_time']].max().reset_index()

    inflections_min.columns = ['user_id', '30_minute_session_count', 'date_time_min']
    inflections_max.columns = ['user_id', '30_minute_session_count', 'date_time_max']

    inflections = pd.merge(inflections_max, inflections_min, on=['user_id', '30_minute_session_count'])

    inflections['last_session'] = inflections.groupby('user_id')['date_time_max'].shift(1)
    inflections['elpased_time'] = inflections['date_time_min'] - inflections.groupby(['user_id'])['date_time_max'].shift(1)
    inflections['elpased_time'] = inflections['elpased_time'].dt.total_seconds() / 60


def main():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print('Starting client')
    client = Client()
    files = glob.glob('contiguous_session_data_count/*.csv')
    files = list(sorted(files))

    df = dd.read_csv(files, usecols=REQUIRED_COLS)

    df['date_time'] = dd.to_datetime(df['date_time'])

    print('Files in memory: calculating min_max')
    inflections_min = df.groupby(['user_id', '30_minute_session_count'])[['date_time']].min().reset_index().compute()
    inflections_max = df.groupby(['user_id', '30_minute_session_count'])[['date_time']].max().reset_index().compute()

    inflections_min.columns = ['user_id', '30_minute_session_count', 'date_time_min']
    inflections_max.columns = ['user_id', '30_minute_session_count', 'date_time_max']

    print('Joining min_max on user_id and 30_minute_session_count')
    inflections = pd.merge(inflections_max, inflections_min, on=['user_id', '30_minute_session_count'])

    print('Calculating last_session and elapsed_time')
    inflections['elpased_time'] = (inflections['date_time_min'] - inflections.groupby(['user_id'])['date_time_max'].shift(1)).dt.total_seconds() / 60

    print('Saving inflections')
    inflections.to_csv('summary_stats_session/csv/elapsed_time_between_sessions.csv', index=False)


if __name__ == "__main__":
    main()
