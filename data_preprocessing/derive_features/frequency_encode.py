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

DROP_COLS = [
    'project_id',
    'workflow_id',
    'user_id',
    'country',
    'date_time',
    'previous_date_time',
    'max_session_time',
    'time_until_end_of_session',
    '5_minute_session_count',
]

def main():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    client = Client()
    files = glob.glob('frequency_encoded_data/*.csv')

    files = list(sorted(files))

    df = dd.read_csv(files, assume_missing=True)
    print(df.count().compute())

    for col in df.columns:
        if col.startswith('Unnamed'):
            df = df.drop(col, axis=1)

    print('Df loaded: calculating project workflow count')
    project_workflow_count = df[['project_id', 'workflow_id']]. \
        groupby('project_id')['workflow_id']. \
        value_counts(). \
        compute(). \
        reset_index(name='project_workflow_count')

    print('Calculating user count')
    user_count = df['user_id'].value_counts().compute().reset_index(name='user_count').rename(columns={'index': 'user_id'})

    print('Calculating country count')
    country_count = df['country'].value_counts().compute().reset_index(name='country_count').rename(columns={'index': 'country'})

    print('Running merge operations')

    df = df.merge(project_workflow_count, on=['project_id', 'workflow_id'], how='left')
    df = df.merge(user_count, on='user_id', how='left')
    df = df.merge(country_count, on='country', how='left')

    # df = df.drop(columns=DROP_COLS)

    print('Merge operations complete, writing to disk')
    print(df.head(10))

    for col in df.columns:
        print(col)
    print('Writing to disk: frequency_encoded_data/')

    df.to_csv('frequency_encoded_data/encoding-*.csv', index=False)

if __name__ == "__main__":
    main()
