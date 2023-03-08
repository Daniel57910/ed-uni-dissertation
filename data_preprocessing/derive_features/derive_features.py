import argparse
import os
import pdb
from random import sample

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
from dask.distributed import LocalCluster

PATH = "data_preprocessing/time_split/*.csv"

def parse_dates(df):
  return pd.to_datetime(df['date_time'], format = '%Y-%m-%d %H:%M:%S')

def main():


    print('Reading dataframes')
    cluster = LocalCluster()
    cluster.scale(10)
    client = Client(cluster)
    df = dd.read_csv(PATH, assume_missing=True)

    print('Applying time information')
    df['date_time'] = df['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s'), meta=('timestamp', 'datetime64[s]'))
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    df['second'] = df['date_time'].dt.second
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['is_weekend'] = df['date_time'].dt.dayofweek > 4

    df.to_csv('time_split/preprocessed_by_time-*.csv', index=False)

if __name__ == '__main__':
    main()
