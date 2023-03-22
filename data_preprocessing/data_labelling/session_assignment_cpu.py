import argparse
import glob

import logging
from session_calculate import SessionCalculate
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import pprint
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
from dask.distributed import LocalCluster
import pprint
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/encoded_time_data')
    parser.add_argument('--write_path', type=str, default='datasets/labelled_session_count_data')
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--test_env', type=bool, default=True)
    args = parser.parse_args()
    return args

ALL_COLUMNS = [
    "project_id",
    "workflow_id",
    "user_id",
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
    "max_session_time",
    "time_until_end_of_session",
    "label",
    "task_within_session_count",
    "project_workflow_count",
    "user_count",
    "country_count"
]

LOAD_COLUMNS = [
    "project_id",
    "workflow_id",
    "user_id",
    "country",  
    "date_time",
]
def get_logger():
    logger = logging.getLogger(__name__)
    return logger


def main(args):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.mode.chained_assignment = None  # default='warn'
    pd.set_option('display.width', 500)

    logger = get_logger()
    read_path, write_path, n_files = args.read_path, args.write_path, args.n_files
    logger.info(f'Read: {read_path}, Write: {write_path}, N Files: {n_files}')
    files_to_read = sorted(list(glob.iglob(f'{read_path}/*.csv')))
    logger.info(f'Found {len(files_to_read)} files to read')
    df = dd.read_csv(files_to_read[:n_files], usecols=LOAD_COLUMNS).compute()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by=['date_time'])
    session_calculator = SessionCalculate(df, args.write_path, args.use_gpu, args.test_env)
    session_calculator.calculate_inflections()
    
    session_calculator.write_inflections_parquet()

  
if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.write_path):
        os.system(f'rm -r {args.write_path}')
    main(args)
    
