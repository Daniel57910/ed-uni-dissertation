import argparse
import logging
import os
from datetime import datetime
from functools import reduce
from pprint import pformat
from typing import Callable
import boto3
import random
import numpy as np
import pandas as pd
import torch
from callback import DistributionCallback
from environment import CitizenScienceEnv
from policies.cnn_policy import CustomConv1dFeatures
from rl_util import setup_data_at_window
from rl_constant import LABEL, METADATA, OUT_FEATURE_COLUMNS, PREDICTION_COLS

ALL_COLS = METADATA + OUT_FEATURE_COLUMNS 

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

S3_BASELINE_PATH = 's3://dissertation-data-dmiller'
USER_INDEX = 1
SESSION_INDEX = 2
CUM_SESSION_EVENT_RAW = 3
TIMESTAMP_INDEX = 11
TRAIN_SPLIT = 0.7
EVAL_SPLIT = 0.15

global logger
logger = logging.getLogger('rl_results_eval')
logger.setLevel(logging.INFO)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='datasets/rl_ready_data')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--n_sequences', type=int, default=40)
    parse.add_argument('--lstm', type=str, default='seq_10')
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--feature_extractor', type=str, default='cnn') 
    parse.add_argument('--window', type=int, default=4)
    args = parse.parse_args()
    return args


def main(args):

    logger.info('Starting offlline evaluation of RL model')
    
    read_path, n_files, n_sequences, lstm, device, feature_extractor, window = (
        args.read_path, args.n_files, args.n_sequences, args.lstm, args.device, args.feature_extractor, args.window
    )
    
    logger.info(f'Loading data from {read_path}')
    
    df = pd.read_parquet(f'{read_path}/files_used_{n_files}/predicted_data.parquet')
    
    logger.info(f'Loaded data with shape {df.shape}')
    logger.info(f'Setting up convolution over {window}T window')
    df = setup_data_at_window(df, window)

if __name__ == '__main__':
    args = parse_args()
    main(args)