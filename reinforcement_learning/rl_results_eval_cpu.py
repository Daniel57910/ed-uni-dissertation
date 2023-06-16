import argparse
import logging
import os
from datetime import datetime
from functools import reduce
from pprint import pformat
from typing import Callable
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
import boto3
import random
import numpy as np
import pandas as pd
import torch
from callback import DistributionCallback
from environment import CitizenScienceEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from policies.cnn_policy import CustomConv1dFeatures
from rl_constant import (
    FEATURE_COLUMNS
)
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
import zipfile
import torch.nn as nn

S3_BASELINE_PATH = 'dissertation-data-dmiller'
N_SEQUENCES = 15
CHECKPOINT_FREQ = 100_000
TB_LOG = 10_000
WINDOW = 2
import glob
TB_LOG = 10_000
WINDOW = 1
REWARD_CLIP = 90
MIN_MAX_RANGE = (10, 90)

global logger

logger = logging.getLogger('rl_results_eval')
logger.setLevel(logging.INFO)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--write_path', type=str, default='rl_evaluation')
    parse.add_argument('--part', type=str, default='eval')
    parse.add_argument('--algo', type=str, default='dqn_pred_cnn'),

    parse.add_argument('--run_date', type=str, default='2023-06-13_16-11-42'),
    parse.add_argument('--n_files', type=int, default=30)
                       
    return parse.parse_args()


def find_s3_candidate(algo, run_date):
    
    folder_prefix = os.path.join(
        'experiments',
        algo,
        run_date,
        'checkpoints'
    )

    
    logger.info(f'Looking for files in {folder_prefix}')
    
    files = [
        {
            'key': file['Key'],
            'last_modified': file['LastModified'],
        }
        for file in client.list_objects_v2(Bucket=S3_BASELINE_PATH, Prefix=folder_prefix)['Contents']
    ]
    
    s3_candidate = max(files, key=lambda x: x['last_modified'])['key']
    
    logger.info(f'Found candiate: {s3_candidate}')
    
    return s3_candidate

def get_policy(algo, run_date):
    
    
    s3_candidate = find_s3_candidate(algo, run_date)
    
    client.download_file(S3_BASELINE_PATH, s3_candidate, s3_candidate)
    return s3_candidate
        

def simplify_experiment(vectorized_df):
    vectorized_df = [
        df[(df['session_size'] >= MIN_MAX_RANGE[0]) & (df['session_size'] <= MIN_MAX_RANGE[1])] for df in vectorized_df
    ]

    return vectorized_df

      
def download_dataset_from_s3(client, base_read_path, full_read_path):
    logger.info(f'Downloading data from {base_read_path}')
    os.makedirs(base_read_path, exist_ok=True)
    
    logger.info(f'Downloading data from dissertation-data-dmiller/{full_read_path}')
    client.download_file(
        'dissertation-data-dmiller',
        full_read_path,
        full_read_path
    )
    logger.info(f'Downloaded data from dissertation-data-dmiller/{full_read_path}')
    

def simplify_experiment(vectorized_df):
    vectorized_df = [
        df[(df['session_size'] >= MIN_MAX_RANGE[0]) & (df['session_size'] <= MIN_MAX_RANGE[1])] for df in vectorized_df
    ]

    return vectorized_df

def _label_or_pred(algo):
    if 'label' in algo:
        return 'label'
    elif 'pred' in algo:
        return 'pred'
    else:
        return None
    
def main(args):
    
    global client
    client = boto3.client('s3')


    logger.info('Starting offlline evaluation of RL model')
    
    write_path, part, algo, run_date, n_files = (
        args.write_path,
        args.part,
        args.algo,
        args.run_date,
        args.n_files
    )
    
    
    read_path = os.path.join(
        'rl_ready_data_conv',
        f'files_used_{n_files}',
        'window_1',
        f'batched_{part}'
    )
   
    logger.info(f'Reading from {read_path}, writing to {write_path}') 
    files_to_read = glob.glob(os.path.join(read_path, '*.parquet'))
    logger.info(f'Found {len(files_to_read)} files to read')
    
    
    feature_cols = FEATURE_COLUMNS + [_label_or_pred(algo)] if _label_or_pred(algo) else FEATURE_COLUMNS
    for col in sorted(feature_cols):
        logger.info(f'Using column {col}')
    
    logger.info(f'n features: {len(feature_cols)}')

    env_files = [
        pd.read_parquet(file) for file in files_to_read[:20]
    ]


    logger.info(f'Loaded env files: clipping to {MIN_MAX_RANGE}')
    
    env_files = simplify_experiment(env_files)

    n_envs = len(env_files) 
    vec_env = DummyVecEnv(
        [lambda: CitizenScienceEnv(df, feature_cols, N_SEQUENCES) for df in env_files]
    )
   

    
    tensorboard_dir = os.path.join(
        args.write_path,
        f'{args.part}/{algo}_{run_date}'
    )
       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    logger.info(f'Logging to {tensorboard_dir}, n envs: {n_envs}: device: {device}')
    
    monitor_env = VecMonitor(vec_env)
    policy_path = get_policy(algo, run_date)
    
    
    DistributionCallback.tensorboard_setup(tensorboard_dir, (TB_LOG) // n_envs)
    logger_callback = DistributionCallback()
    callack_list = CallbackList([logger_callback])
    logger.info(f'Setting up model')
    if 'cnn' in algo:
        CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1, len(feature_cols) + 3)
        logger.info(f'Using custom CNN feature extractor')
        policy_kwargs = dict(
            features_extractor_class=CustomConv1dFeatures,
            net_arch=[12],
            normalize_images=False,
            activation_fn=nn.ELU
        )
        model = DQN(
            policy='CnnPolicy', 
            env=monitor_env,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            policy_kwargs=policy_kwargs,
            device=device,
            stats_window_size=1000
        )
                    
    else:
        model = DQN(
            policy='MlpPolicy', 
            env=monitor_env,
            verbose=1, 
            tensorboard_log=tensorboard_dir, 
            policy_kwargs=dict(
                activation_fn=nn.ELU,
                normalize_images=False,
            ),
            device=device, 
            stats_window_size=1000
        )
    

    logger.info(f'Loading model from {policy_path}')
    model = model.load(policy_path)
    
    
    
    logger.info(f'Running evaluation')
    evaluate_policy(
        model,
        monitor_env,
        n_eval_episodes=100,
        deterministic=True,

    )
    
    
    comp_sessions = monitor_env.get_attr('episode_bins')
    values_to_log = [sess_list for sess_list in comp_sessions if sess_list is not None]
    values_to_log = [item for sublist in values_to_log for item in sublist]
    df = pd.DataFrame(values_to_log)
    
    logger.info(f'Evaluation completed: {df.shape}')
    write_path = os.path.join(
        write_path,
        part,
        f'{algo}_{run_date}',
        'finished_sessions.parquet'
    )
    
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
    
    logger.info(f'Writing to {write_path}')
    df.to_parquet(write_path)
if __name__ == '__main__':
    args = parse_args()
    main(args)