import argparse
import logging
import os
import random
from datetime import datetime
from functools import reduce
from pprint import pformat
from typing import Callable

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from stable_baselines3 import A2C, DQN, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                StopTrainingOnMaxEpisodes)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (DummyVecEnv, VecMonitor,
                                              VecNormalize)
from stable_baselines3.dqn.policies import DQNPolicy

from callback import DistributionCallback
from environment import CitizenScienceEnv
from environment_q2 import CitizenScienceEnvQ2
from policies.cnn_policy import CustomConv1dFeatures
from rl_constant import (FEATURE_COLUMNS, LOAD_COLS, METADATA, PREDICTION_COLS,
                         RL_STAT_COLS)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=1000, suppress=True)
torch.set_printoptions(precision=4, linewidth=500, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import torch.nn as nn

global logger
logger = logging.getLogger('rl_exp_train')
logger.setLevel(logging.INFO)

S3_BASELINE_PATH = 's3://dissertation-data-dmiller/'
N_SEQUENCES = 15
CHECKPOINT_FREQ = 300_000
TB_LOG = 10_000
WINDOW = 1
REWARD_CLIP = 90
MIN_MAX_RANGE = (10, 90)
"""
Reward clip based on achieving maximum reward for 90 minute session at
(s / 45) * (s - 45)
"""

MODEL_PROTOS = {
    'dqn': DQN,
    'ppo': PPO,
    'a2c': A2C
}



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='rl_ready_data_conv')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--n_episodes', type=int, default=10_000)
    parse.add_argument('--lstm', type=str, default='label')
    parse.add_argument('--part', type=str, default='train')
    parse.add_argument('--feature_extractor', type=str, default='cnn') 
    parse.add_argument('--q2', type=bool, default=True)
    parse.add_argument('--model', type=str, default='dqn')
    args = parse.parse_args()
    return args


def simplify_experiment(vectorized_df):
    vectorized_df = [
        df[(df['session_size'] >= MIN_MAX_RANGE[0]) & (df['session_size'] <= MIN_MAX_RANGE[1])] for df in vectorized_df
    ]
    
    return vectorized_df

def rebatch_data(vectorized_df):
    df_sublist = []
    for i in range(0, len(vectorized_df), 5):
        df_sublist.append(pd.concat(vectorized_df[i:i+5], ignore_index=True))
    return df_sublist
        
def main(args):
   
    
    logger.info('Starting Incentive Reinforcement Learning')
    logger.info(pformat(args.__dict__))
    exec_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    read_path, n_files, n_episodes, lstm, part, feature_ext, q2, model = (
        args.read_path, 
        args.n_files, 
        args.n_episodes, 
        args.lstm,
        args.part,
        args.feature_extractor,
        args.q2,
        args.model
    )

    base_read_path = os.path.join(read_path, f'files_used_{n_files}', f'window_{WINDOW}', f'batched_{part}')
    logger.info(f'Reading data from {base_read_path}')
    files= os.listdir(base_read_path)
    n_envs = len(files)
    logger.info(f'Files found: {len(files)} for environment vectorization')


    df_files = [
        pd.read_parquet(os.path.join(base_read_path, file), columns=LOAD_COLS)
        for file in files
    ]
   
    df_files = simplify_experiment(df_files)
    
    if args.q2:
        logger.info(f'Using Q2 environment: rebatching data')
        df_files = rebatch_data(df_files)
        logger.info(f'Rebatched to n envs: {len(df_files)}')
    

    n_envs = len(df_files)
    
    logger.info(f'Files used: {len(df_files)} for environment vectorization')
    
    out_features = FEATURE_COLUMNS + [lstm] if lstm else FEATURE_COLUMNS
    
    logger.info(f'Out features: {out_features}')
    if q2:
        citizen_science_vec = DummyVecEnv([lambda: CitizenScienceEnvQ2(vec_df, out_features, N_SEQUENCES) for vec_df in df_files])
    else:
        citizen_science_vec =DummyVecEnv([lambda: CitizenScienceEnv(vec_df, out_features, N_SEQUENCES) for vec_df in df_files])
    

    monitor_train = VecMonitor(citizen_science_vec)
    
    logger.info(f'Vectorized environments created')

    base_exp_path = os.path.join('experiments', "q2" if q2 else "q1", f'{model}_{lstm}_{feature_ext}/{exec_time}')


    tensorboard_dir, checkpoint_dir = (
        os.path.join(base_exp_path, 'training_metrics'),
        os.path.join(base_exp_path, 'checkpoints')
    )

    if not os.path.exists(tensorboard_dir):
        logger.info(f'Creating directory {tensorboard_dir} for tensorboard logs')
        os.makedirs(tensorboard_dir)
   
    if not os.path.exists(checkpoint_dir):
        logger.info(f'Creating directory {checkpoint_dir} for checkpoints')
        os.makedirs(checkpoint_dir) 

    checkpoint_freq = int(CHECKPOINT_FREQ // (n_envs // 2))
    log_freq = int(TB_LOG // n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir, 
        verbose=2
    )
    
    DistributionCallback.tensorboard_setup(tensorboard_dir, (TB_LOG * 5) // n_envs)
    logger_callback = DistributionCallback()
    
    callback_list = CallbackList([checkpoint_callback, logger_callback])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Setting up model: {model}, device: {device}')
    

    CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1, len(out_features) + 3)
    logger.info('Using custom 1 dimensional CNN feature extractor')
    policy_kwargs = dict(
    features_extractor_class=CustomConv1dFeatures,
        net_arch=[12],
        normalize_images=False,
        activation_fn=nn.ELU,
    )
        
    model_proto = MODEL_PROTOS[model]
        
    model = model_proto(
        policy='CnnPolicy', 
        env=monitor_train, 
        verbose=1, 
        tensorboard_log=tensorboard_dir, 
        policy_kwargs=policy_kwargs, 
        device=device, 
        stats_window_size=1000)

        
    logger.info(f'Model created: policy')
   
    logger.info(pformat(model.policy))
        
    logger.info(f'Beginning training') 
    
            
    logger.info(pformat([
        'n_episodes: {}'.format(n_episodes),
        'read_path: {}'.format(read_path),
        'n_files: {}'.format(n_files),
        'n_sequences: {}'.format(N_SEQUENCES),
        'n_envs: {}'.format(n_envs),
        'device: {}'.format(device),
        'lstm: {}'.format(lstm),
        'part: {}'.format(part),
        'feature_extractor: {}'.format(feature_ext),
        'tensorboard_dir: {}'.format(tensorboard_dir),
        'checkpoint_dir: {}'.format(checkpoint_dir),
        'checkpoint_freq: {}'.format(checkpoint_freq),
        'tb_freq: {}'.format(log_freq),
        'running q2: {}'.format(q2),
    ]))
    

    model.learn(total_timesteps=10_000, log_interval=log_freq, progress_bar=True, callback=callback_list)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    