# %load incentive_reinforcement_learning_cpu.py
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


from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                StopTrainingOnMaxEpisodes)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.dqn.policies import DQNPolicy

from rl_constant import (
    FEATURE_COLUMNS,
    METADATA,
    RL_STAT_COLS,
    PREDICTION_COLS,
    LOAD_COLS
)

from environment import CitizenScienceEnv
from callback import DistributionCallback
from policies.cnn_policy import CustomConv1dFeatures


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
global logger
logger = logging.getLogger('rl_exp_train')
logger.setLevel(logging.INFO)

S3_BASELINE_PATH = 's3://dissertation-data-dmiller/'
N_SEQUENCES = 40
CHECKPOINT_FREQ = 250_000
TB_LOG = 10_000
WINDOW = 1

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='rl_ready_data_conv')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--n_episodes', type=int, default=50)
    parse.add_argument('--n_envs', type=int, default=100)
    parse.add_argument('--lstm', type=str, default='label')
    parse.add_argument('--part', type=str, default='train')
    parse.add_argument('--feature_extractor', type=str, default='cnn') 
    args = parse.parse_args()
    return args



def main(args):
    
    df = pd.read_csv('experiments/dqn_label_cnn/2023-06-05_10-39-12/training_metrics/1.csv')
    print(df.head())
    return
    
    
    logger.info('Starting Incentive Reinforcement Learning')
    logger.info(pformat(args.__dict__))
    exec_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    read_path, n_files, n_episodes, n_envs, lstm, part, feature_ext = (
        args.read_path, 
        args.n_files, 
        args.n_episodes, 
        args.n_envs,
        args.lstm,
        args.part,
        args.feature_extractor,
    )

    base_read_path = os.path.join(read_path, f'files_used_{n_files}', f'window_{WINDOW}', f'batched_{part}')
    logger.info(f'Reading data from {base_read_path}')
    files= os.listdir(base_read_path)
    logger.info(f'Files found: {len(files)} for environment vectorization')


    vectorized_df = [
        pd.read_parquet(read_path, columns=LOAD_COLS)
        for file in files
    ]

    out_features = FEATURE_COLUMNS + [lstm] if lstm else FEATURE_COLUMNS
    logger.info(f'Out features: {out_features}')
    environment = CitizenScienceEnv(vectorized_df[0], out_features, N_SEQUENCES)

    citizen_science_vec =DummyVecEnv([lambda: CitizenScienceEnv(vec_df, out_features, N_SEQUENCES) for vec_df in vectorized_df])
    monitor_train = VecMonitor(citizen_science_vec)
    
    logger.info(f'Vectorized environments created')

    base_exp_path = os.path.join('experiments', f'dqn_{lstm}_{feature_ext}/{exec_time}')


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

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=n_episodes, verbose=1)
    checkpoint_freq = int(CHECKPOINT_FREQ // n_envs)
    log_freq = int(TB_LOG // n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir, 
        verbose=2
    )
    
    DistributionCallback.tensorboard_setup(tensorboard_dir, TB_LOG // n_envs * 2) 
    logger_callback = DistributionCallback()
    
    callback_list = CallbackList([checkpoint_callback, logger_callback, callback_max_episodes])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if feature_ext == 'cnn':
        CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1, len(out_features))
        logger.info('Using custom 1 dimensional CNN feature extractor')
        policy_kwargs = dict(
            features_extractor_class=CustomConv1dFeatures,
            net_arch=[10]
        )
        model = DQN(policy='CnnPolicy', env=monitor_train, verbose=1, tensorboard_log=tensorboard_dir, policy_kwargs=policy_kwargs, device=device, stats_window_size=1000)
    else:
        logger.info('Using default MLP feature extractor')
        model = DQN(policy='MlpPolicy', env=monitor_train, verbose=1, tensorboard_log=tensorboard_dir, device=device, stats_window_size=1000)
        
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
    ]))
    
    model.learn(total_timesteps=25_00, progress_bar=True, log_interval=log_freq, callback=callback_list)
    
if __name__ == '__main__':
    args = parse_args()
    
    main(args)