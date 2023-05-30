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
from rl_constant import LABEL, METADATA, OUT_FEATURE_COLUMNS, PREDICTION_COLS
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                StopTrainingOnMaxEpisodes)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.dqn.policies import DQNPolicy

from rl_util import setup_data_at_window

ALL_COLS = METADATA + OUT_FEATURE_COLUMNS 

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

S3_BASELINE_PATH = 's3://dissertation-data-dmiller'


N_SEQUENCES = 40
CHECKPOINT_FREQ = 100_000
TB_LOG = 10_000
WINDOW = 2

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='rl_ready_data_conv')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--n_episodes', type=int, default=50)
    parse.add_argument('--n_envs', type=int, default=100)
    parse.add_argument('--lstm', type=str, default='seq_10')
    parse.add_argument('--part', type=str, default='train')
    parse.add_argument('--feature_extractor', type=str, default='cnn') 
    args = parse.parse_args()
    return args

def _lstm_loader(lstm):
    
    return LABEL[0] if lstm == 'label' else lstm

def load_and_dedupe(read_path):
    df = pd.read_parquet(read_path)
    df = df.reset_index(drop=True)
    df = df.loc[:,~df.columns.duplicated()]
    return df
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def main(args):
    
    exec_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logger.info('Starting Incentive Reinforcement Learning')
    logger.info(pformat(args.__dict__))
    
    read_path, n_files, n_episodes, n_envs, lstm, part, feature_ext = (
        args.read_path, 
        args.n_files, 
        args.n_episodes, 
        args.n_envs,
        args.lstm,
        args.part,
        args.feature_extractor,
    )
    
    read_path = os.path.join(
        'rl_ready_data_conv',
        f'files_used_{n_files}',
        f'window_{WINDOW}_{part}.parquet'
    )
    
    logger.info(f'Reading data from {read_path}')

    df = load_and_dedupe(read_path)
    
    if not os.path.exists(read_path):
        raise ValueError(f'No data found at {read_path}')
 
    lstm = _lstm_loader(lstm)
    
    
    read_cols, out_features = (
        ALL_COLS + [lstm] if lstm else ALL_COLS,
        OUT_FEATURE_COLUMNS + [lstm] if lstm else OUT_FEATURE_COLUMNS
    )
     
    logger.info(f'Loaded data with shape {df.shape}')
    logger.info(f'Setting up convolution over {WINDOW}T minutes')
 
    citizen_science_vec =DummyVecEnv([lambda: CitizenScienceEnv(df, out_features, N_SEQUENCES) for i in range(n_envs)])
    logger.info(f'Vectorized environments created')
    
    base_path = os.path.join(
        S3_BASELINE_PATH,
        'reinforcement_learning_incentives_3',
        f'n_files_{n_files}',
        feature_ext + '_' + 'label' if lstm.startswith('continue') else feature_ext + f'_{lstm}',
        'results',
        exec_time,
    ) 
    
    
    tensorboard_dir, checkpoint_dir = (
        os.path.join(base_path, 'training_metrics'),
        os.path.join(base_path, 'checkpoints')
    )

    logger.info(f'Creating callbacks, monitors and loggerss')
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=n_episodes, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ// (n_envs // 2), save_path=checkpoint_dir, name_prefix='rl_model')
    callback_list = CallbackList([callback_max_episodes, checkpoint_callback])
    monitor_train = VecMonitor(citizen_science_vec)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if feature_ext == 'cnn':
        CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1,len(out_features))
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
    
    # retutrn
            
    logger.info(pformat([
        'n_episodes: {}'.format(n_episodes),
        'read_path: {}'.format(read_path),
        'n_files: {}'.format(n_files),
        'n_sequences: {}'.format(N_SEQUENCES),
        'n_envs: {}'.format(n_envs),
        'total_timesteps: {}'.format(df.shape),
        'device: {}'.format(device),
        'tensorboard_dir: {}'.format(tensorboard_dir),
        'checkpoint_dir: {}'.format(checkpoint_dir)
    ]))
    
    model.learn(total_timesteps=10000, progress_bar=True, log_interval=TB_LOG, callback=callback_list)
    

if __name__ == '__main__':
    args = parse_args()
    
    main(args) 