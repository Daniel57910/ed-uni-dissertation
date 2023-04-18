import argparse
import numpy as np
import torch
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3 import PPO, A2C
import logging
USER_INDEX = 1
SESSION_INDEX = 2
CUM_SESSION_EVENT_RAW = 3
TIMESTAMP_INDEX = 11
TRAIN_SPLIT = 0.7
EVAL_SPLIT = 0.15
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import check_env
from npz_extractor import NPZExtractor
from pprint import pformat
import os
from environment import CitizenScienceEnv
from callback import DistributionCallback
from constant import TORCH_LOAD_COLS, OUT_FEATURE_COLUMNS, METADATA, LABEL

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)


S3_BASELINE_PATH = 's3://dissertation-data-dmiller'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/calculated_features/files_used')
    parser.add_argument('--n_files', type=str, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=18)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--return_distribution', type=str, default='stack_overflow_v1')
    parser.add_argument('--agent', type=str, default='constant_20')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--event_sample', type=float, default=0.25)
    
    args = parser.parse_args()
    return args

def train_eval_split(dataset, logger):
    train_split = int(dataset.shape[0] * TRAIN_SPLIT)
    eval_split = int(dataset.shape[0] * EVAL_SPLIT)
    test_split = dataset.shape[0] - train_split - eval_split
    logger.info(f'Train size: 0:{train_split}, eval size: {train_split}:{train_split+eval_split}: test size: {train_split + eval_split}:{dataset.shape[0]}')
    train_dataset, eval_dataset, test_split = dataset[:train_split], dataset[train_split:train_split+eval_split], dataset[train_split+eval_split:]
    
    return {
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_split
    }

def generate_metadata(dataset, logger):
    
    session_size = dataset.groupby(['user_id', 'session_30_raw']).size().reset_index(name='session_size')
    session_size['sim_size'] = (session_size['session_size'] * .75).astype(int).apply(lambda x: x if x > 1 else 1)
    dataset = dataset.merge(session_size, on=['user_id', 'session_30_raw'])
    return dataset
    



def run_reinforcement_learning_incentives(environment, logger, n_episodes=1):
    for epoch in range(n_episodes):
        environment_comp = False
        state = environment.reset()
        i = 0
        while not environment_comp:
            next_action = (
                1 if np.random.uniform(low=0, high=1) > 0.8 else 0
            )
            state, rewards, environment_comp, meta = environment.step(next_action)
            i +=1
            if i % 100 == 0:
                logger.info(f'Step: {i} - Reward: {rewards}')
                
        logger.info(f'Epoch: {epoch} - Reward: {rewards}')
        print(environment.user_sessions.head(10))

    

def main(args):
    
    exec_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logger.info('Starting Incentive Reinforcement Learning')
    
    read_path, n_files, n_sequences, n_features, n_episodes, device, event_sample = (
        args.read_path, 
        args.n_files, 
        args.n_sequences, 
        args.n_features, 
        args.n_episodes, 
        args.device,
        args.event_sample
    )
    
    logger.info(f'Reading data from {read_path}_{n_files}.parquet')
    df = pd.read_parquet(f'{read_path}_{n_files}.parquet', columns=TORCH_LOAD_COLS)
    logger.info('Data read: generating metadata')
    df['reward'] = df['delta_last_event']
    df = generate_metadata(df, logger)
    max_session = df['session_30_raw'].max()
    session_ranges = np.arange(1, max_session + 1)
    logger.info(f'Metadata generated: instantiating environment: session_ranges: 1 => {max_session}')
    environment = CitizenScienceEnv(df, session_ranges, n_sequences)

    state = environment.reset()
    done = False
    while not done:
        action = 0
        state, reward, done, meta = environment.step(action)
        print(reward, state.shape)



if __name__ == "__main__":
    args = parse_args()
    main(args)