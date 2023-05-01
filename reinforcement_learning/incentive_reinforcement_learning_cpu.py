import argparse
import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3 import A2C, DQN, PPO
from policies.cnn_policy import CustomConv1dPolicy, TestConv1
from stable_baselines3.common.env_checker import check_env
import logging
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from datetime import datetime
from pprint import pformat
import os
from environment import CitizenScienceEnv
from callback import DistributionCallback
from rl_constant import (
    LABEL,
    METADATA,
    OUT_FEATURE_COLUMNS,
    PREDICTION_COLS
)

# %load incentive_reinforcement_learning_cpu.py
import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3 import A2C, DQN
import logging
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from datetime import datetime
from pprint import pformat
import os


ALL_COLS = LABEL + METADATA + OUT_FEATURE_COLUMNS 

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

def generate_metadata(dataset):
    
    session_size = dataset.groupby(['user_id', 'session_30_raw'])['size_of_session'].max().reset_index(name='session_size')
    session_minutes = dataset.groupby(['user_id', 'session_30_raw'])['cum_session_time_raw'].max().reset_index(name='session_minutes')
    session_minutes['sim_minutes'] = session_minutes['session_minutes'] * .7
    session_size['sim_size'] = (session_size['session_size'] * .7).astype(int).apply(lambda x: x if x > 1 else 1)
    dataset = dataset.merge(session_size, on=['user_id', 'session_30_raw'])
    dataset = dataset.merge(session_minutes, on=['user_id', 'session_30_raw'])
    dataset['reward'] = dataset['cum_session_time_raw']
    return dataset
   
   

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='datasets/rl_ready_data')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--n_episodes', type=int, default=50)
    parse.add_argument('--n_sequences', type=int, default=10)
    parse.add_argument('--n_envs', type=int, default=100)
    parse.add_argument('--lstm', type=str, default='seq_10')
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--checkpoint_freq', type=int, default=1000)
    parse.add_argument('--tb_log', type=int, default=100)
    
    args = parse.parse_args()
    return args



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


def remove_events_in_2_minute_window(df):
    df['second_window'] = df['second'] // 10
    df = df.drop_duplicates(
        subset=['user_id', 'session_30_raw', 'year', 'month', 'day', 'hour', 'minute'],
        keep='last'
    ).reset_index(drop=True)

    return df


def convolve_delta_events(df):
    df['convolved_delta_event'] = (
        df.set_index('date_time').groupby(by=['user_id', 'session_30_raw'], group_keys=False) \
            .rolling('2T', min_periods=1)['delta_last_event'] \
            .mean()
            .reset_index(name='convolved_event_delta')['convolved_event_delta']
    )

    df['delta_last_event'] = df['convolved_delta_event']

    return df

def main(args):
    
    exec_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logger.info('Starting Incentive Reinforcement Learning')
    
    read_path, n_files, n_sequences, n_episodes, device, n_envs, lstm, tb_log, check_freq = (
        args.read_path, 
        args.n_files, 
        args.n_sequences, 
        args.n_episodes, 
        args.device,
        args.n_envs,
        args.lstm,
        args.tb_log,
        args.checkpoint_freq
    )
    
    read_path = os.path.join(
        read_path,
        f'files_used_{n_files}',
        f'predicted_data.parquet'
    )
    
    logger.info(f'Reading data from {read_path}_{n_files}.parquet')
    cols = ALL_COLS + [lstm] if lstm else ALL_COLS
    out_features = OUT_FEATURE_COLUMNS + [lstm] if lstm else OUT_FEATURE_COLUMNS
    if args.lstm:
        logger.info(f'Including LSTM prediction/label: {lstm}')
        df = pd.read_parquet(read_path, columns=cols)
    else:
        logger.info(f'Setting up baseline without prediction/label')
        df = pd.read_parquet(read_path, columns=ALL_COLS)
    df['date_time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']], errors='coerce')


    df = df.sort_values(by=['date_time'])    
    
    logger.info(f'N events:: {df.shape[0]} creating training partitions')
    df = df.head(int(df.shape[0] * .7))
    logger.info(f'N events after 70% split: {df.shape[0]}')
    size_of_session = df.groupby(['user_id', 'session_30_raw']).size().reset_index(name='size_of_session')
    df = pd.merge(df, size_of_session, on=['user_id', 'session_30_raw'])
    df['cum_session_event_raw'] = df.groupby(['user_id', 'session_30_raw'])['date_time'].cumcount() + 1
    df = convolve_delta_events(df)
   
    df = df[df['minute'] % 2 == 0]
    logger.info(f'Data read: {df.shape[0]} rows, {df.shape[1]} columns, dropping events within 2 minute window')
    df = remove_events_in_2_minute_window(df)
    df = df.reset_index(drop=True)
    
    logger.info(f'Number of events after dropping events within 2 minute window: {df.shape[0]}')
    df = generate_metadata(df)

    unique_episodes = df[['user_id', 'session_30_raw']].drop_duplicates()
    unique_sessions = df[['session_30_raw']].drop_duplicates()
    df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'second_window'])
    
    env = CitizenScienceEnv(df, unique_episodes, unique_sessions, out_features, n_sequences)
    
    policy_kwargs = dict(
        features_extractor_class=CustomConv1dPolicy,
        features_extractor_kwargs=dict(features_dim=64)
    )
    
    policy = CustomConv1dPolicy(env.observation_space, 11, 21, 64)
    
    step = env.reset()
    step = torch.from_numpy(step).float()
    step = policy(step)
    print(step.shape)
    return
    
    custom_dqn = PPO(policy="CnnPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)
    print(custom_dqn.policy)
    
    # custom_dqn.learn(total_timesteps=100_000, progress_bar=True) 
    return
    citizen_science_vec = DummyVecEnv([lambda: CitizenScienceEnv(df, unique_episodes, unique_sessions, out_features, n_sequences) for _ in range(n_envs)])
   
    logger.info(f'Vectorized environments created, wrapping with monitor')

    base_path = os.path.join(
        # S3_BASELINE_PATH,
        'reinforcement_learning_incentives',
        f'n_files_{n_files}',
        'results',
        exec_time,
    ) 
    
    tensorboard_dir, checkpoint_dir = (
        os.path.join(base_path, 'training_metrics'),
        os.path.join(base_path, 'checkpoints')
    )

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=n_episodes, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=check_freq // n_envs, save_path=checkpoint_dir, name_prefix='rl_model')
    dist_callback = DistributionCallback()
    DistributionCallback.tensorboard_setup(tensorboard_dir, tb_log)
    callback_list = CallbackList([callback_max_episodes, dist_callback, checkpoint_callback])
    monitor_train = VecMonitor(citizen_science_vec)
    
    model = DQN("MlpPolicy", monitor_train, verbose=1, tensorboard_log=tensorboard_dir, stats_window_size=1000)
            
    logger.info(pformat([
        'n_epochs: {}'.format(n_episodes),
        'read_path: {}'.format(read_path),
        'n_files: {}'.format(n_files),
        'n_sequences: {}'.format(n_sequences),
        'n_envs: {}'.format(n_envs),
        'total_timesteps: {}'.format(df.shape),
        f'unique_episodes: {unique_episodes.shape[0]}',
        'device: {}'.format(device),
        'tensorboard_dir: {}'.format(tensorboard_dir),
        'checkpoint_dir: {}'.format(checkpoint_dir)
    ]))


    model.learn(total_timesteps=100_000_000, progress_bar=True, log_interval=1000, callback=callback_list)
    

    

if __name__ == "__main__":
    args = parse_args()
main(args)