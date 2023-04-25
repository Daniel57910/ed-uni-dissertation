import argparse
import numpy as np
import torch
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3 import A2C
from callback import DistributionCallback
from stable_baselines3.common.env_checker import check_env
import logging
USER_INDEX = 1
SESSION_INDEX = 2
CUM_SESSION_EVENT_RAW = 3
TIMESTAMP_INDEX = 11
TRAIN_SPLIT = 0.7
EVAL_SPLIT = 0.15
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from pprint import pformat
import os
from environment import CitizenScienceEnv
from callback import DistributionCallback
from rl_constant import TORCH_LOAD_COLS, OUT_FEATURE_COLUMNS, METADATA, LABEL
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)


S3_BASELINE_PATH = 's3://dissertation-data-dmiller'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/rl_ready_data')
    parser.add_argument('--n_files', type=str, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--n_envs', type=int, default=100)
    parser.add_argument('--lstm', type=str, default='ordinal_10')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    return args

import numpy as np
import torch
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3 import A2C
import logging
USER_INDEX = 1
SESSION_INDEX = 2
CUM_SESSION_EVENT_RAW = 3
TIMESTAMP_INDEX = 11
TRAIN_SPLIT = 0.7
EVAL_SPLIT = 0.15
import pandas as pd
# import cudf as gpu_pd
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from pprint import pformat
import os
# from cuml.preprocessing import MinMaxScalerGPU
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)


S3_BASELINE_PATH = 's3://dissertation-data-dmiller'


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
    
    session_size = dataset.groupby(['user_id', 'session_30_raw']).size().reset_index(name='session_size')
    session_minutes = dataset.groupby(['user_id', 'session_30_raw'])['cum_session_time_raw'].max().reset_index(name='session_minutes')
    session_size['sim_size'] = (session_size['session_size'] * .7).astype(int).apply(lambda x: x if x > 1 else 1)
    dataset = dataset.merge(session_size, on=['user_id', 'session_30_raw'])
    dataset = dataset.merge(session_minutes, on=['user_id', 'session_30_raw'])
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
    
    read_path, n_files, n_sequences, n_episodes, device, n_envs = (
        args.read_path, 
        args.n_files, 
        args.n_sequences, 
        args.n_episodes, 
        args.device,
        args.n_envs
    )
    
    file_ext = '.gzip' if not torch.cuda.is_available() else None
    
    read_path = os.path.join(
        read_path,
        f'files_used_{n_files}',
        f'rl_ready_data.parquet{file_ext}'
    )
    
    logger.info(f'Reading data from {read_path}_{n_files}.parquet')
    df = pd.read_parquet(read_path, columns=TORCH_LOAD_COLS)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by=['date_time'])
    logger.info('Data read: generating metadata')
    df['reward'] = df['delta_last_event']
    df = generate_metadata(df)
    df[OUT_FEATURE_COLUMNS] = MinMaxScaler().fit_transform(df[OUT_FEATURE_COLUMNS])
    if torch.cuda.is_available():
        df = df.to_pandas()
    # df = df.to_pandas()
    unique_episodes = df[['user_id', 'session_30_raw']].drop_duplicates()
    unique_sessions = df[['session_30_raw']].drop_duplicates()
    logger.info(f'Parralelizing environment with {n_envs} environments')
    
    citizen_science_vec = DummyVecEnv([lambda: CitizenScienceEnv(df, unique_episodes, unique_sessions, n_sequences) for _ in range(n_envs)])

    logger.info(f'Vectorized environments created, wrapping with monitor')

    base_path = os.path.join(
        S3_BASELINE_PATH,
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
    checkpoint_callback = CheckpointCallback(save_freq=1000 // n_envs, save_path=checkpoint_dir, name_prefix='rl_model')
    dist_callback = DistributionCallback()
    DistributionCallback.tensorboard_dir(tensorboard_dir)
    callback_list = CallbackList([callback_max_episodes, dist_callback, checkpoint_callback])
    monitor_train = VecMonitor(citizen_science_vec)
    
    model = A2C("MlpPolicy", monitor_train, verbose=2, tensorboard_log=tensorboard_dir)
            
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


    model.learn(total_timesteps=100_000, progress_bar=True, log_interval=10, callback=callback_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)