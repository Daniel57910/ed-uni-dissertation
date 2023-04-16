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
TIMESTAMP_INDEX = 11
TRAIN_SPLIT = 0.7
EVAL_SPLIT = 0.15
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from npz_extractor import NPZExtractor
from pprint import pformat
import os
from environment import CitizenScienceEnv
from callback import DistributionCallback

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)


S3_BASELINE_PATH = 's3://dissertation-data-dmiller'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--n_files', type=str, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=18)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--return_distribution', type=str, default='stack_overflow_v1')
    parser.add_argument('--agent', type=str, default='constant_20')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--session_sample', type=float, default=1.0)
    
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
     
    logger.info('Generating metadata tasks per session')
    sessions = pd.DataFrame(
        dataset[:, [USER_INDEX, SESSION_INDEX]],
        columns=['user_id', 'session_id']
    )
    
    sessions = sessions.groupby(['user_id', 'session_id']).size().reset_index(name='counts')
    sessions['sim_counts'] = (sessions['counts'] * 0.8).astype(int)
    sessions['sim_counts'] = sessions['sim_counts'].apply(lambda x: 1 if x == 0 else x)
    sessions['incentive_index'] = 0
    
    sessions['task_index'] = 0
    sessions['total_reward'] = 0
    sessions['total_reward'] = sessions['total_reward'].astype(float)
    sessions['ended'] = 0
    return sessions


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
    
    
    read_path, n_files, n_sequences, n_features, n_episodes, device, session_sample = (
        args.read_path, 
        args.n_files, 
        args.n_sequences, 
        args.n_features, 
        args.n_episodes, 
        args.device,
        args.session_sample
    )
    
    npz_extractor = NPZExtractor(
        read_path,
        n_files,
        n_sequences,
        None,
        10000
    )
    
    cpu_count = os.cpu_count()
   
    logger.info(f'Starting experiment at {exec_time}') 
    logger.info(f'Extracting dataset from npz files to tensor' )
    dataset = np.concatenate(npz_extractor.get_dataset_pointer(), axis=1)
    datasets = train_eval_split(dataset, logger)
    train_data = datasets['train']
 
    logger.info(f'Dataset shape: {dataset.shape}: generating metadata tensor')
    sessions_train = generate_metadata(train_data, logger)
    logger.info(f'Metadata train: {sessions_train.shape}')
    sessions_train = sessions_train.sample(frac=session_sample)
    logger.info(f'resetting number of sessions to sample: {sessions_train.shape}')

    logger.info(f'Creating vectorized training environment: num envs: {cpu_count}')
   
    citizen_science_vec = SubprocVecEnv([lambda: CitizenScienceEnv(sessions_train, train_data, n_sequences, n_features) for _ in range(2)])

    """
    Eval environment is not used in training and is used after training to evaluate the agent
    """    
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
        os.path.join(base_path, 'checkpoints'),
    )
 
    monitor_train = VecMonitor(citizen_science_vec)
    agent = A2C(
        'MlpPolicy',
        monitor_train,
        verbose=1,
        device=args.device,
        tensorboard_log=tensorboard_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // 2,
        name_prefix='a2c',
        save_path=checkpoint_dir,
        verbose=1
    )
        
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=n_episodes, verbose=1)
    
    dist_callback = DistributionCallback()
    callback_list = CallbackList([dist_callback, callback_max_episodes, checkpoint_callback])

    logger.info(pformat([
        'n_episodes: {}'.format(n_episodes),
        'read_path: {}'.format(read_path),
        'n_files: {}'.format(n_files),
        'n_sequences: {}'.format(n_sequences),
        'n_features: {}'.format(n_features),
        'total_timesteps: {}'.format(dataset.shape[0] -1),
        'device: {}'.format(device),
        'tensorboard_dir: {}'.format(tensorboard_dir),
        'checkpoint_dir: {}'.format(checkpoint_dir)
    ]))

    agent.learn(
        total_timesteps=int(10e7),
        log_interval=100, 
        progress_bar=True,
        callback=callback_list
    )
    
if __name__ == '__main__':

    args = parse_args()
    main(args)
