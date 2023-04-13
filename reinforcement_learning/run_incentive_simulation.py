import argparse
import numpy as np
from npz_extractor import NPZExtractor
import torch
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
from environment import CitizenScienceEnv
from callback import DistributionCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3 import PPO, A2C
import logging
USER_INDEX = 1
SESSION_INDEX = 2
TIMESTAMP_INDEX = 11
from stable_baselines3.common.logger import configure
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from pprint import pformat, pprint
import os
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--n_files', type=str, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=18)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--return_distribution', type=str, default='stack_overflow_v1')
    parser.add_argument('--agent', type=str, default='constant_20')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    return args

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


def run_reinforcement_learning_incentives(environment, logger, n_epochs=1):
    for epoch in range(n_epochs):
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
    
    read_path, n_files, n_sequences, n_features, n_epochs, device = (
        args.read_path, 
        args.n_files, 
        args.n_sequences, 
        args.n_features, 
        args.n_epochs, 
        args.device
    )
    
    npz_extractor = NPZExtractor(
        read_path,
        n_files,
        n_sequences,
        None,
        1000
    )
   
    logger.info(f'Starting experiment at {exec_time}') 
    logger.info(f'Extracting dataset from npz files to tensor' )
    dataset = np.concatenate(npz_extractor.get_dataset_pointer(), axis=1)

    logger.info(f'Dataset shape: {dataset.shape}: generating metadata tensor')
    sessions = generate_metadata(dataset, logger)
    logger.info(f'Metadata shape: {sessions.shape}')
    logger.info('Creating vectorized training and evaluation environments')
    
    citizen_science_vec = DummyVecEnv([lambda: CitizenScienceEnv(sessions, dataset, n_sequences, n_features) for _ in range(2)])
    citizen_science_vec_eval = DummyVecEnv([lambda: CitizenScienceEnv(sessions, dataset, n_sequences, n_features) for _ in range(2)])
    
    logger.info(f'Vectorized environments created, wrapping with monitor')
    
    monitor_train, monitor_eval = VecMonitor(citizen_science_vec), VecMonitor(citizen_science_vec_eval)
    base_path = os.path.join(
        'reinforcement_learning_incentives',
        f'n_files_{n_files}',
        'results',
        exec_time,
    ) 
    
    tensorboard_dir, checkpoint_dir = (
        os.path.join(base_path, 'training_metrics'),
        os.path.join(base_path, 'checkpoints')
    )
 
    agent = A2C(
        'MlpPolicy',
        monitor_train,
        verbose=0,
        tensorboard_log=tensorboard_dir,
    )
    
    eval_callback = EvalCallback(
        monitor_eval,
        best_model_save_path=checkpoint_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=4, verbose=0)
    
    dist_callback = DistributionCallback()
    callback_list = CallbackList([dist_callback, callback_max_episodes, eval_callback])

    logger.info(pformat([
        'n_epochs: {}'.format(n_epochs),
        'read_path: {}'.format(read_path),
        'n_files: {}'.format(n_files),
        'n_sequences: {}'.format(n_sequences),
        'n_features: {}'.format(n_features),
        'total_timesteps: {}'.format(dataset.shape[0] -1),
        'device: {}'.format(device),
        'agent type: {}'.format(args.agent),
        'return distribution: {}'.format(args.return_distribution),
        'tensorboard_dir: {}'.format(tensorboard_dir),
        'checkpoint_dir: {}'.format(checkpoint_dir)
    ]))
    
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)


    agent.learn(
        total_timesteps=int(10e7),
        log_interval=10, 
        progress_bar=True,
        callback=callback_list
    )
    



if __name__ == '__main__':
    args = parse_args()
    main(args)
