import argparse
import numpy as np
from npz_extractor import NPZExtractor
import torch
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
from environment import CitizenScienceEnv
from distributions import DISTRIBUTIONS
import logging
from typing import List, Dict
TIMESTAMP_INDEX = 6
USER_INDEX = 3
N_EVENTS_INDEX = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/torch_ready_data_4/')
    parser.add_argument('--n_files', type=str, default='5')
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=17)
    parser.add_argument('--return_distribution', type=str, default='stack_overflow_v1')
    
    args = parser.parse_args()
    return args

def create_user_trajectories(unique_users: torch.tensor, sequences: torch.tensor) -> Dict[int, torch.tensor]:
    """
    Returns a dictionary of user_id to their respective trajectories.
    Sorts sequences by time of event.
        Example: {
            user_id: [
                [[event_1, event_2]]   
            ]
        }
    """
    
    trajectories = {}
    for user in unique_users:
        trajectory = sequences[sequences[:, USER_INDEX] == user]
        trajectory = trajectory[trajectory[:, N_EVENTS_INDEX].sort()[1]]
        trajectory[:, N_EVENTS_INDEX] = torch.arange(1, trajectory.shape[0] + 1)
        trajectories[user.int().item()] = trajectory
    
    return trajectories, unique_users[0].int().item()
        
def _extract_reward(comp_bin_dict):
    rewards = [
        comp_bin_dict[k]['n_events'] for k in comp_bin_dict
    ]
    return sum(rewards) 

def main(args):
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logger = logging.getLogger(__name__)
    
    read_path, n_files, n_sequences, n_features = args.read_path, args.n_files, args.n_sequences, args.n_features
    
    npz_extractor = NPZExtractor(
        read_path,
        n_files,
        n_sequences,
        None,
        10000
    )
    
    dataset = torch.tensor(np.concatenate(npz_extractor.get_dataset_pointer(), axis=1))
    unique_users =  torch.unique(dataset[:, 3].unique())
    user_trajectories, initial_user = create_user_trajectories(unique_users, dataset)
    initial_clickstream = user_trajectories[initial_user]
    logger.info(f'Number of user trajectories: {len(user_trajectories)}')
    
    citizen_science_env = CitizenScienceEnv(unique_users, user_trajectories, initial_user, n_sequences, n_features)
    baseline_rewards = dataset.shape[0]
    environment_comp = False
    
    while not environment_comp:
        action = 1
        environment_comp, next_state = citizen_science_env.step(1)
    
    cumalitive_reward = _extract_reward(citizen_science_env.comp_bin)
    logger.info(f'Baseline reward: {baseline_rewards}, cumalitive reward: {cumalitive_reward}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
