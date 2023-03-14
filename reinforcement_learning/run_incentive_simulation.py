import argparse
import numpy as np
from npz_extractor import NPZExtractor
import torch
from environment import CitizenScienceEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/torch_ready_data_4/')
    parser.add_argument('--n_files', type=str, default='5')
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=17)
    
    args = parser.parse_args()
    return args

def create_user_trajectories(user_id, sequences):
    """
    Returns a dictionary of user_id to their respective trajectories.
        Example: {
            user_id: [
                [[event_1, event_2]]   
            ]
        }
    """
    trajectories = {
        u.int().item(): sequences[sequences[:, 3] == u] for u in user_id
    }
    
    return trajectories, user_id[0].int().item()

def _extract_reward(comp_bin_dict):
    rewards = [
        comp_bin_dict[k]['n_events'] for k in comp_bin_dict
    ]
    return sum(rewards) 
    
def main(args):
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
    user_trajectories, initial_user= create_user_trajectories(unique_users, dataset)
    print(f'Number of user trajectories: {len(user_trajectories)}')
    citizen_science_env = CitizenScienceEnv(unique_users, user_trajectories, initial_user, n_sequences, n_features)
    baseline_rewards = dataset.shape[0]
    environment_comp = False
    
    while not environment_comp:
        action = 1
        environment_comp, next_state = citizen_science_env.step(1)
    
    cumalitive_reward = _extract_reward(citizen_science_env.comp_bin)
    print(f'Baseline reward: {baseline_rewards}, cumalitive reward: {cumalitive_reward}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
