import gym
import argparse
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from npz_extractor import NPZExtractor
import pdb
USER_INDEX = 3
N_EVENT_INDEX = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, default='datasets/torch_ready_data_4/')
    parser.add_argument('--n_files', type=str, default='5')
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=17)
    
    args = parser.parse_args()
    return args

class CitizenScienceEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, unique_users, trajectories, initial_user, n_sequences, n_features) -> None:
        """
        trajectories: dictionary of user_id to their respective trajectories.
        n_sequences: number of sequences used for preprocessing.
        n_features: number of features used for preprocessing.
        """
        super(CitizenScienceEnv, self).__init__()
        self.unique_users = unique_users
        self.trajectories = trajectories
        self.comp_bin = {}
        self.comp_bin[initial_user] = {
            'n_events': 0,
            'award_2_events': False,
        }

        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Box(low=0, high=1, shape=(n_sequences + 1, n_features + 1), dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(1)
        self.n_sequences = n_sequences
        self.n_features = n_features
        self.user = initial_user
        self.user_index = 0
        self.current_step = 0
        self.dist = torch.distributions.Normal(loc=2, scale=0.5)
        
    def _extract_features(self, tensor):

        label, total_events, user_id, features, shifters = (
            tensor[0], tensor[1], tensor[3], tensor[5:5+self.n_features], tensor[5+self.n_features:]
        )

        shifters = torch.reshape(shifters, (self.n_sequences, self.n_features + 1))
        shifter_features = shifters[:, 1:]
        features = torch.unsqueeze(features, 0)
        features = torch.cat((features, shifter_features), dim=0)
        label = torch.unsqueeze(label, 0).repeat(features.shape[0], 1)
        event = torch.cat((label, features), dim=1)
        return user_id, total_events, event
    
    def step(self, action):
       
        print(f'Comp bin len: {len(self.comp_bin)}') 
        if len(self.comp_bin) == 58:
            print(sorted(self.comp_bin.keys()))
            print(sorted(self.unique_users))
            return True, None

        if self.user not in self.comp_bin:
            self._update_user()
        
        rewards = self.comp_bin[self.user]['n_events']
        if rewards > 2:
            self._update_user()
        
        trajectories = self.trajectories[self.user]
        historical_event = trajectories[trajectories[:, N_EVENT_INDEX] == self.comp_bin[self.user]['n_events']]
        self._take_action(action)
        next_state = self._calculate_next_state(historical_event)
        return False, next_state
        
            
    def _take_action(self, action):
        if action == 1 and self.comp_bin[self.user]['n_events'] == 2:
            self.comp_bin[self.user]['award_2_events'] = True
    
    def _calculate_next_state(self, historical_event):
        next_index = self.comp_bin[self.user]['n_events']
        if next_index < self.trajectories[self.user].shape[0]:
            self.comp_bin[self.user]['n_events'] += 1
            print(f'Calculating next state for user: {self.user} next index: {next_index}')
            return self.trajectories[self.user][next_index]
        else:
            print(f'Self user: {self.user} next index: {next_index} not in range: {self.trajectories[self.user].shape[0]}')
            probability_return = torch.exp(self.dist.log_prob(torch.tensor(next_index))) + self._guassian_noise()
            print(f'Probability of returning: {probability_return}')
            if probability_return > 0.5:
                next_state = historical_event + self._positive_gaussian_noise_vector(historical_event.shape[1])
                self.comp_bin[self.user]['n_events'] += 1
                next_state[:, N_EVENT_INDEX] = self.comp_bin[self.user]['n_events']
                self.trajectories[self.user] = torch.cat((self.trajectories[self.user], next_state), dim=0)
                return next_state
            else:
                self._update_user()
                current_trajcectory = self.trajectories[self.user]
                current_step = self.comp_bin[self.user]['n_events']
                print(f'Beginning new trajectory for user: {self.user} at step: {current_step}')
                return current_trajcectory[current_trajcectory[:, N_EVENT_INDEX] == current_step]

    def _update_user(self):
        self.user = self._get_next_user()
        self.comp_bin[self.user] = {
            'award_2_events': False,
            'n_events': 0,
        }
        
    def _guassian_noise(self):
        return torch.randn(1) * (0.1**0.5) 
    
    def _positive_gaussian_noise_vector(self, size):
        positive_gauss_noise = torch.abs(torch.randn(size) * (0.1**0.5)).unsqueeze(0)
        positive_gauss_noise[:, N_EVENT_INDEX] = 0
        return positive_gauss_noise

    def _get_next_user(self):
        self.user_index += 1
        next_user = self.unique_users[self.user_index].int().item()
        print(f'Updating user index: {self.user_index}, user: {next_user}')
        return next_user


        # tensor = self.citizen_science_dataset[self.current_step]
        # user_id, total_events, features = self._extract_features(tensor)
        # reward = self.current_step
        
        # return features, reward, done, {}
    
    def reset(self):
        self.current_step = 0
        return self._extract_features(self.citizen_science_dataset[self.current_step])


    def render(self, mode='human', close=False):
        print(f'Current Step: {self.current_step}')
    