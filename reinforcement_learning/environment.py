import gym
import argparse
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from npz_extractor import NPZExtractor
import pdb

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
    
    def __init__(self, citizen_science_dataset, n_sequences, n_features) -> None:
        super(CitizenScienceEnv, self).__init__()
        self.citizen_science_dataset = citizen_science_dataset
        """
        Observation Space defines the current state, or current window of a users work session.
        """
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Box(low=0, high=1, shape=(n_sequences + 1, n_features + 1), dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(1)
        self.n_sequences = n_sequences
        self.n_features = n_features
        self.total_events = self.citizen_science_dataset.shape[1]
        self.current_step = 0
        
    def _extract_features(self, tensor):
        """
        Returns:
            label regarding user ending session in next 30 minutes
            total events associated with user
            preprocessed features
        """


        label, total_events, features, shifters = (
            tensor[0], tensor[1], tensor[5:5+self.n_features], tensor[5+self.n_features:]
        )


        shifters = torch.reshape(shifters, (self.n_sequences, self.n_features + 1))
        shifter_features = shifters[:, 1:]
        features = torch.unsqueeze(features, 0)
        features = torch.cat((features, shifter_features), dim=0)
        label = torch.unsqueeze(label, 0).repeat(features.shape[0], 1)
        event = torch.cat((label, features), dim=1)
        return event
    
    def step(self, action):
        if self.current_step >= self.total_events:
            done = True
        else:
            self.current_step += 1
            done = False
        
        print(f'Current Step: {self.current_step}, action: {action}')
        tensor = self.citizen_science_dataset[self.current_step]
        features = self._extract_features(tensor)
        reward = self.current_step
        
        return features, reward, done, {}
    
    def reset(self):
        self.current_step = 0
        return self._extract_features(self.citizen_science_dataset[self.current_step])


    def render(self, mode='human', close=False):
        print(f'Current Step: {self.current_step}')
        
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
         
    citizen_science_env = CitizenScienceEnv(dataset, n_sequences, n_features)
    tensor = citizen_science_env._extract_features(dataset[0])
    print(f'Shape of tensor: {tensor.shape}')
    for i in range(10):
        print(f'Action: {i}')
        citizen_science_env.step(1)
    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    