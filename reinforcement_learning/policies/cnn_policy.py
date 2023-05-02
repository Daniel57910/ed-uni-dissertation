from typing import Dict, List, Type, Union
import gym
from gym import spaces
import torch
from torch import nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomConv1dFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, n_sequences=11, n_features=21, features_dim=20):
        super().__init__(observation_space, features_dim)
        
        
        self.cnn_1 = nn.Sequential(
            nn.Conv1d(n_features, n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_features*2),
            nn.ReLU(),
            nn.Conv1d(n_features*2, n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(n_features, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_features // 2),
            nn.ReLU()
        )
        
        self.act = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(1),
        )
    
    def forward(self, obs):
        print(f'Feature extractor: obs shape on input: {obs.shape}')
        out = self.cnn_1(obs)
        print(f'Feature extractor: Obs shape after cnn_1: {out.shape}')
        out = self.cnn_2(out)
        print(f'Feature extractor: Obs shape after cnn_2: {out.shape}')
        out = self.act(out)
        print(f'Feature extractor: Obs shape after act: {out.shape}')
        return out


        

    