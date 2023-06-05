from typing import Dict, List, Type, Union

import gym
import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn


class CustomConv1dFeatures(BaseFeaturesExtractor):
    
    @classmethod
    def setup_sequences_features(cls, n_sequences, n_features):
        cls.n_sequences = n_sequences
        cls.n_features = n_features
        
    
    def __init__(self, observation_space: spaces.Box, features_dim=20):
        super().__init__(observation_space, features_dim)
        
        
        self.cnn_1 = nn.Sequential(
            nn.Conv1d(self.n_features, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ReLU(),
            
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ReLU(),
            
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            
            nn.AvgPool1d(2)
        )
        
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(self.n_features*2, self.n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            
            nn.Conv1d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU()
        )
        
        self.act = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            out_shape = self.act(self.cnn_2(self.cnn_1(torch.zeros((1, self.n_features, self.n_sequences))))).shape[1]
            self.linear = nn.Linear(out_shape, features_dim)
    
    def forward(self, obs):
        out = self.cnn_1(obs)
        out = self.cnn_2(out)
        out = self.act(out)
        return self.linear(out)


        