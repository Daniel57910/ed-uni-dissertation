import gym
from gym import spaces
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomConv1dPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 2):
        super().__init__(observation_space, features_dim)
        
        
        self.cnn_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),
        )
        
        self.cnn_2 = nn.Sequential(
            nn.conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        self.act = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Flatten(),
        )
        
        self.m_pool = nn.MaxPool1d(5)
       
        with torch.no_grad():
            n_flatten = self.act(self.cnn_1(torch.zeros(1, 1, 220))).shape[1] 
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU()
            )
        

    
    def forward(self, obs):
        out = self.cnn_1(obs)
        out = self.cnn_2(out)
        out = self.act(out)
        out = self.linear(out)
        return out