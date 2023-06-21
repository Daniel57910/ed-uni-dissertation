import logging
from typing import Dict, List, Type, Union

import gym
import torch
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn

global logger
logger = logging.getLogger(__name__)

class CustomConv1dFeatures(BaseFeaturesExtractor):
    
    @classmethod
    def setup_sequences_features(cls, n_sequences, n_features):
        cls.n_sequences = n_sequences
        cls.n_features = n_features
        
    
    def __init__(self, observation_space: spaces.Box, features_dim=24):
        super().__init__(observation_space, features_dim)
        
        
        self.cnn_1 = nn.Sequential(
            nn.Conv1d(self.n_features, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ELU(),
            
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ELU(),
            
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ELU()
            
        )
        
        self.conv_1_reshape = nn.Conv1d(
            self.n_features,
            self.n_features*2,
            kernel_size=1,
            padding=0
        
        )
        
        self.a_pool_1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.cnn_bottleneck_wide = nn.Sequential(
            nn.Conv1d(self.n_features*2, self.n_features*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*4),
            nn.ELU(),
            
            nn.Conv1d(self.n_features*4, self.n_features*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*4),
            nn.ELU(),
            
            nn.Conv1d(self.n_features*4, self.n_features*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*4),
            nn.ELU()   
        )
        
        self.conv_2_reshape = nn.Conv1d(
            self.n_features*2,
            self.n_features*4,
            kernel_size=1,
            padding=0
        )
        
        
        self.cnn_bottleneck_narrow = nn.Sequential(
            nn.Conv1d(self.n_features*4, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ELU(),
            
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ELU(),
            
            nn.Conv1d(self.n_features*2, self.n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features*2),
            nn.ELU()
        )
        
        self.conv_3_reshape = nn.Conv1d(
            self.n_features*4,
            self.n_features*2,
            kernel_size=1,
            padding=0
        )
        
        self.downsample = nn.Sequential(
            nn.Conv1d(self.n_features*2, self.n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features),
            nn.ELU(),
            
            nn.Conv1d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features),
            nn.ELU(),
            
            nn.Conv1d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features),
            nn.ELU()
        )
        
        self.conv_4_reshape = nn.Conv1d(
            self.n_features*2,
            self.n_features,
            kernel_size=1,
            padding=0
        )
                
        self.down_max = nn.Sequential(
            nn.Conv1d(self.n_features, self.n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features // 2),
            nn.ELU(),
            
            nn.Conv1d(self.n_features // 2, self.n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features // 2),
            nn.ELU(),
            
            nn.Conv1d(self.n_features // 2, self.n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_features // 2),
            nn.ELU(),
        )
        
        
        self.mpool_flat = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.down_max_reshape = nn.Conv1d(
            self.n_features,
            self.n_features // 2,
            kernel_size=1,
            padding=0
        )
        
        with torch.no_grad():
            sample_tensor = torch.zeros((1, self.n_features, self.n_sequences))
            sample_tensor = self.cnn_1(sample_tensor) + self.conv_1_reshape(sample_tensor)
            sample_tensor = self.a_pool_1(sample_tensor)
            sample_tensor = self.cnn_bottleneck_wide(sample_tensor) + self.conv_2_reshape(sample_tensor)
            sample_tensor = self.cnn_bottleneck_narrow(sample_tensor) + self.conv_3_reshape(sample_tensor)
            sample_tensor = self.downsample(sample_tensor) + self.conv_4_reshape(sample_tensor)
            sample_tensor = self.down_max(sample_tensor) + self.down_max_reshape(sample_tensor)
            mpool_flat_out = self.mpool_flat(sample_tensor)
            linear_in = mpool_flat_out.shape[1]
            self.final_out_linear = nn.Sequential(

                nn.Linear(linear_in, features_dim),
                nn.ELU()
            )

        



        

    def forward(self, obs):

        
        obs_cnn_1 = self.cnn_1(obs) + self.conv_1_reshape(obs)
        
        obs_cnn_1 = self.a_pool_1(obs_cnn_1)
        
        obs_cnn_2 = self.cnn_bottleneck_wide(obs_cnn_1) + self.conv_2_reshape(obs_cnn_1)
        
        obs_cnn_3 = self.cnn_bottleneck_narrow(obs_cnn_2) + self.conv_3_reshape(obs_cnn_2)
        
        obs_cnn_4 = self.downsample(obs_cnn_3) + self.conv_4_reshape(obs_cnn_3)
       
        obs_cnn_5 = self.down_max(obs_cnn_4) + self.down_max_reshape(obs_cnn_4)
        
        mpool_flat_out = self.mpool_flat(obs_cnn_5)
        
        return self.final_out_linear(mpool_flat_out)
        