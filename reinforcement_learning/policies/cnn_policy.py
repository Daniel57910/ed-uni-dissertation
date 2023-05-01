from typing import Dict, List, Type, Union
import gym
from gym import spaces
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor

class CustomConv1dFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, n_sequences=11, n_features=21, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        
        self.cnn_1 = nn.Sequential(
            nn.Conv1d(n_features, n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_sequences),
            nn.ReLU(),
            nn.Conv1d(n_features*2, n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_sequences),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(n_features, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_sequences // 2),
            nn.ReLU()
        )
        
        self.act = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
    
    def forward(self, obs):
        out = self.cnn_1(obs)
        out = self.cnn_2(out)
        out = self.act(out)
        return out

# class CustomConv1dMLP(MlpExtractor):
#     def __init__(self, feature_dim: int, net_arch: List[int]  Dict[str, List[int]], activation_fn: Type[Module], device: device | str = "auto") -> None:
#         super().__init__(feature_dim, net_arch, activation_fn, device)
        
        
# class TestConv1(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
    
#         self.cnn_1 = nn.Sequential(
#             nn.Conv1d(21, 42, kernel_size=3, padding=1),
#             nn.BatchNorm1d(11),
#             nn.ReLU()
#         )
    
#     def forward(self, x):
#         x = self.cnn_1(x)
#         return x