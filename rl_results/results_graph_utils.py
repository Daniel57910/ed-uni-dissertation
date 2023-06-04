import pandas as pd
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict
from statsmodels.tsa.filters.hp_filter import hpfilter
import re
from datetime import datetime

LOG_DIR = 'rl_stats/question_1'

def tensorboard_results(log_matrix: Dict, scalar_key: str):
    log_df = {}
    for label, log_dir in log_matrix.items():
        print(f'Getting {label} results')
        events = EventAccumulator(log_dir)
        events.Reload()
        reward = events.Scalars(scalar_key)
        df = pd.DataFrame({
                'wall_time': [x.wall_time for x in reward], # 'step': [x.step for x in loss],
                'step': [x.step for x in reward],
                'reward': [x.value for x in reward],
            })
        df['wall_time'] = pd.to_datetime(df['wall_time'], unit='s')
        log_df[label] = df
        file_prefix = re.sub(r'[^a-zA-Z0-9]', '_', label)
        scalar_key = re.sub(r'[^a-zA-Z0-9]', '_', scalar_key)
        file_path = os.path.join(LOG_DIR, f'{file_prefix}_{scalar_key}.csv')
        print(f'Writing to {file_path}')
        # df.to_csv(file_path, index=False)
        
    return log_df


def tensorboard_results_loss(log_matrix: Dict):
    log_df = {}
    for label, log_dir in log_matrix.items():
        print(f'Getting {label} results')
        events = EventAccumulator(log_dir)
        events.Reload()
        reward = events.Scalars('train/loss')
        df = pd.DataFrame({
                'wall_time': [x.wall_time for x in reward], # 'step': [x.step for x in loss],
                'step': [x.step for x in reward],
                'loss': [x.value for x in reward],
            })
        df['wall_time'] = pd.to_datetime(df['wall_time'], unit='s')
        log_df[label] = df
        
    return log_df

        

def convolve_and_apply_hp_filter(df, col, window, lamb):
    df = df.set_index('wall_time') \
        .resample(f'{window}T') \
        .mean() \
        .reset_index()
    
    df['step'] = df['step'].astype(int)
    
    cycle, trend = hpfilter(df[col], lamb)
    df[f'hp_{col}'] = trend
    return df

def fetch_end_filtered(df_matrix, col):
    filtered_df = []
    for k, v in df_matrix.items():
        filtered_df.append(
            {
                "Exp Name": k,
                col: v.iloc[-1][col]
            }
        )
    return pd.DataFrame(filtered_df)


def fetch_session_stats(log_matrix: Dict):
    for label, log_dir in log_matrix.items():
        print(f'Getting {label} results')
        scalar_container = []
        events = EventAccumulator(log_dir)
        
        events.Reload()
        print(events.Scalars)
        for scalar in ['inc_time', 'reward', 'session_minutes', 'sim_minutes']:
            print(f'Getting {scalar} results')
            result = events.Scalars(f'time/{scalar}')
            scalar_df = pd.DataFrame({
                    'wall_time': [x.wall_time for x in result], # 'step': [x.step for x in loss],
                    'step': [x.step for x in result],
                    scalar: [x.value for x in result],
                })
            scalar_df['wall_time'] = pd.to_datetime(scalar_df['wall_time'], unit='s')
            scalar_container.append(scalar_df)
        df = pd.concat(scalar_container, axis=1, join='outer')
        df = df.loc[:,~df.columns.duplicated()]
        df['distance_sess_reward'] = df['session_minutes'] - df['reward']
        df['distance_sim_reward'] = df['reward'] - df['sim_minutes']
        df['distance_inc_pl'] = df['sim_minutes'] - df['inc_time']
        
        label_prefix = re.sub(r'[^a-zA-Z0-9]', '_', label).lower()
        df.to_csv(f'rl_stats/q_1_cnn/experiment_stats/{label_prefix}_stats.csv', index=False)
        log_matrix[label] = df
    return log_matrix


def fetch_end_evaluation(df_matrix):
    end_evaluation_container = []
    for label, df in df_matrix.items():
        end_evaluation_container.append(
            {
                "Exp Name": label,
                'HP Reward': df.iloc[-1]['hp_reward'],
                'HP Distance Session': df.iloc[-1]['hp_distance_sess_reward'],
                'HP Inc Placements': df.iloc[-1]['hp_distance_inc_pl'],
            }
        )
   
   
    return pd.DataFrame(end_evaluation_container)
 
SUMMARY_STAT_COLS = [
    'distance_sess_reward', 
    'distance_sim_reward', 
    'distance_inc_pl', 
    'reward', 
    'session_minutes', 
    'sim_minutes'
]
