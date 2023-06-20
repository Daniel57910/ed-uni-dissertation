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
import glob

LOG_DIR = 'dissertation_project_code/rl_results/dqn_csv'

EXPERIMENTS = {
    'dqn_pred_cnn': 'experiments/dqn_pred_cnn/2023-06-13_16-11-42/training_metrics',
    'dqn_label_cnn': 'experiments/dqn_label_cnn/2023-06-14_07-19-19/training_metrics',
    'dqn_none_cnn': 'experiments/dqn_None_cnn/2023-06-14_13-40-54/training_metrics',
    'dqn_pred_mlp': 'experiments/dqn_pred_mlp/2023-06-14_20-10-32/training_metrics',
    'dqn_pred_cnn_no_pen': 'experiments/dqn_pred_cnn/2023-06-20_10-38-22/training_metrics',
    'dqn_cnn_none_no_pen': None

}
EXPERIMENT_PATH = 'experiments'
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dqn_label_cnn')
    parser.add_argument('--exp_index', type=int, default=4)
    args = parser.parse_args()
    return args

def tensorboard_concatenate_rl_results():
    args = parse_args()
    
    for exp_name, exp_path in EXPERIMENTS.items():
        print('exp_name: ', exp_name)
        dataset = 'train' if 'train' in exp_path else 'test'

        print(f'file_path: {exp_path}')
        files = glob.glob(f'{exp_path}/*.csv')
        print(f'N files: {len(files)} for concatenation')

        df = pd.concat([pd.read_csv(f) for f in files])
    
        print(f'Files concaentated: no of sessoins: {len(df)}')
        df = df.drop(columns=['Unnamed: 0'])
        write_path = os.path.join(LOG_DIR, dataset)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        write_path = os.path.join(write_path, f'{exp_name}.csv')
        print(f'write_path: {write_path}')
        df.to_csv(write_path, index=False)

if __name__ == "__main__":
    tensorboard_concatenate_rl_results()