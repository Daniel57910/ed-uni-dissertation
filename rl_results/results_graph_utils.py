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
EXPERIMENT_PATH = 'experiments'
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dqn_label_cnn')
    parser.add_argument('--date', type=str, default='2023-06-09_14-14-21')
    parser.add_argument('--exp_index', type=int, default=4)
    args = parser.parse_args()
    return args

def tensorboard_concatenate_rl_results():
    args = parse_args()

    file_path = os.path.join(EXPERIMENT_PATH, args.model, args.date, 'training_metrics')
    print(f'file_path: {file_path}')
    files = glob.glob(f'{file_path}/*.csv')
    print(f'N files: {len(files)} for concatenation')

    df = pd.concat([pd.read_csv(f) for f in files])
    
    print(f'Files concaentated: no of sessoins: {len(df)}')
    df = df.drop(columns=['Unnamed: 0'])
    write_path = os.path.join(LOG_DIR, f'exp_{args.model}_{args.exp_index}.csv')
    print(f'write_path: {write_path}')
    df.to_csv(write_path, index=False)

    
if __name__ == "__main__":
    tensorboard_concatenate_rl_results()