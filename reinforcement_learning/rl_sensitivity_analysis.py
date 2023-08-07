import logging
import os
import re

import boto3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
global logger, client
logger = logging.getLogger(__name__)
client = boto3.client('s3')
import argparse
import glob

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from environment import CitizenScienceEnv
from environment_q2 import CitizenScienceEnvQ2
from itertools import product
from rl_constant import FEATURE_COLUMNS, LOAD_COLS, METADATA, RL_STAT_COLS
from pprint import pprint, pformat

MIN_MAX_RANGE = (10, 90)
from tqdm import tqdm

N_SEQUENCES = 15

S3_BASELINE_PATH = 'dissertation-data-dmiller'

SENSITIVITY_PARAMS = {
    "window": (.8,  .6),
    "mid": {.15, .04},
    "large": {.3, .09},
}

MODEL_PARAMS = {
    'dqn_pred_cnn': DQN,
    'a2c_pred_cnn': A2C,
}

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--algo', type=str, default='dqn_pred_cnn')
    parse.add_argument('--run_date', type=str, default='2023-06-22_14-22-36')
    parse.add_argument('--write_path', type=str, default='rl_evaluation')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--q2', default=1, type=int)
    args = parse.parse_args()
    return args


def find_s3_candidate(algo, run_date):
    
    folder_prefix = os.path.join(
        'experiments',
        "q2",
        algo,
        run_date,
        'checkpoints'
    )

    
    logger.info(f'Looking for files in {folder_prefix}')
    
    files = [
        {
            'key': file['Key'],
            'last_modified': file['LastModified'],
            'check_index': int(re.sub('[^0-9]', '', file['Key'].split('/')[-1]))
        }
        for file in client.list_objects_v2(Bucket=S3_BASELINE_PATH, Prefix=folder_prefix)['Contents']
    ]
    
    s3_candidate = sorted(files, key=lambda x: x['check_index'])[-1]['key']
    


    
    logger.info(f'Found candiate: {s3_candidate}')
    
    return s3_candidate

def get_policy(algo, run_date):
        
    s3_candidate = find_s3_candidate(algo, run_date)
    if not os.path.exists(os.path.dirname(s3_candidate)):
        logger.info(f'Creating policy path {os.path.dirname(s3_candidate)}')
        
        os.makedirs(os.path.dirname(s3_candidate), exist_ok=True)
       
    # return s3_candidate 
    client.download_file(S3_BASELINE_PATH, s3_candidate, s3_candidate)
    return s3_candidate
        
def simplify_experiment(vectorized_df):
    df_container = []
    vectorized_df = [
        df[(df['session_size'] >= MIN_MAX_RANGE[0]) & (df['session_size'] <= MIN_MAX_RANGE[1])] for df in vectorized_df
    ]
     
    for df in vectorized_df:
        df['project_count'] = 0
        df_container.append(df)
    
    return df_container


def _label_or_pred(algo):
    if 'label' in algo:
        return 'label'
    elif 'pred' in algo:
        return 'pred'
    else:
        return None
   

def run_sensitivity_analysis(env_datasets, policy_path, feature_cols, param_combos, q2, algo): 
    
    p_bar = tqdm(param_combos, unit='item')
    out_df_container = []
    for combo in p_bar:
        params = {
            "window": combo[0].round(2),
            "mid": combo[1].round(2),
            "large": combo[2].round(2),
            "soc_freq": int(combo[3])
        }
        p_bar.set_description(f'Running combo {params}')
        
        print(f'running q2: {q2}')
        vec_monitor = VecMonitor(DummyVecEnv([lambda: CitizenScienceEnvQ2(dataset, feature_cols, N_SEQUENCES, params) for dataset in env_datasets]))
        
        model = MODEL_PARAMS[algo].load(
            policy_path,
            env=vec_monitor,
            verbose=0,
        )
        
        evaluate_policy(
            model,
            model.get_env(),
            deterministic=False,
            n_eval_episodes=10
        )
        
        dists = model.get_env().get_attr('episode_bins')
        values_to_log = [item for sublist in dists for item in sublist if len(sublist) > 0]
        out_df = pd.DataFrame(values_to_log)
        out_df['window'] = params['window']
        out_df['mid'] = params['mid']
        out_df['large'] = params['large']
        out_df_container.append(out_df)
        break
    
    
    return pd.concat(out_df_container)
        

def rebatch_data(vectorized_df):
    df_sublist = []
    for i in range(0, len(vectorized_df), 10):
        df_sublist.append(pd.concat(vectorized_df[i:i+10], ignore_index=True))
    return df_sublist  


def main(args):
    algo, run_date, write_path, n_files, q2 = args.algo, args.run_date, args.write_path, args.n_files, args.q2

    params_window = np.arange(*SENSITIVITY_PARAMS['window'], -.02).tolist()
    params_mid = np.arange(*SENSITIVITY_PARAMS['mid'], -.01).tolist()
    params_large = np.arange(*SENSITIVITY_PARAMS['large'], -.02).tolist()
    social_params = [3, 5, 7]
    
    logger.info(f'Window params: {params_window}')
    logger.info(f'Mid params: {params_mid}')
    logger.info(f'Large params: {params_large}')
    logger.info(f'Social params: {social_params}')
    
    param_combos = np.array(list(product(params_window, params_mid, params_large, social_params)))
    logger.info(f'Combination parameters obtained: {param_combos.shape}, running monte carlo simulation on 200 random samples')
    param_combos = param_combos[np.random.choice(param_combos.shape[0], 200, replace=False), :]
   
    policy_path = get_policy(algo, run_date)
    logger.info(f'Policy path downloaded, evaluating experiment: {policy_path}')
    
    read_path = os.path.join('rl_ready_data_conv', f'files_used_{n_files}', 'window_1', 'batched_eval')
    files_to_read = glob.glob(os.path.join(read_path, '*.parquet'))
    logger.info(f'Found {len(files_to_read)} files to read')
    env_datasets = [
        pd.read_parquet(file) for file in files_to_read
    ]

    env_datasets = simplify_experiment(env_datasets)
    
    if 'a2c' in algo:
        logger.info(f'Rebatching data for A2C')
        env_datasets = rebatch_data(env_datasets)
        
    feature_cols = FEATURE_COLUMNS + [_label_or_pred(algo)] if _label_or_pred(algo) else FEATURE_COLUMNS
    logger.info(f'Length of features: {len(feature_cols)}')
    logger.info(f'Running sensitivity analysis per monte carlo simulation')
    logger.info(pformat(
        {
            'algo': algo,
            'q2': q2==1,
            'n_envs': len(env_datasets),
            'n_features': len(feature_cols),
            
        }
    ))
    sensitivity_df = run_sensitivity_analysis(env_datasets, policy_path, feature_cols, param_combos, q2==1, algo)

    
    write_path = os.path.join(write_path, f'sensitivity_analysis', f'{algo}.parquet')
    if not os.path.exists(os.path.dirname(write_path)):
        logger.info(f'Creating write path {os.path.dirname(write_path)}')
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
    
    logger.info(f'Writing sensitivity analysis to {write_path}')
    sensitivity_df.to_parquet(write_path)

    


if __name__ == "__main__":
    args = parse_args()
    main(args)