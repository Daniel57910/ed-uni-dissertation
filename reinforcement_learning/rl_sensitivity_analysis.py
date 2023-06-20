import os
import logging
import re
import boto3
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
global logger, client
logger = logging.getLogger(__name__)
client = boto3.client('s3')
import numpy as np
import argparse
from rl_constant import FEATURE_COLUMNS, METADATA, RL_STAT_COLS, LOAD_COLS
from itertools import combinations, product
from stable_baselines3 import DQN
from environment import CitizenScienceEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import pandas as pd
import glob
MIN_MAX_RANGE = (10, 90)
from tqdm import tqdm
N_SEQUENCES = 15

S3_BASELINE_PATH = 'dissertation-data-dmiller'

SENSITIVITY_PARAMS = {
    "window": (.8,  .6),
    "mid": {.15, .04},
    "large": {.3, .09},
    
}

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--algo', type=str, default='dqn_pred_cnn'),
    parse.add_argument('--run_date', type=str, default='2023-06-13_16-11-42'),
    parse.add_argument('--write_path', type=str, default='rl_evaluation'),
    parse.add_argument('--n_files', type=int, default=2),
    args = parse.parse_args()
    return args


def find_s3_candidate(algo, run_date):
    
    folder_prefix = os.path.join(
        'experiments',
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
    vectorized_df = [
        df[(df['session_size'] >= MIN_MAX_RANGE[0]) & (df['session_size'] <= MIN_MAX_RANGE[1])] for df in vectorized_df
    ]

    return vectorized_df


def _label_or_pred(algo):
    if 'label' in algo:
        return 'label'
    elif 'pred' in algo:
        return 'pred'
    else:
        return None
   

def run_sensitivity_analysis(env_datasets, policy_path, feature_cols, param_combos): 
    
    p_bar = tqdm(param_combos, unit='item')
    out_df_container = []
    for combo in p_bar:
        params = {
            "window": combo[0].round(2),
            "mid": combo[1].round(2),
            "large": combo[2].round(2)
        }
        p_bar.set_description(f'Running combo {params}')
        
        vev_envs = DummyVecEnv([lambda: CitizenScienceEnv(dataset, feature_cols, N_SEQUENCES, params) for dataset in env_datasets])
        
        vec_monitor = VecMonitor(vev_envs)
        
        model = DQN.load(
            policy_path,
            env=vec_monitor,
            verbose=0,
        )
        
        evaluate_policy(
            model,
            model.get_env(),
            deterministic=False,
            n_eval_episodes=1_000
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
        

   


        

def main(args):
    algo, run_date, write_path, n_files = args.algo, args.run_date, args.write_path, args.n_files

    params_window = np.arange(*SENSITIVITY_PARAMS['window'], -.02).tolist()
    params_mid = np.arange(*SENSITIVITY_PARAMS['mid'], -.01).tolist()
    params_large = np.arange(*SENSITIVITY_PARAMS['large'], -.02).tolist()
    
    logger.info(f'Window params: {params_window}')
    logger.info(f'Mid params: {params_mid}')
    logger.info(f'Large params: {params_large}')
    
    param_combos = np.array(list(product(params_window, params_mid, params_large)))
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
    feature_cols = FEATURE_COLUMNS + [_label_or_pred(algo)] if _label_or_pred(algo) else FEATURE_COLUMNS
    logger.info(f'Length of features: {len(feature_cols)}')
    logger.info(f'Running sensitivity analysis per monte carlo simulation')
    sensitivity_df = run_sensitivity_analysis(env_datasets, policy_path, feature_cols, param_combos)

    
    write_path = os.path.join(write_path, f'sensitivity_analysis', f'{algo}.parquet')
    if not os.path.exists(os.path.dirname(write_path)):
        logger.info(f'Creating write path {os.path.dirname(write_path)}')
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
    
    logger.info(f'Writing sensitivity analysis to {write_path}')
    sensitivity_df.to_parquet(write_path)

    


if __name__ == "__main__":
    args = parse_args()
    main(args)