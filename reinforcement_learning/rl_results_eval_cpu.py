import argparse
import logging
import os
from datetime import datetime
from functools import reduce
from pprint import pformat
from typing import Callable
import boto3
import random
import numpy as np
import pandas as pd
import torch
from callback import DistributionCallback
from environment_eval import CitizenScienceEnv
from policies.cnn_policy import CustomConv1dFeatures
from rl_util import setup_data_at_window
from rl_constant import LABEL, METADATA, OUT_FEATURE_COLUMNS, PREDICTION_COLS, RL_STAT_COLS
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
import tqdm
import json

ALL_COLS = LABEL + METADATA + OUT_FEATURE_COLUMNS  + PREDICTION_COLS

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
import zipfile
S3_BASELINE_PATH = 'dissertation-data-dmiller'
USER_INDEX = 1
SESSION_INDEX = 2
CUM_SESSION_EVENT_RAW = 3
TIMESTAMP_INDEX = 11
TRAIN_SPLIT = 0.7
N_SEQUENCES = 40
EVAL_SPLIT = 0.15

global logger

logger = logging.getLogger('rl_results_eval')
logger.setLevel(logging.INFO)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='datasets/rl_ready_data')
    parse.add_argument('--read_path_conv', type=str, default='datasets/rl_ready_data_conv')
    parse.add_argument('--write_path', type=str, default='datasets/rl_results')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--n_sequences', type=int, default=40)
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--window', type=int, default=2)
    parse.add_argument('--run_time', type=str)
    
    args = parse.parse_args()
    return args


def find_s3_candidate(client, feature_extractor, lstm, run_time):
    
    if lstm == 'seq_20':
        lstm = 'seq_40'
    folder_prefix = os.path.join(
        'reinforcement_learning_incentives',
        'n_files_30',
        f'{feature_extractor}_{lstm}',
        'results',
        run_time,
        'checkpoints',
    )
    
    logger.info(f'Looking for files in {folder_prefix}')
    
    files = [
        {
            'key': file['Key'],
            'last_modified': file['LastModified'],
        }
        for file in client.list_objects_v2(Bucket=S3_BASELINE_PATH, Prefix=folder_prefix)['Contents']
    ]
    
    s3_candidate = max(files, key=lambda x: x['last_modified'])['key']
    
    logger.info(f'Found candiate: {s3_candidate}')
    
    return s3_candidate

def get_policy(client, feature_extractor, lstm, run_time, algo):
    
    
    s3_candidate = find_s3_candidate(client, feature_extractor, lstm, run_time)
    
    model_base_path, download_path = (
        os.path.join('reinforcement_learning_incentives', f'{feature_extractor}_{lstm}'),
        os.path.join('reinforcement_learning_incentives', f'{feature_extractor}_{lstm}', f'{algo}.zip') 
    )
    
    if not os.path.exists(model_base_path):
        logger.info(f'Creating directory {model_base_path}')
        os.makedirs(model_base_path)
        client.download_file(S3_BASELINE_PATH, s3_candidate, download_path)
        logger.info(f'Loading model from {s3_candidate} to {download_path}')

    logger.info(f'Checkpoint load path: {download_path}')
    return download_path
        
def _lstm_loader(lstm):
    if lstm:   
        return LABEL if lstm == 'label' else ['seq_20']
    return []

def run_session(dataset, feature_meta, model, out_features, n_sequences):
    subset = dataset[
        (dataset['user_id'] == feature_meta['user_id']) &
        (dataset['session_30_raw'] == feature_meta['session_30_raw'])
    ]
    
    env = CitizenScienceEnv(subset, out_features, n_sequences)
    step = env.reset()
    done = False
    while not done:
        action, _states = model.predict(step, deterministic=True)
        step, rewards, done, info = env.step(action)
    return info

import pdb
def run_experiment(model, dataset, out_features, n_sequences):
    
    dataset = dataset.loc[:,~dataset.columns.duplicated()].copy()
    info_container = []
    unique_sessions = dataset[['user_id', 'session_30_raw']].drop_duplicates() 
    logger.info(f'Running experiment with {model}: n_session={len(unique_sessions)}')
    
    unique_sessions_bar = tqdm.tqdm(unique_sessions.iterrows(), total=len(unique_sessions))
    for _, session in enumerate(unique_sessions_bar):
        indx, session = session
        user_id, session_30_raw = int(session['user_id']), int(session['session_30_raw'])
        unique_sessions_bar.set_description(f'U {user_id} S {session_30_raw}:')
        session_meta = run_session(dataset, {'user_id': user_id, 'session_30_raw': session_30_raw}, model, out_features, n_sequences)
        info_container.append(session_meta)
        if _ > 10:
            break
    return info_container
        
    
        

def get_dataset(read_path, conv_path, n_files, window):
    
    conv_path, read_path = (
        os.path.join(conv_path, f'files_used_{n_files}'),
        os.path.join(read_path, f'files_used_{n_files}', 'predicted_data.parquet')
    )

    if not os.path.exists(conv_path):
        logger.info(f'Creating directory {conv_path}')
        os.makedirs(conv_path)
    
    conv_path = os.path.join(conv_path, f'rl_conv_{window}.parquet')
    
    if not os.path.exists(conv_path):
        logger.info(f'Convolutional dataset not found at {conv_path}: creating')
        logger.info(f'Getting dataset from {read_path}')
        df = pd.read_parquet(read_path)
        df = setup_data_at_window(df, window)
        logger.info(f'Saving convolutional dataset to {conv_path}')
        df.to_parquet(conv_path)
    else:
        logger.info(f'Loading convolutional dataset from {conv_path}')
        df = pd.read_parquet(conv_path)
        
    logger.info(f'Dataset loaded: {df.shape}')
    
    return df


def run_exp_wrapper(args, df, write_path):
        policy_weights = get_policy(client, args['feature_extractor'].lower(), args['lstm'], args['run_time'], args['algo'])
        print(policy_weights)
        all_features, out_features = (
            METADATA + OUT_FEATURE_COLUMNS + RL_STAT_COLS + _lstm_loader(args['lstm']),
            OUT_FEATURE_COLUMNS + _lstm_loader(args['lstm'])
        )
        df = df[all_features]
        env = CitizenScienceEnv(df, out_features, 40)
        
        if args['feature_extractor'].lower() == 'cnn':
            CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1, 21)
            logger.info(f'Using custom CNN feature extractor')
            policy_kwargs = dict(
                features_extractor_class=CustomConv1dFeatures,
                net_arch=[10]
            )
        
            model = DQN(policy='CnnPolicy', env=env, policy_kwargs=policy_kwargs)
            model.set_parameters(policy_weights)
            
        experiment = run_experiment(model, df, out_features, N_SEQUENCES)
        experiemnt_df = pd.DataFrame(experiment)
        
        logger.info(f'Finished experiment: {args}')
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        
        write_path = os.path.join(
            write_path,
            f'{args["algo"]}_{args["feature_extractor"]}_{args["lstm"]}_{args["run_time"]}.parquet'   
        )
        
        logger.info(f'Writing experiment to {write_path}')
        
        experiemnt_df.to_parquet(write_path)
        
    
    
     
def main(args):
    
    global client
    client = boto3.client('s3')


    logger.info('Starting offlline evaluation of RL model')
    
    read_path, conv_path, n_files, device,  window, write_path = (
        args.read_path,
        args.read_path_conv,
        args.n_files, 
        args.device, 
        args.window, 
        args.write_path
    )
    
    
    df = get_dataset(read_path, conv_path, n_files, window)
    with open('rl_policies.json') as f:
        rl_meta = json.load(f)
    
    for r in rl_meta:
        run_exp_wrapper(r, df, write_path)
        
    
    return


    policy_weights = get_policy(client, feature_extractor, lstm, run_time, algo)
    
    logger.info(f'Loading data from {read_path}')
    
    logger.info(f'Loaded data with shape {df.shape}')
    env = CitizenScienceEnv(df, out_features, n_sequences)
    logger.info(f'Loaded data with shape {df.shape}')
    
    if feature_extractor == 'cnn':
        CustomConv1dFeatures.setup_sequences_features(n_sequences + 1, len(out_features))
        logger.info(f'Using custom CNN feature extractor')
        policy_kwargs = dict(
            features_extractor_class=CustomConv1dFeatures,
            net_arch=[10]
        )
        
        model = DQN(policy='CnnPolicy', env=env, policy_kwargs=policy_kwargs)
        model.set_parameters(policy_weights)
    
    
    logger.info(f'Getting evaluation with following parameters')
    logger.info(pformat({
        'algo': algo,
        'feature_extractor': feature_extractor,
        'lstm': lstm,
        'n_sequences': n_sequences,
        'n_files': n_files,
        'window': window,
        'run_time': run_time,
         
    }))
    
    experiments = run_experiment(model, df, out_features, n_sequences, device, lstm)
    
    logger.info(f'Experiments ran')
    
    experiments_df = pd.DataFrame(experiments) 
    
    if not os.path.exists(args.write_path):
        os.makedirs(args.write_path)
        
    write_path = os.path.join(
        args.write_path,
        f'{algo}_{feature_extractor}_{lstm}_{run_time}.parquet'
    )
    
    logger.info(f'Writing results to {write_path}')
    
    experiments_df.to_parquet(write_path)

    


if __name__ == '__main__':
    args = parse_args()
    main(args)