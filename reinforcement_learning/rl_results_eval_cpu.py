import argparse
import logging
import os
from datetime import datetime
from functools import reduce
from pprint import pformat
from typing import Callable
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
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
from joblib import Parallel, delayed
import json
from pqdm.processes import pqdm
from policy_list import POLICY_LIST
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
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--read_path', type=str, default='rl_ready_data_conv')
    parse.add_argument('--write_path', type=str, default='datasets/rl_results')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--window', type=int, default=2)
    parse.add_argument('--data_part', type=str, default='train')
    parse.add_argument('--model', type=str, default='DQN')
    parse.add_argument('--feature_extractor', type=str, default='CNN')
    parse.add_argument('--lstm', type=str, default='seq_40')
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
    if lstm == 'no_pred': return [] 
    return LABEL if lstm == 'label' else ['seq_20']

def run_session(args):
    dataset, feature_meta, model, out_features, n_sequences, info_container = args
    _, feature_meta = feature_meta
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
    info_container.append(info)
    
    return info_container

def run_experiment(model, dataset, out_features, n_sequences):
    
    info_container = []
    
    dataset = dataset.loc[:,~dataset.columns.duplicated()].copy()
    info_container = []
    unique_sessions = dataset[['user_id', 'session_30_raw']].drop_duplicates() 
    logger.info(f'Running experiment with {model}: n_session={len(unique_sessions)}')
    
    args = [
        (dataset, feature_meta, model, out_features, n_sequences, info_container) for _, feature_meta in unique_sessions.iterrows()
    ]
    
    pqdm(args, run_session, n_jobs=8)
    
    return info_container

        
    
        

def get_dataset(conv_path, n_files, window, part='train'):
    
    
    conv_path =  os.path.join(conv_path, f'files_used_{n_files}')


    if not os.path.exists(conv_path):
        logger.info(f'Creating directory {conv_path}')
        os.makedirs(conv_path)
        
    
    conv_path = os.path.join(conv_path, f'window_{window}_{part}.parquet')
    
    if not os.path.exists(conv_path):
        logger.info(f'Convolutional dataset not found at {conv_path}: creating')
        logger.info(f'Getting dataset from bucket: {S3_BASELINE_PATH}, key: {conv_path}')
        client.download_file(S3_BASELINE_PATH, conv_path, conv_path)
        

    logger.info(f'Loading convolutional dataset from {conv_path}')
    df = pd.read_parquet(conv_path)
        
    logger.info(f'Dataset loaded: {df.shape}')
    
    return df


def run_exp_wrapper(args, df):
    
        logger.info(f'Getting policy for {args["feature_extractor"].lower()}, {args["lstm"]}, {args["run_time"]}, {args["algo"]}')
        policy_weights = get_policy(client, args['feature_extractor'].lower(), args['lstm'], args['run_time'], args['algo'])
        
        logger.info(f'Policy weights loaded')

        all_features, out_features = (
            METADATA + OUT_FEATURE_COLUMNS + RL_STAT_COLS + _lstm_loader(args['lstm']),
            OUT_FEATURE_COLUMNS + _lstm_loader(args['lstm'])
        )
        
        logger.info(f'Vectorizing dataset: {df.shape}')
        df = df[all_features]
        df = df.loc[:,~df.columns.duplicated()]    
        
        citizen_science_vec = DummyVecEnv(
            [lambda: CitizenScienceEnv(df, out_features, 40) 
             for i in range(10)]
        )
    
        logger.info(f'Vectorized dataset, setting policy weights')
        if args['feature_extractor'].lower() == 'cnn':
            CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1, 21)
            logger.info(f'Using custom CNN feature extractor')
            policy_kwargs = dict(
                features_extractor_class=CustomConv1dFeatures,
                net_arch=[10]
            )
        
            model = DQN(policy='CnnPolicy', env=citizen_science_vec, policy_kwargs=policy_kwargs)
        else:
            model = DQN(policy='MlpPolicy', env=citizen_science_vec)
        
        model.set_parameters(policy_weights)
        
        logger.info(f'Policy weights set, running experiment')
        logger.info(f'Evaluation beginning: {args["feature_extractor"]} {args["lstm"]} {args["algo"]} n_episodes=10')

        evaluate_policy(
            model,
            citizen_science_vec,
            10,
            deterministic=True
        )
        
        logger.info(f'Evaluation complete: {args["feature_extractor"]} {args["lstm"]} {args["algo"]} n_episodes=10')
        
        info_container = citizen_science_vec.get_attr('episode_bins')
        info_container = [i for sublist in info_container for i in sublist]
        
        return pd.DataFrame(info_container)
            
      
 
def main(args):
    
    global client
    client = boto3.client('s3')


    logger.info('Starting offlline evaluation of RL model')
    
    read_path, write_path, n_files, window, data_part, model, feat_ext, lstm = (
        args.read_path,
        args.write_path,
        args.n_files,
        args.window,
        args.data_part,
        args.model,
        args.feature_extractor,
        args.lstm
    )
    
    df = get_dataset(read_path, n_files, window, data_part)
    
    
    policy_meta = next(r for r  in POLICY_LIST if r['algo'] == model and r['feature_extractor'] == feat_ext and r['lstm'] == lstm)
    
    if policy_meta is None:
        raise ValueError(f'No policy found for {model}, {feat_ext}, {lstm}')

    logger.info(f'Running evaluation for {policy_meta}')
    
    evaluation_results = run_experiment(policy_meta, df)
    logger.info(f'Summary of evaluation results: {evaluation_results.shape}')
    write_path = os.path.join(write_path, f'window_{window}_{data_part}_{model}_{feat_ext}_{lstm}.parquet')

    logger.info(f'Writing evaluation results to {write_path}')
    
    evaluation_results.to_parquet(write_path)




if __name__ == '__main__':
    args = parse_args()
    main(args)