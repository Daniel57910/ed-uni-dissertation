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
from environment import CitizenScienceEnv
# from environment_eval import CitizenScienceEnv
from policies.cnn_policy import CustomConv1dFeatures
from rl_constant import (
    FEATURE_COLS, METADATA_COLS, PREDICTION_COLS, RL_STAT_COLS, PREDICTION_COLS, LOAD_COLS
)
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
import tqdm
from joblib import Parallel, delayed
import json
from pqdm.processes import pqdm
from policy_list import POLICY_LIST


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
import zipfile


S3_BASELINE_PATH = 'dissertation-data-dmiller'
N_SEQUENCES = 40
CHECKPOINT_FREQ = 100_000
TB_LOG = 10_000
WINDOW = 2
import glob

global logger

logger = logging.getLogger('rl_results_eval')
logger.setLevel(logging.INFO)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--write_path', type=str, default='datasets/rl_results')
    parse.add_argument('--n_files', type=int, default=2)
    parse.add_argument('--part', type=str, default='train')
    parse.add_argument('--model', type=str, default='DQN')
    parse.add_argument('--feature_extractor', type=str, default='CNN')
    parse.add_argument('--lstm', type=str, default='label')
    args = parse.parse_args()
    return args


def find_s3_candidate(client, feature_extractor, lstm, run_time):
    
    if lstm == 'seq_20':
        lstm = 'seq_40'
    folder_prefix = os.path.join(
        'reinforcement_learning_incentives_3',
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
        os.path.join('reinforcement_learning_incentives_3', f'{feature_extractor}_{lstm}'),
        os.path.join('reinforcement_learning_incentives_3', f'{feature_extractor}_{lstm}', f'{algo}.zip') 
    )
    
    if not os.path.exists(model_base_path):
        logger.info(f'Creating directory {model_base_path}')
        os.makedirs(model_base_path)
    client.download_file(S3_BASELINE_PATH, s3_candidate, download_path)
    logger.info(f'Loading model from {s3_candidate} to {download_path}')

    logger.info(f'Checkpoint load path: {download_path}')
    return download_path
        


def run_exp_wrapper(args, vectorized_df):
    
        logger.info(f'Getting policy for {args["feature_extractor"].lower()}, {args["lstm"]}, {args["run_time"]}, {args["algo"]}')
        policy_weights = get_policy(client, args['feature_extractor'].lower(), args['lstm'], args['run_time'], args['algo'])
       
        
        logger.info(f'Policy weights loaded')
        
        out_features = FEATURE_COLS + ([args['lstm']] if args['lstm'] else [])

        citizen_science_vec = DummyVecEnv(
            [lambda: CitizenScienceEnv(df, out_features, N_SEQUENCES) for df in vectorized_df]
        )
        
    
        logger.info(f'Vectorized dataset, setting policy weights')
        if args['feature_extractor'].lower() == 'cnn':
            CustomConv1dFeatures.setup_sequences_features(N_SEQUENCES + 1, len(out_features))
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
            
      
def download_dataset_from_s3(client, base_read_path, full_read_path):
    logger.info(f'Downloading data from {base_read_path}')
    os.makedirs(base_read_path, exist_ok=True)
    
    logger.info(f'Downloading data from dissertation-data-dmiller/{full_read_path}')
    client.download_file(
        'dissertation-data-dmiller',
        full_read_path,
        full_read_path
    )
    logger.info(f'Downloaded data from dissertation-data-dmiller/{full_read_path}')
    
def main(args):
    
    global client
    client = boto3.client('s3')


    logger.info('Starting offlline evaluation of RL model')
    
    write_path, n_files, part, model, feat_ext, lstm = (
        args.write_path,
        args.n_files,
        args.part,
        args.model,
        args.feature_extractor,
        args.lstm,
    )
   
    base_read_path = os.path.join('rl_ready_data_conv', f'files_used_{n_files}')
    vec_df_path = os.path.join(base_read_path, f'citizen_science_{part}_batched')
    vec_df_files = glob.glob(os.path.join(vec_df_path, '*.parquet'))
    
    vectorized_df = [
        pd.read_parquet(file)
        for file in vec_df_files
    ]

    policy_meta = next(r for r  in POLICY_LIST if r['algo'] == model and r['feature_extractor'] == feat_ext and r['lstm'] == lstm)
    if policy_meta is None:
        raise ValueError(f'No policy found for {model}, {feat_ext}, {lstm}')

    logger.info(f'Running evaluation for {policy_meta}')
    
    evaluation_results = run_exp_wrapper(policy_meta, vectorized_df)

    logger.info(f'Summary of evaluation results: {evaluation_results.shape}')
    write_path = os.path.join(write_path, f'window_{WINDOW}_{part}')
   
    if not os.path.exists(write_path):
        logger.info(f'Creating directory {write_path}')
        os.makedirs(write_path)
        
    full_write_path = os.path.join(write_path, f'{model}_{feat_ext}_{lstm}.parquet').lower()

    logger.info(f'Writing evaluation results to {full_write_path}')
    evaluation_results.to_parquet(full_write_path)

    print(evaluation_results.head())


if __name__ == '__main__':
    args = parse_args()
    main(args)