import torch
import os
import argparse
from npz_extractor import NPZExtractor
from model_protos import LSTMOrdinal
from data_module import ClickstreamDataset
from torch.utils.data import DataLoader, Dataset
from torch_model_bases import LSTMOrdinal
import logging
import boto3
from pprint import pformat
from torch import nn
import io
import tqdm
import numpy as np
import pandas as pd
from constant import (
    LABEL,
    METADATA,
    OUT_FEATURE_COLUMNS,
    LOAD_COLS,
    S3_BUCKET,
    BASE_CHECK_PATH,
    LSTM_CHECKPOINTS
)

if torch.cuda.is_available():
    import cudf as gpu_pd
    import pandas as pd
    import cupy as cp


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)    
torch.set_printoptions(sci_mode=False, linewidth=400, precision=2)
np.set_printoptions(suppress=True, precision=4, linewidth=200)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

CHECKPOINT_DIR='s3://dissertation-data-dmiller/lstm_experiments/checkpoints/data_v1/n_files_30/ordinal/sequence_length_10/data_partition_None/2023_03_30_07_54'
METADATA_INDEX = 14
logger = logging.getLogger('likelihood_engagement')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--n_sequences', type=int, default=20)
    parser.add_argument('--file_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--write_path', type=str, default='datasets/rl_ready_data')
    parser.add_argument('--model_type', type=str, default='ordinal')
    args = parser.parse_args()
    return args

def _extract_features(tensor, n_sequences, n_features):
    
    features_dict = {}
        
    metadata, features = tensor[:, :METADATA_INDEX], tensor[:, METADATA_INDEX:] 
                
    features = torch.flip(
        torch.reshape(features, (features.shape[0], n_sequences, n_features)),
        dims=[1]
    )
    
    features_dict['features_20'] = features
    features_dict['features_10'] = features[:, 10:, :]
    features_dict['last_sequence'] = features[:, -1, :]
     
    return metadata, features_dict



def get_models(checkpoints: dict, s3_client, device):
    """_summary_
    Downloads models from s3 and loads them into memory.
    """
    models = {}
    for name, checkpoint in checkpoints.items():
        logger.info(f'Downloading model: {name}')
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=checkpoint
        )
        buffer = io.BytesIO(response['Body'].read())
        state = torch.load(buffer, map_location=torch.device(device))
        model = LSTMOrdinal()
        model.load_state_dict(state['state_dict'])
        model.to(device)
        models[name] = model
    return models

@torch.no_grad()
def generate_static_predictions():
    
    user_metadata_container = []
    
    logger.info('Generating static prediction likelihoods for experiment')
    args = parse_args()
    npz_extractor = NPZExtractor(
        args.file_path,
        args.n_files,
        args.n_sequences,
        None,
        None
           
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    logger.info(f'Setting device to {device}')
    
    logger.info('generating dataset pointer')
    dataset = npz_extractor.get_dataset_pointer()
    
    logger.info('Downloading model checkpoint')
    
    write_path = os.path.join(args.write_path, f'files_used_{args.n_files}')
    if not os.path.exists(write_path):
        logger.info(f'Creating directory: {write_path}')
        os.makedirs(write_path)
    
    client = boto3.client('s3')
    
    logger.info(f'Downloading models from checkpoints {LSTM_CHECKPOINTS.keys()}')
    
    models = get_models(LSTM_CHECKPOINTS, client, device)
    
    dataset = ClickstreamDataset(dataset)
    loader = DataLoader(dataset, batch_size=2047, shuffle=False)
    activation = nn.Sigmoid()
    

    p_bar = tqdm.tqdm(loader, total=len(loader))
    
    for indx, data in enumerate(p_bar):
        p_bar.set_description(f'Processing batch: {indx}')
        data.to(device)
        metadata, features_dict = _extract_features(data, args.n_sequences + 1, 20)
        
        preds_10 = activation(models['seq_10'](features_dict['features_10']))
        preds_20 = activation(models['seq_20'](features_dict['features_20']))
        last_event = features_dict['last_sequence']
        
        user_metadata = torch.cat([metadata, last_event, preds_10, preds_20], dim=1)
        user_metadata_container.append(user_metadata.squeeze())

    
    predicted_data = torch.cat(user_metadata_container, dim=0).cpu().numpy()
    logger.info(f'Predicted data shape: {predicted_data.shape}: generating df')
    predicted_data = pd.DataFrame(predicted_data, columns=LABEL + METADATA + OUT_FEATURE_COLUMNS + ['seq_10', 'seq_20'])

    logger.info('Decoding date time and sorting')
    predicted_data['date_time'] = pd.to_datetime(predicted_data[['year', 'month', 'day', 'hour', 'minute', 'second']])
    predicted_data = predicted_data.sort_values(by=['date_time', 'user_id'])
    
    logger.info(f'Writing to parquet: {os.path.join(write_path, "predicted_data.parquet")}')
    predicted_data.to_parquet(os.path.join(write_path, 'predicted_data.parquet'))
    
    
    
    

   
    
if __name__ == "__main__":
    generate_static_predictions()