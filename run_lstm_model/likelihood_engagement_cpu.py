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
from rl_constant import LABEL, METADATA, DATE_COLS, OUT_FEATURE_COLUMNS, PREDICTION_COLS

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
METADATA_INDEX = 13
logger = logging.getLogger('likelihood_engagement')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--file_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--write_path', type=str, default='datasets/rl_ready_data')
    parser.add_argument('--model_type', type=str, default='ordinal')
    args = parser.parse_args()
    return args

def _extract_features(tensor, n_sequences, n_features):
        
    metadata, features = tensor[:, :METADATA_INDEX], tensor[:, METADATA_INDEX:] 
                
    features = torch.flip(
        torch.reshape(features, (features.shape[0], n_sequences, n_features)),
        dims=[1]
    )
        
    return metadata, features

def _extract_last_sequence(tensor):
    """_summary_
    Extracts the last sequence from a tensor of sequences.
    """
    return tensor[:, -1, :]

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
    
    logger.info('generating dataset pointer')
    dataset = npz_extractor.get_dataset_pointer()
    
    logger.info('Downloading model checkpoint')
    
    write_path = os.path.join(args.write_path, f'files_used_{args.n_files}')
    if not os.path.exists(write_path):
        logger.info(f'Creating directory: {write_path}')
        os.makedirs(write_path)
    
    client = boto3.client('s3')
    
    checkpoint = client.get_object(
        Bucket='dissertation-data-dmiller',
        Key='lstm_experiments/checkpoints/data_v1/n_files_30/ordinal/sequence_length_10/data_partition_None/2023_03_30_07_54/clickstream-epoch=83-loss_valid=0.29.ckpt'
    )
    

    logger.info('Loading model checkpoint')
    
    buffer = io.BytesIO(checkpoint['Body'].read())
   
    logger.info('checkpoint loaded from buffer. Loading model')
    model_state = torch.load(buffer, map_location=torch.device('cpu'))
    model = LSTMOrdinal()
    model.load_state_dict(model_state['state_dict'])
    logger.info(f'Model loaded. Creating dataset: n_events {dataset[0].shape[0]}')
    
    dataset = ClickstreamDataset(dataset)
    loader = DataLoader(dataset, batch_size=2047, shuffle=False)
    

    p_bar = tqdm.tqdm(loader, total=len(loader))
    
    for indx, data in enumerate(p_bar):
        p_bar.set_description(f'Processing batch: {indx}')
        metadata, features = _extract_features(data, args.n_sequences + 1, 18)
        last_sequence = _extract_last_sequence(features)
        preds = model(features)
        preds = nn.Sigmoid()(preds)
        user_metadata = torch.cat([metadata, preds, last_sequence], dim=1)
        user_metadata_container.append(user_metadata)

    predicted_data = torch.tensor(user_metadata).cpu().numpy()
    predicted_data = pd.DataFrame(user_metadata, columns=LABEL + METADATA + DATE_COLS + OUT_FEATURE_COLUMNS + PREDICTION_COLS)
    predicted_data = predicted_data.rename(columns={
        'prediction': f'pred_{args.model_type}_seq_{args.n_sequences}'
    })
    
    predicted_data = predicted_data.drop(columns=DATE_COLS)
    output_path = os.path.join(write_path, f'{args.model_type}_seq_{args.n_sequences}.parquet')
    logger.info(f'Writing rl ready data: {output_path}')
    
    logger.info(f'Percentage data correct: {predicted_data.count().min() / predicted_data.shape[0]}')
    predicted_data.to_parquet(output_path, index=False)
    

   
    
if __name__ == "__main__":
    generate_static_predictions()