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
from constant import TORCH_LOAD_COLS, LABEL, METADATA, DATE_TIME

CHECK_COLS = LABEL + METADATA + DATE_TIME + ['prediction']

torch.set_printoptions(sci_mode=False, linewidth=400, precision=2)
np.set_printoptions(suppress=True, precision=4, linewidth=200)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

if torch.cuda.is_available():
    import cudf as pd
    import pandas as cpu_pd
    cpu_pd.set_option('display.max_columns', 500)
    cpu_pd.set_option('display.width', 1000)
    
    import cupy as np
else:
    import pandas as pd
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    import numpy as np
    
    

CHECKPOINT_DIR='s3://dissertation-data-dmiller/lstm_experiments/checkpoints/data_v1/n_files_30/ordinal/sequence_length_10/data_partition_None/2023_03_30_07_54'
METADATA_INDEX = 13
logger = logging.getLogger('likelihood_engagement')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--file_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--write_path', type=str, default='datasets/lstm_predictions')
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
    
    write_path = os.path.join(args.write_path, f'files_used_{args.n_files}/{args.model_type}_seq_{args.n_sequences}')
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
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    

    p_bar = tqdm.tqdm(loader, total=len(loader))
    
    for indx, data in enumerate(p_bar):
        p_bar.set_description(f'Processing batch: {indx}')
        metadata, features = _extract_features(data, args.n_sequences + 1, 18)
        user_metadata = metadata[:, :4]
        preds = model(features)
        preds = nn.Sigmoid()(preds)
        user_metadata = torch.cat([user_metadata, preds], dim=1)
        user_metadata_container.append(user_metadata.cpu().numpy())

   
    user_metadata = np.concatenate(user_metadata_container, axis=0)
    user_metadata = pd.DataFrame(user_metadata, columns=['user_label', 'user_id', 'session_id', 'event_id', 'prediction'])
    
    
   
    logger.info(f'Writing predictions to {write_path}/predictions.parquet')
 
    user_metadata.to_parquet(f'{write_path}/predictions.parquet')


def join_pred_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='ordinal')
    parser.add_argument('--rl_data', type=str, default='datasets/rl_ready_data')
    
    args = parser.parse_args()
    return args

def join_predictions_on_original():
    args = join_pred_args()
    predictions_path = f'lstm_predictions/files_used_{args.n_files}/{args.model_type}_seq_{args.n_sequences}/predictions.parquet'
    dataset_path = f'calculated_features/files_used_{args.n_files}.parquet'
    if not torch.cuda.is_available():
        predictions_path, dataset_path = (
            os.path.join('datasets', predictions_path),
            os.path.join('datasets', dataset_path)
        )
    logger.info(f'Loading predictions from {predictions_path}')
    logger.info(f'Loading dataset from {dataset_path}')
    
    predictions, original = (
        pd.read_parquet(predictions_path),
        pd.read_parquet(dataset_path, columns=TORCH_LOAD_COLS)
    )
    
    predictions = predictions.rename(columns={
        'session_id': 'session_30_raw',
        'event_id': 'cum_session_event_raw'
    })
    
    logger.info(f'Shape of predictions: {predictions.shape}')
    logger.info(f'Shape of original: {original.shape}')
    
    
    logger.info(f'Joining predictions on original dataset')

    predictions = predictions.set_index(['user_id', 'session_30_raw', 'cum_session_event_raw']) \
        .join(original.set_index(['user_id', 'session_30_raw', 'cum_session_event_raw'])) \
        .reset_index() \
        .drop(columns=['user_label'])
    
    
    logger.info(f'Predictions joined: {predictions.shape}: columns')
    logger.info(pformat(predictions.columns.tolist()))
    
    logger.info(predictions[CHECK_COLS].head(10))
    
    write_path = os.path.join(
        args.rl_data,
        f'files_used_{args.n_files}',
        f'{args.model_type}_seq_{args.n_sequences}'
    )
    
    if not os.path.exists(write_path):
        logger.info(f'Creating directory: {write_path}')
        os.makedirs(write_path)
    
    logger.info(f'Writing joined predictions to {write_path}/rl_ready_data.parquet.gzip')
    predictions.to_parquet(f'{write_path}/rl_ready_data.parquet.gzip', compression='gzip')
   
    
if __name__ == "__main__":
    join_predictions_on_original()