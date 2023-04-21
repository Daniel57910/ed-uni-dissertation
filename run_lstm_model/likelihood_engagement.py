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
from torch import nn
import io
import tqdm
import numpy as np

torch.set_printoptions(sci_mode=False, linewidth=200)
np.set_printoptions(suppress=True, precision=4, linewidth=200)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

if torch.cuda.is_available():
    import cudf as pd
    import cupy as np
else:
    import pandas as pd
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
def main():
    
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
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    

    p_bar = tqdm.tqdm(loader, total=len(loader))
    
    for indx, data in enumerate(p_bar):
        p_bar.set_description(f'Processing batch: {indx}')
        metadata, features = _extract_features(data, args.n_sequences + 1, 18)
        user_metadata = metadata[:, :4]
        preds = model(features)
        preds = nn.Sigmoid()(preds)
        user_metadata = torch.cat([user_metadata, preds], dim=1)
        user_metadata_container.append(user_metadata.cpu().numpy())
        break

   
    user_metadata_container = np.concatenate(user_metadata_container, axis=0)
    user_metadata_container = pd.DataFrame(user_metadata_container, columns=['user_label', 'user_id', 'session_id', 'event_id', 'prediction'])
    
    print(user_metadata_container.head(100))
    
if __name__ == "__main__":
    main()