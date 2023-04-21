import torch
import os
import argparse
from lightning import Trainer
from npz_extractor import NPZExtractor
from model_protos import LSTMOrdinal
import logging
import boto3
import io
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

if torch.cuda.is_available():
    import cudf as pd
    import cupy as np
else:
    import pandas as pd
    import numpy as np
    
CHECKPOINT_DIR='s3://dissertation-data-dmiller/lstm_experiments/checkpoints/data_v1/n_files_30/ordinal/sequence_length_10/data_partition_None/2023_03_30_07_54'

logger = logging.getLogger('likelihood_engagement')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--file_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    return args


def main():
    
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
    
    model = torch.load(buffer, map_location=torch.device('cpu'))
    
    for m in model['state_dict']:
        print(m)
  
if __name__ == "__main__":
    main()