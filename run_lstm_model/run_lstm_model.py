#  %load run_lstm_model.py
import argparse
import logging
import os
import sys
from datetime import datetime
from io import BytesIO

import boto3
import numpy as np
import torch
from data_module import ClickstreamDataModule
from model_protos import (
    LSTMOrdinal,
    LSTMEmbedUser,
)
from model_protos import LSTMOrdinal
from npz_extractor import NPZExtractor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import json
import pdb
S3_BUCKET = 's3://dissertation-data-dmiller'
SNS_TOPIC = 'arn:aws:sns:eu-west-1:774141665752:gradient-task'

USER_INDEX = 9
PROJECT_INDEX = 10
COUNTRY_INDEX = 11
METADATA_INDEX = 14


np.set_printoptions(precision=8, suppress=True, linewidth=400)
torch.set_printoptions(precision=8, linewidth=400, sci_mode=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
"""
Embedding dim based on cube root of number of unique values
"""


def setup_logging():

    logger = logging.getLogger(__name__)

    handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(f'Setup logging')
    return logger


def _device_count():
    if 'ipykernel' in sys.modules: return 1

    if torch.cuda.is_available():
        return torch.cuda.device_count()

    return 1


def setup_logging():

    logger = logging.getLogger(__name__)
    return logger


def _device_count():
    if 'ipykernel' in sys.modules: return 1

    if torch.cuda.is_available():
        return torch.cuda.device_count()

    return 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ordinal')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=1)

    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_sequences', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=20)

    parser.add_argument('--data_input_path', type=str, default='datasets/torch_ready_data')
    parser.add_argument('--data_partition', type=int, default=1000)

    parser.add_argument('--n_files', type=str, default='2')

    parser.add_argument('--progress_bar', type=bool, default=True)
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--find_hparams', type=bool, default=False)

    parser.add_argument('--zero_heuristic', type=bool, default=False)
    parser.add_argument('--validate_only', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_model(
    logger,
    model_type,
    n_features,
    n_sequences,
    hidden_size,
    dropout,
    learning_rate,
    batch_size,
    zero_heuristic,
    embedding_params=None
):
    if model_type.startswith('ordinal') or model_type.startswith('heuristic'):
        logger.info('Creating LSTMOrdinal model')
        return LSTMOrdinal(
            n_features,
            n_sequences,
            hidden_size,
            dropout,
            learning_rate,
            batch_size,
            zero_heuristic
        )
    
    return LSTMEmbedUser(
        n_features,
        n_sequences,
        embedding_params,
        hidden_size,
        dropout,
        learning_rate,
        batch_size,
        zero_heuristic
    )


def main(args):
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")

    logger = setup_logging()
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logger.info(f'Running experiment at {date_time}')

    s3_client = boto3.client(
        's3'
    )
    
    npz_extractor = NPZExtractor(
        args.data_input_path,
        args.n_files,
        args.n_sequences,
        s3_client,
        args.data_partition)
    
    clickstream_data_loader = ClickstreamDataModule(npz_extractor.get_dataset_pointer(), args.batch_size, args.n_sequences + 1)
    
    if args.model_type.startswith('embedded'):
        users = npz_extractor.get_dataset_pointer()[0][:, 1].max() + 1
        embed_params = { 'user_embed': int(users), 'embed_out': int(max(users ** 0.25, 3)) }

    else:
        embed_params = None
    
    logger.info(f'Running model with embedding params {embed_params}')
    model = get_model(
        logger,
        args.model_type,
        args.n_features,
        args.n_sequences + 1,
        args.hidden_size,
        args.dropout,
        args.learning_rate,
        args.batch_size,
        args.zero_heuristic,
        embed_params
    )
    
    data_version = "1"
    checkpoint_path = os.path.join(
        S3_BUCKET,
        'lstm_experiments',
        'checkpoints',
        f'data_v{data_version}',
        f'n_files_{str(args.n_files)}',
        args.model_type,
        f'sequence_length_{args.n_sequences}',
        f'data_partition_{str(args.data_partition)}',
        date_time)

    checkpoint = ModelCheckpoint(
        monitor='loss_valid',
        dirpath=checkpoint_path,
        filename='clickstream-{epoch:02d}-{loss_valid:.2f}',
        every_n_epochs=2,
        save_top_k=3
    )

    callbacks = [checkpoint]
    if args.progress_bar:
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks += [progress_bar]

    metric_logger = TensorBoardLogger(
        save_dir=f's3://dissertation-data-dmiller/lstm_experiments/results/data_v{data_version}/n_files_{args.n_files}/{args.model_type}',
        name=f'sequence_length_{args.n_sequences}/data_partition_{args.data_partition}/{date_time}',
        flush_secs=60,
        log_graph=True,
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    device_count = _device_count()
    strategy = 'ddp' if 'ipykernel' not in sys.modules else None

    config = "\n".join([
        f'data input path: {args.data_input_path}',
        f'data partition: {args.data_partition}',
        f'batch_size: {args.batch_size}',
        f'n_epoch: {args.n_epochs}',
        f'n_workers: 8',
        f'device: {accelerator}',
        f'train_samples: {clickstream_data_loader.training_data[0].shape[0]}',
        f'val_samples: {clickstream_data_loader.validation_data[0].shape[0]}',
        f'hidden size: {args.hidden_size}',
        f'dropout: {args.dropout}',
        f'n_sequences: {args.n_sequences}',
        f'n_features: {args.n_features}',
        f'learning_rate: {args.learning_rate}',
        f'accelerator: {accelerator}',
        f'device_count: {device_count}',
        f'strategy: {strategy}',
        f'model_type: {args.model_type}',
        f'zero_heuristic: {args.zero_heuristic}',
    ])


    logger.info(f'Beginning validation:\n {config}')
    logger.info(f'log_path=\n tensorboard --logdir {metric_logger.save_dir}/{metric_logger.name}/version_0')
    logger.info(f'checkpoint_path=\n {checkpoint_path}')
    trainer = Trainer(
        precision=16,
        check_val_every_n_epoch=1,
        accelerator=accelerator,
        devices=device_count,
        max_epochs=args.n_epochs,
        callbacks=callbacks,
        logger=metric_logger,
        enable_progress_bar=args.progress_bar,
        log_every_n_steps=500
        )
    
    if args.checkpoint:
        checkpoint_s3_path = os.path.join('s3://dissertation-data-dmiller', args.checkpoint)
        logger.info(f'Running model from checkpoint: {checkpoint_s3_path}')
        trainer.fit(model, ckpt_path=checkpoint_s3_path, datamodule=clickstream_data_loader)
    else:
        logger.info('Running model from scratch')
        trainer.fit(model, datamodule=clickstream_data_loader) 

if __name__ == "__main__":
    args = parse_args()
    main(args)