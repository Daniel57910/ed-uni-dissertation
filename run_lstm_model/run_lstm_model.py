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
    LSTMEmbedUserProject,
    LSTMEmbedUser,
    LSTMEmbedOneLSTM
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


np.set_printoptions(precision=8, suppress=True, linewidth=400)
torch.set_printoptions(precision=8, linewidth=400, sci_mode=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
"""
Embedding dim based on cube root of number of unique values
"""

EMBEDDING_DIMS = {
    '5': {
        'user_id': (17891, 2),
        'project_id': (328, 1),
    },
    '30': {
        'user_id': (60459 ,int(60459**0.25)),
        'project_id': (617. , int(617**0.25)),
    },
    '45': {
        'user_id': (85663, int(85663**0.25)),
        'project_id': (757, int(757**0.25)),
    },
    '61': {
        'user_id': (104744, 18),
        'project_id': (846, 6),
    }
}


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
    parser.add_argument('--n_features', type=int, default=18)

    parser.add_argument('--data_input_path', type=str, default='datasets/torch_ready_data_5')
    parser.add_argument('--data_partition', type=int, default=1000)

    parser.add_argument('--n_files', type=str, default='1')

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
    n_files
):
    if model_type == 'ordinal':
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
    elif model_type == 'embed_user_project':
        return LSTMEmbedUserProject(
            n_features,
            n_sequences,
            EMBEDDING_DIMS[n_files],
            hidden_size,
            dropout,
            learning_rate,
            batch_size,
        )
    elif model_type == 'embed_user':
        return LSTMEmbedUser(
            n_features,
            n_sequences,
            EMBEDDING_DIMS[n_files],
            hidden_size,
            dropout,
            learning_rate,
            batch_size,
        )
    else:
        return LSTMEmbedOneLSTM(
            n_features,
            n_sequences,
            EMBEDDING_DIMS[n_files],
            hidden_size,
            dropout,
            learning_rate,
            batch_size,
        )



def main(args):
    np.set_printoptions(precision=8, suppress=True, linewidth=400)
    torch.set_printoptions(precision=8, linewidth=400, sci_mode=False)
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")

    logger = setup_logging()
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logger.info(f'Running experiment at {date_time}')

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    )
    npz_extractor = NPZExtractor(
        args.data_input_path,
        args.n_files,
        args.n_sequences,
        s3_client,
        args.data_partition)

    clickstream_data_loader = ClickstreamDataModule(npz_extractor.get_dataset_pointer(), args.batch_size, args.n_sequences + 1, args.n_features)
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
        args.n_files
    )
    checkpoint_path = os.path.join(
        S3_BUCKET,
        'lstm_experiments'
        'checkpoints',
        'data_v5',
        str(args.n_files),
        args.model_type,
        f'sequence_length_{args.n_sequences}',
        str(args.data_partition),
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
        save_dir=f's3://dissertation-data-dmiller/lstm_experiments/results/data_v5/{args.n_files}/{args.model_type}',
        name=f'sequence_length_{args.n_sequences}/{args.data_partition}/{date_time}',
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
        f'device: {model.runtime_device}',
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
    logger.info(f'log_path= tensorboard --logdir {metric_logger.save_dir}/{metric_logger.name}/version_0')
    trainer = Trainer(
        precision='bf16' if torch.cuda.is_available() else 16,
        check_val_every_n_epoch=1,
        accelerator=accelerator,
        strategy=None,
        devices=device_count,
        max_epochs=args.n_epochs,
        callbacks=callbacks,
        logger=metric_logger,
        enable_progress_bar=args.progress_bar,
        log_every_n_steps=10
        )
    
    if args.checkpoint:
        checkpoint_s3_path = os.path.join('s3://dissertation-data-dmiller', args.checkpoint)
        logger.info(f'Running model from checkpoint: {checkpoint_s3_path}')
        trainer.fit(model, ckpt_path=checkpoint_s3_path, datamodule=clickstream_data_loader)
    else:
        logger.info('Running model from scratch')
        trainer.fit(model, datamodule=clickstream_data_loader) 
            

        # checkpoint_s3_path = os.path.join('s3://dissertation-data-dmiller', args.checkpoint)
        # # logger.info(f'Downloading checkpoint from {checkpoint_s3_path}')
        # # model.load_from_checkpoint(checkpoint_s3_path)
        # # auc_user_val, auc_user_test = (
        # #     auc_by_user_bin(model, clickstream_data_loader.val_dataloader(), args.n_sequences, "val"),
        # #     auc_by_user_bin(model, clickstream_data_loader.test_dataloader(), args.n_sequences, "test")
        # # )

        # # df_val = pd.DataFrame(auc_user_val, columns=['user_bin', 'acc', 'prec', 'rec', 'f1'])
        # # df_test = pd.DataFrame(auc_user_test, columns=['user_bin', 'acc', 'prec', 'rec', 'f1'])

        # # df_val.to_csv(f'auc_user_val_seq_{args.n_sequences}_heuristic_{args.zero_heuristic}.csv') 
        # # df_test.to_csv(f'auc_user_test_seq_{args.n_sequences}_heuristic_{args.zero_heuristic}.csv')

if __name__ == '__main__':
    args = parse_args()
    main(args)