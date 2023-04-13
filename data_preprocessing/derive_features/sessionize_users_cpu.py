from sessionize import SessionizeData
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler
from constant import (
    TORCH_LOAD_COLS,
    OUT_FEATURE_COLUMNS,
    GROUPBY_COLS
)

import torch
import numpy as np
from pprint import pformat
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=4)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

def get_logger():
    logger = logging.getLogger(__name__)
    return logger


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

torch.set_printoptions(sci_mode=False, precision=4, linewidth=400)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

def get_logger():
    logger = logging.getLogger(__name__)
    return logger


def scale_feature_cols(df, scaler, scaler_columns):
    df[scaler_columns] = scaler.fit_transform(df[scaler_columns])
    return df

def main(args):
    
    logger = get_logger()
    logger.info('Starting sessionize_users_cpu.py with arguments')
    logger.info(pformat(args.__dict__))
    
    data_read = os.path.join(args.input_path, f'files_used_{args.data_subset}')

    logger.info(f'Reading data from {data_read}')
    df = pd.read_parquet(data_read, columns=TORCH_LOAD_COLS)
    logger.info(f'Data read: {df.shape}')
    logger.info('Casting date time and sorting by date time')
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by=['date_time'])
    logger.info('Data read: scaling scaler columns')
    df = scale_feature_cols(df, MinMaxScaler(), OUT_FEATURE_COLUMNS)
    logger.info('Scaling complete: implement sessionize')
    
    for seq_index in args.seq_list:

        sessionize = SessionizeData(
            df,
            seq_index,
            os.path.join(args.output_path, f'files_used_{args.data_subset}'),
            [col for col in TORCH_LOAD_COLS if col != 'date_time'],
            OUT_FEATURE_COLUMNS,
            GROUPBY_COLS,
            args.save_s3
        )
    
        logger.info(f'Generating sequence for {seq_index}')
        sessionize.generate_sequence()
    
    logger.info(f'Sessionize complete for sequences {args.seq_list}')
   
class Arguments:
    seq_list = [10]
    input_path = 'datasets/calculated_features'
    output_path = 'datasets/torch_ready_data_main'
    data_subset = 2
    save_s3 = False

if __name__ == '__main__':
    args = Arguments()
    main(args)