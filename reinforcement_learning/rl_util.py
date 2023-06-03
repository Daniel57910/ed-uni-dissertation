import os
import logging
global logger

logger = logging.getLogger(__name__)
import numpy as np
from pqdm.processes import pqdm
import pdb

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
    

def _parralel_partition_users(unique_sessions, df, index, vec_df_path):
    subset_session = df.merge(unique_sessions, on=['user_id', 'session_30_count_raw'], how='inner')
    subset_session.to_parquet(f'{vec_df_path}/batch_{index}.parquet')
    return subset_session.copy().reset_index(drop=True)
    

def batch_environments_for_vectorization(df, n_envs, vec_df_path):
   
    df[['user_id', 'session_30_count_raw']] = df[['user_id', 'session_30_count_raw']].astype(int)
   
    unique_sessions = df[['user_id', 'session_30_count_raw']].drop_duplicates().sample(frac=1).reset_index(drop=True)
    logger.info(f'Unique sessions shape: {unique_sessions.shape}. Splitting into {n_envs} environments')
    unique_session_split = np.array_split(unique_sessions, n_envs)
    
    unique_session_args = [{
        'unique_sessions': sess,
        'df': df,
        'index': i,
        'vec_df_path': vec_df_path,
    } for i, sess in enumerate(unique_session_split)]

    logger.info(f'Environments split: running parralel partitioning')
    result = pqdm(unique_session_args, _parralel_partition_users, n_jobs=os.cpu_count(), argument_type='kwargs')
    logger.info(f'Environments split: finished parralel partitioning')
    return result
        
    