import logging

import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=400)
# import torch
# import torch.nn as nn
# LABEL_INDEX = 0
# from io import BytesIO
# SAMPLE_PATH = 'datasets/torch_ready_data/files_used_45/sequence_index_10'
# N_FEATURES = 12
# N_SEQUENCES = 31
# ORDINAL_INDEX = 8
# import boto3
# def get_dataset_from_s3(n_files, dset_type, n_sequences, s3_client, logger):

#     key = f'torch_ready_data/files_used_{n_files}/sequence_index_{n_sequences}/{dset_type}.npz'
#     logger.info(f'loading dataset from s3: {key}')
#     print(f'loading dataset from s3: {key}')
#     data = s3_client.get_object(
#         Bucket='dissertation-data-dmiller',
#         Key=key
#     )['Body'].read()

#     dataset = np.load(BytesIO(data))
#     return dataset['arr_0']
# import pdb
# def main():
#     logger = logging.getLogger(__name__)
#     client = boto3.client('s3')

#     np.set_printoptions(precision=5, suppress=True, linewidth=400)

#     dset_range = ['125k', '5m', '10m', '20m', 'full']

#     for d in dset_range:
#         dset = get_dataset_from_s3(45, d, 20, client, logger)
#         print(f'loaded {d} dataset with shape: {dset.shape}')
#         features = dset[:, 1:]
#         print(f'Features shape: {features.shape}')
#         user_id = features[:, 9]
#         print(f'User id shape: {user_id.shape}: unique_shape {np.unique(user_id).shape}')
#         print(user_id.max() + 1)
# if __name__ == '__main__':
#     main()

if __name == '__main__':
    main()
