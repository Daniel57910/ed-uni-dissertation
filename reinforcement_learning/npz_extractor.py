import logging
import os
import zipfile

import boto3
import numpy as np
import torch
import logging 

class NPZExtractor:
    logger = logging.getLogger(__name__)
    def __init__(self, input_path, n_files, n_sequences, s3_client, data_partition) -> None:
        self.input_path = input_path
        self.n_files = n_files
        self.n_sequences = n_sequences
        self.s3_client = s3_client
        self.data_partition = data_partition


    def get_dataset_pointer(self):

        read_path = os.path.join(self.input_path, f'files_used_{self.n_files}')
        if not os.path.exists(read_path):
            print(f'Creating directory: {read_path}')
            os.makedirs(read_path)


        for _ in range(0, self.n_sequences +1, 10):
            key_zip, key_npy = (
                os.path.join(read_path, f'sequence_index_{_}.npz'),
                os.path.join(read_path, f'sequence_index_{_}')
            )

            self.logger.info(f'Loading pointer to dataset: {key_npy}: derived from {key_zip}')

            if not os.path.exists(key_npy):
                self.logger.info(f'Zip file to extract: {key_zip}: npy file to load: {key_npy}')
                # self.s3_client.download_file(
                #     'dissertation-data-dmiller',
                #     key_zip,
                #     key_zip
                # )
                self.logger.info(f'Zip file downloaded: {key_zip}')
                self._zip_extract(key_zip, key_npy)

        lz_concatenated_results = self._lazy_concatenate()

        if self.data_partition:
            return [p[:self.data_partition] for p in lz_concatenated_results]
        else:
            return lz_concatenated_results


    def _zip_extract(self, key_zip, key_npy):
        self.logger.info(f'Extracting file: {key_zip} -> {key_npy}')

        with zipfile.ZipFile(key_zip, 'r') as zip_ref:
            zip_ref.extractall(path=key_npy, members=['arr_0.npy'])


        self.logger.info(f'Zip file exracted: {key_zip} -> {key_npy}/arr_0.npy')

    def _lazy_concatenate(self):
        lz_concat = []
        for _ in range(0, self.n_sequences +1, 10):
            path_to_load = os.path.join(self.input_path, f'files_used_{self.n_files}', f'sequence_index_{_}', f'arr_0.npy')
            self.logger.info(f'Loading: {path_to_load}')
            lz_concat.append(np.load(path_to_load))
        return lz_concat