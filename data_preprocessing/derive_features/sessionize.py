import torch
import boto3
import os
import numpy 
import logging
if torch.cuda.is_available():
    import GPUtil

class SessionizeData:
    logger = logging.getLogger(__name__)
    def __init__(self, df, max_sequence_index, write_path, metadata_cols, feature_cols, grouper, save_s3=True):
        self.df = df
        self.max_sequence_index = max_sequence_index + 1
        self.min_sequence_index = self.max_sequence_index - 10
        self.device = self._device()
        self.sequences = numpy.arange(self.min_sequence_index, self.max_sequence_index).tolist()
        self.seq_container = []
        self.torch_sequences = None
        self.output_path = write_path
        self.save_s3 = save_s3
        self.metadata_cols = metadata_cols
        self.feature_cols = feature_cols
        self.grouper = grouper
        

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _sequence_lazy(self):
         return next(self._lazy_load_shifted_index())

    def _shifters(self):
        for _ in range(self.min_sequence_index, self.max_sequence_index):
            self.logger.info(f'Loading sequence: {_} -> {self.max_sequence_index}')
            self.seq_container.append(self._sequence_lazy())
        if torch.cuda.is_available():
            GPUtil.showUtilization()

        sequences = torch.cat(self.seq_container, dim=1).half()
        return sequences

    def generate_sequence(self):

        self.logger.info(f'Generating shifted clickstreams from {self.min_sequence_index} -> {self.max_sequence_index}')
        sequence = self._shifters()

        self.logger.info(f'Shifter shape: {sequence.shape}')
        

        self.logger.info(f'Loading intial clickstream to {self.device}')

        if self.max_sequence_index == 11:
            self.logger.info('Initial clickstream writing to disk')
            initial_clickstream = self.df[self.metadata_cols + self.feature_cols].values.astype(numpy.float32)
            self._sequence_to_disk(initial_clickstream, 0)

        self.logger.info(f'Writing sequence to disk: {self.max_sequence_index - 1}')
        self._sequence_to_disk(sequence.cpu().numpy(), self.max_sequence_index - 1)


    def _sequence_to_disk(self, partition, sequence_index):
        if self.save_s3:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            )

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        partition_path = os.path.join(self.output_path, f'sequence_index_{sequence_index}.npz')
        self.logger.info(f'Saving to disk: {partition_path}')
        numpy.savez_compressed(partition_path, partition)

        if self.save_s3:
            self.logger.info(f'Uploading to s3: dissertation-data-dmiller/{partition_path}')
            s3_client.upload_file(partition_path, 'dissertation-data-dmiller', partition_path)

    def _lazy_load_shifted_index(self):

        torch.cuda.empty_cache()
        indx = self.sequences.pop(0)
        torch_container = []
        for col in self.feature_cols:
            sequence = self.df.groupby(self.grouper)[col].shift(indx).fillna(0).values.astype(numpy.float16)
            sequence_tensor = torch.tensor(sequence).to(self.device).half()
            torch_container.append(sequence_tensor.unsqueeze(1))
            torch.cuda.empty_cache()

        yield torch.cat(torch_container, dim=1).half()