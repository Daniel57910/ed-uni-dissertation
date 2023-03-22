import torch
import boto3
import os
if torch.cuda.is_available():
    import GPUtil

class SessionizeData:
    def __init__(self, df, max_sequence_index, write_path, partition_list=PARTITION_LIST, save_s3=True):
        self.df = df
        self.max_sequence_index = max_sequence_index + 1
        self.min_sequence_index = self.max_sequence_index - 10
        self.device = self._device()
        self.sequences = numpy.arange(self.min_sequence_index, self.max_sequence_index).tolist()
        self.seq_container = []
        self.torch_sequences = None
        self.output_path = write_path
        self.partition_list = partition_list
        self.save_s3 = save_s3

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _sequence_lazy(self):
         return next(self._lazy_load_shifted_index())

    def _shifters(self):
        for _ in range(self.min_sequence_index, self.max_sequence_index):
            print(f'Loading sequence: {_} -> {self.max_sequence_index}')
            self.seq_container.append(self._sequence_lazy())
        if torch.cuda.is_available():
            GPUtil.showUtilization()

        sequences = torch.cat(self.seq_container, dim=1).half()
        return sequences

    def generate_sequence(self):

        print(f'Generating shifted clickstreams from {self.min_sequence_index} -> {self.max_sequence_index}')
        sequence = self._shifters()

        print(f'Shifters shape: {sequence.shape}')

        cols_required =  ['label', 'total_events'] + ENCODED_COLS + SCALED_COLS + GENERATED_COLS
        print(f'Columns required: {cols_required}')
        print(f'Loading intial clickstream to {self.device}')

        if self.max_sequence_index == 11:
            print('Initial clickstream writing to disk')
            initial_clickstream = self.df[cols_required].values.astype(np.float32)
            self._sequence_to_disk(initial_clickstream, 0)

        print(f'Writing sequence to disk: {self.max_sequence_index - 1}')
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
        print(f'Saving to disk: {partition_path}')
        np.savez_compressed(partition_path, partition)

        if self.save_s3:
            print(f'Uploading to s3: dissertation-data-dmiller/{partition_path}')
            s3_client.upload_file(partition_path, 'dissertation-data-dmiller', partition_path)

    def _lazy_load_shifted_index(self):

        torch.cuda.empty_cache()
        indx = self.sequences.pop(0)
        torch_container = []
        for col in SCALED_COLS + GENERATED_COLS:
            sequence = self.df.groupby(GROUPBY_COLS)[col].shift(indx).fillna(0).values.astype(np.float16)
            sequence_tensor = torch.tensor(sequence).to(self.device).half()
            torch_container.append(sequence_tensor.unsqueeze(1))
            torch.cuda.empty_cache()

        yield torch.cat(torch_container, dim=1).half()