import argparse
import glob
import os
import pdb
import pprint as pp
from datetime import datetime

import boto3
import torch
import tqdm

if torch.cuda.is_available():
    import cupy as np
    import numpy
    import cudf as pd
    import dask_cudf as dd
    from cuml.preprocessing import MinMaxScaler
    import GPUtil
else:
    import numpy as np
    import numpy
    import pandas as pd
    import dask.dataframe as dd
    from sklearn.preprocessing import MinMaxScaler

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=4)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=200)

torch.set_printoptions(linewidth=400, precision=4, sci_mode=False)

SCALED_COLS =[
    'timestamp',
    'time_diff_seconds',
    '30_minute_session_count',
    '5_minute_session_count',
    'task_within_session_count',
    'user_count',
    'project_count',
    'country_count',
]

GENERATED_COLS = [
    'cum_events',
    'cum_projects',
    'cum_time',
    'cum_time_within_session',
    'av_time_across_clicks',
    'av_time_across_clicks_session',
    'rolling_average_tasks_within_session',
    'rolling_av_time_within_session',
    'rolling_time_between_sessions',
]

ENCODED_COLS = [
    'user_id',
    'project_id',
    'country'
]


GROUPBY_COLS = ['user_id']

TIMESTAMP_INDEX = 1

INITIAL_LOAD_COLUMNS = ENCODED_COLS +  ['label', 'date_time'] +  [col for col in SCALED_COLS if 'timestamp' not in col and 'project_count' not in col]

TIMESTAMP_INDEX = 1

COUNTRY_ENCODING = {
    'Finland': 1,
    'United States': 2,
    'China': 3,
    'Singapore': 4,
}

PARTITION_LIST = [
    {
        'name': '125k',
        'size': 125000,
        'indexes': None
    },
    {
        'name': '125m',
        'size': 1250000,
        'indexes': None
    },
    {
        'name': '5m',
        'size': 5000000,
    },
    {
        'name': '10m',
        'size': 10000000,
    },
    {
        'name': '20m',
        'size': 20000000,
    },
    {
        'name': 'full',
        'size': None,
    }
]

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

def _encode_countries(x):
        if x == 'Finland':
            return 1
        elif x == 'United States':
            return 2
        elif x == 'China':
            return 3
        else:
            return 4

import pdb

def join_for_encodings(df):


    project_id_value_counts = df['project_id'].value_counts().reset_index().rename(columns={'index': 'project_id', 'project_id': 'project_count'})
    df = df.merge(project_id_value_counts, on='project_id', how='left')

    if torch.cuda.is_available():
        df = df.compute().to_pandas()
    else:
        df = df.compute()

    df['country'] = df['country'].apply(_encode_countries)

    if torch.cuda.is_available():
        df = pd.from_pandas(df)

    df['date_time'] = pd.to_datetime(df['date_time'])
    total_events = df['user_id'].value_counts().reset_index().rename(columns={'index': 'user_id', 'user_id': 'total_events'})

    df = df.merge(total_events, on='user_id', how='left')
    df['cum_events'] = df[['user_id', 'task_within_session_count']].groupby('user_id').cumcount() + 1
    df['cum_projects'] = (df.groupby('user_id')['project_id'].transform(lambda x: pd.CategoricalIndex(x).codes) + 1).astype('int32')
    df['cum_time'] = df.groupby('user_id')['time_diff_seconds'].cumsum()
    df['cum_time_within_session'] = df.groupby(['user_id', '30_minute_session_count'])['time_diff_seconds'].cumsum()
    df['av_time_across_clicks'] = df.groupby('user_id')['time_diff_seconds'].rolling(1000, min_periods=1).mean().reset_index()['time_diff_seconds']
    df['av_time_across_clicks_session'] = df.groupby(['user_id', '30_minute_session_count'])['time_diff_seconds'].rolling(10, min_periods=1).mean().reset_index()['time_diff_seconds']


    av_num_events_within_session = df.groupby(['user_id', '30_minute_session_count'])['task_within_session_count'].max().reset_index()
    av_num_events_within_session['rolling_average_tasks_within_session'] = av_num_events_within_session.groupby('user_id')['task_within_session_count'].rolling(10, min_periods=1).mean().reset_index()['task_within_session_count']

    av_time_within_session = df.groupby(['user_id', '30_minute_session_count']).apply(lambda x: x['date_time'].max() - x['date_time'].min()).reset_index().rename(columns={0: 'av_time_within_session'})
    av_time_within_session['av_time_within_session'] = av_time_within_session['av_time_within_session'].apply(lambda x: x.total_seconds())
    av_time_within_session['rolling_av_time_within_session'] = av_time_within_session.groupby('user_id')['av_time_within_session'].rolling(10, min_periods=1).mean().reset_index()['av_time_within_session']


    session_start_time = df.groupby(['user_id', '30_minute_session_count'])['date_time'].min().reset_index().rename(columns={'date_time': 'session_start_time'})
    session_end_time = df.groupby(['user_id', '30_minute_session_count'])['date_time'].max().reset_index().rename(columns={'date_time': 'session_end_time'})

    session_meta = session_start_time.merge(session_end_time, on=['user_id', '30_minute_session_count'], how='left')
    session_meta['previous_end_time'] = session_meta.groupby(['user_id'])['session_end_time'].shift()
    session_meta['previous_end_time'] = session_meta['previous_end_time'].fillna(0)
    session_meta['time_between_sessions'] = session_meta.apply(lambda x: 0 if x['previous_end_time'] == 0 else (x['session_start_time'] - x['previous_end_time']), axis=1)
    session_meta['time_between_sessions'] = session_meta['time_between_sessions'].apply(lambda x: 0 if x == 0 else x.seconds)
    session_meta['rolling_time_between_sessions'] = session_meta.groupby('user_id')['time_between_sessions'].rolling(5,  min_periods=1).mean().reset_index()['time_between_sessions']

    df = df.merge(av_num_events_within_session[['user_id', '30_minute_session_count', 'rolling_average_tasks_within_session']], on=['user_id', '30_minute_session_count'], how='left')
    df = df.merge(av_time_within_session[['user_id', '30_minute_session_count', 'rolling_av_time_within_session']], on=['user_id', '30_minute_session_count'], how='left')
    df = df.merge(session_meta[['user_id', '30_minute_session_count', 'rolling_time_between_sessions']], on=['user_id', '30_minute_session_count'], how='left')
    user_id_hash = pd.DataFrame(df['user_id'].unique()).reset_index().rename(columns={'index': 'user_id_hash', 0: 'user_id'})
    project_id_hash = pd.DataFrame(df['project_id'].unique()).reset_index().rename(columns={'index': 'project_id_hash', 0: 'project_id'})

    user_id_hash['user_id_hash'] = user_id_hash['user_id_hash'] + 1
    project_id_hash['project_id_hash'] = project_id_hash['project_id_hash'] + 1

    df = df.merge(user_id_hash, on='user_id', how='left')
    df = df.merge(project_id_hash, on='project_id', how='left')

    df = df.drop(columns=['user_id', 'project_id'])
    df = df.rename(columns={'user_id_hash': 'user_id', 'project_id_hash': 'project_id'})
    return df

def prepare_for_sessionization(data_paths: list, scaler: MinMaxScaler):

    df = dd.read_csv(data_paths, usecols=INITIAL_LOAD_COLUMNS)

    df = join_for_encodings(df)
    df['timestamp_raw'] = df['date_time'].astype('int64') // 10**9
    df = df.sort_values(by='date_time')

    df['timestamp'] = df['timestamp_raw']
    print(f'Loaded data: shape = {df.shape}, min_date, max_date: {df.date_time.min()}, {df.date_time.max()}')
    print(f'Data label true: {df.label.value_counts() / len(df)}')
    df = df[SCALED_COLS + GENERATED_COLS + ENCODED_COLS + ['timestamp_raw', 'label', 'total_events']]
    df[SCALED_COLS + GENERATED_COLS] = scaler.fit_transform(df[SCALED_COLS + GENERATED_COLS].values)
    return df.astype('float32')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_index', type=int, default=10)
    parser.add_argument('--input_path', type=str, default='../datasets/frequency_encoded_data')
    parser.add_argument('--output_path', type=str, default='torch_ready_data')
    parser.add_argument('--data_subset', type=int, default=60, help='Number of files to read from input path')
    return parser.parse_args()

def main(args):
    #
    torch.set_printoptions(sci_mode=False)
    torch.set_printoptions(precision=4)

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)


    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    print(f"Starting {current_time}\nsubset of data: {args.data_subset}\nreading from {args.input_path}\nwrite_path {args.output_path}\nseq_list {args.seq_list}")


    files = glob.glob(f'{args.input_path}/*.csv')
    files = sorted(list(files))
    files = files[:args.data_subset]

    print(f"Using {len(files)} files")


    df = prepare_for_sessionization(files, MinMaxScaler())
    print('Dataframe columns:')
    pp.pprint(list(df.columns))

    for seq in args.seq_list:
        output_path = os.path.join(
            args.output_path,
            f'files_used_{args.data_subset}',
        )
        sessionize = SessionizeData(df, seq, output_path, save_s3=args.save_s3)
        sessionize.generate_sequence()
        print(f'Finished writing {seq} to disk')
    print(f'Exiting application...')




class Arguments:
    def __init__(self, seq_list):
        self.seq_list = seq_list
        self.input_path = 'datasets/frequency_encoded_data'
        self.output_path = 'datasets/torch_ready_data_2'
        self.data_subset = 5
        self.save_s3 = False


if __name__ == "__main__":
    args = Arguments([10, 20, 30])
    main(args)
