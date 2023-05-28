from derive_features.sessionization_pandas_v2 import DATA_COLS
from derive_features.sessionization_pandas_v2 import INITIAL_LOAD_COLUMNS
from derive_features.sessionization_pandas_v2 import SORT_DATE_COLS
from derive_features.sessionization_pandas_v2 import VALUE_COLS

T1_PASSING = True
import torch

import pytest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import glob

from derive_features.sessionization_pandas_v2 import (
    prepare_for_sessionization,
    SessionizeData
)
SEQUENCE_DATA_PATH = '../frequency_encoded_data/'


@pytest.fixture
def data_paths():
    return sorted(list(glob.glob(f'{SEQUENCE_DATA_PATH}/*.csv')))[:2]

@pytest.fixture(autouse=True)
def set_print_options():
    torch.set_printoptions(precision=10)
    torch.set_printoptions(sci_mode=False)


@pytest.fixture
def df_for_session(data_paths):
    return prepare_for_sessionization(data_paths, MinMaxScaler())

# @pytest.mark.skipif(T1_PASSING, reason='Test already passing')
@pytest.mark.usefixtures('data_paths')
def test_prepare_for_sessionization(data_paths):
    df = prepare_for_sessionization(data_paths, MinMaxScaler())

    assert set(df.columns.tolist()) == set(INITIAL_LOAD_COLUMNS + ['date_time', 'timestamp'])
    assert df['date_time'].dtype == 'datetime64[ns]'

    assert all(
        df[col].mean() >= 0 and df[col].mean() <= 1 and df[col].var() >= 0 and df[col].var() <= 1
        for col in df[DATA_COLS]
    )

    dater_sample = df[['date_time', 'timestamp']].head(100)
    dater_sample['validated_ts'] = pd.to_datetime(dater_sample['timestamp'], unit='s')

    assert all(dater_sample['date_time'] == dater_sample['validated_ts'])

@pytest.mark.skipif(T1_PASSING, reason='Test already passing')
@pytest.mark.usefixtures('df_for_session')
def test_sessionize_data_one_sequence(df_for_session):
    session = SessionizeData(df_for_session, max_sequence_index=3)
    assert session.device == torch.device('mps')
    np.testing.assert_array_equal(session.sequences, np.array([1, 2, 3]))

    result = session.generate_sequence()
    assert result.shape[1] == len(DATA_COLS)

@pytest.mark.skipif(T1_PASSING, reason='Test already passing')
@pytest.mark.usefixtures('df_for_session')
def test_sessionize_data_two_sequences(df_for_session):
    session = SessionizeData(df_for_session, max_sequence_index=3)

    res_1, res_2 = next(session._sequence_lazy()), next(session._sequence_lazy())

    assert res_1.shape[1] == len(DATA_COLS) == res_2.shape[1]
    assert res_1.shape[0] == res_2.shape[0]


@pytest.mark.usefixtures('df_for_session')
@pytest.mark.skipif(T1_PASSING, reason='Test already passing')
def test_sessionize_all_sequences(df_for_session):
    session = SessionizeData(df_for_session, max_sequence_index=3)
    sequences = session._shifters()

    assert sequences.shape[0] == df_for_session.shape[0]
    assert sequences.shape[1] == len(DATA_COLS * 3)


@pytest.mark.usefixtures('df_for_session')
@pytest.mark.skipif(T1_PASSING, reason='Test already passing')
def test_sessionization_works_e2e(df_for_session):
    session = SessionizeData(df_for_session, max_sequence_index=3)
    session.generate_sequence()

    label, timestamp = session.torch_sequences[:, 2],  session.torch_sequences[:, 1][:10000]
    features = session.torch_sequences[:, 3:]
    assert torch.equal(timestamp, torch.sort(timestamp)[0])
    assert features.shape[1] == len(DATA_COLS) * 4
    results = features.reshape((-1, 4, len(DATA_COLS)))
    sample_result = results[0]
    assert sample_result.shape == (4, len(DATA_COLS))

    assert all(np.isnan(sample_result[2].cpu().numpy()))
