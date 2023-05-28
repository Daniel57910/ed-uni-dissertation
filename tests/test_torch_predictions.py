import os
import pdb

import numpy as np
import pytest
import torch
from model_development.dataset import ClickstreamDataLoader
from model_development.dataset import ClickstreamDataset
from model_development.lstm_constant import BASE_BATCH_SIZE
from model_development.lstm_constant import MAX_SEQ_INDEX
from model_development.lstm_constant import N_FEATURES
from model_development.model import ClickstreamModel
PYTORCH_DATA_PATH = '../torch_ready_data_v3_datasets_test'


torch.set_printoptions(precision=3)
torch.set_printoptions(linewidth=500)
torch.set_printoptions(sci_mode=False)

@pytest.fixture
def return_tensor():
    # setup device mps or cuda or cpu
    return np.load(f'{PYTORCH_DATA_PATH}/small.npy')


@pytest.mark.usefixtures('return_tensor')
def test_dataset_implementation(return_tensor):

    clickstream_dataset = ClickstreamDataset(return_tensor)
    assert clickstream_dataset.features[:100].shape == (100, MAX_SEQ_INDEX, N_FEATURES)
    assert clickstream_dataset.labels[:100].shape == torch.Size([100])


@pytest.mark.usefixtures('return_tensor')
def test_dataloader_implementation_pad_null_clickstream(return_tensor):
    clickstream_dataset = ClickstreamDataset(return_tensor)
    clickstream_dataloader = ClickstreamDataLoader(clickstream_dataset)

    features, labels = next(iter(clickstream_dataloader))
    # 11th column
    assert features.shape == (BASE_BATCH_SIZE, MAX_SEQ_INDEX, N_FEATURES)
    assert labels.shape == torch.Size([BASE_BATCH_SIZE])
    assert features[0][1].equal(torch.tensor([0 for _ in range(N_FEATURES)]))


@pytest.mark.usefixtures('return_tensor')
def test_model_data_implementation(return_tensor):
    model = ClickstreamModel(
        f'{PYTORCH_DATA_PATH}/small.npy',
        batch_size=BASE_BATCH_SIZE,
        n_workers=8)

    training_partition, evaluation_partition = (
        model.training_data.shape[0] / return_tensor.shape[0],
        model.evaluation_data.shape[0] / return_tensor.shape[0]
    )

    assert np.isclose([training_partition, evaluation_partition], [0.75, 0.25]).all()

def test_model_forward_pass_implementation():
    model = ClickstreamModel(f'{PYTORCH_DATA_PATH}/small.npy', batch_size=BASE_BATCH_SIZE, n_workers=8)
    features, labels = next(iter(model.train_dataloader()))
    logits = model(features)

    assert logits.shape == torch.Size([BASE_BATCH_SIZE, 1])
