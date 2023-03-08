from model_development.generate_datasets import DatasetPartition
from model_development.generate_datasets import FILE_INDEX
from model_development.generate_datasets import partition_files

READ_PATH = '../torch_ready_data_v3_part'
WRITE_PATH = '../torch_ready_data_v3_datasets'

import numpy as np

TEST_SMALL = {
    'TRAIN:': 7500,
    'TEST': 2500,
    'EVAL': 2000
}

TEST_MEDIUM = {
    'TRAIN': 75000,
    'TEST': 25000,
    'EVAL': 20000
}

def test_subsets_across_partitions_unique():
    files = partition_files(READ_PATH, 10)
    file_ints = [int(file.split('_')[-1].split('.')[0]) for file in files]
    np.testing.assert_array_equal(file_ints, np.arange(FILE_INDEX))

def test_partitions_small_and_medium_unique():

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    np.random.seed(42)

    files = partition_files(READ_PATH, 5)
    dataset_partition = DatasetPartition(files, partition_small=TEST_SMALL, partition_middle=TEST_MEDIUM)
    dataset_partition.partition()

    assert dataset_partition.small_dataset.shape[0] == sum(TEST_SMALL.values())
    assert dataset_partition.medium_dataset.shape[0] == sum(TEST_MEDIUM.values())

    timestamp_small = dataset_partition.small_dataset[:, 1].astype(int)
    timestamp_medium = dataset_partition.medium_dataset[:, 1].astype(int)
    np.testing.assert_array_equal(timestamp_small[:1000], np.sort(timestamp_small[:1000]))
    np.testing.assert_array_equal(timestamp_medium[:1000], np.sort(timestamp_medium[:1000]))
