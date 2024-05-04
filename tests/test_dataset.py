import sys
import pytest 
import torch
sys.path.append('../')

import handpose

def test_read_labels():
    text_file_path = 'tests/test_labels/test_label1.txt'
    labels = handpose.dataset.read_labels(text_file_path)

    assert (type(labels) == torch.Tensor)
    assert (labels.shape == (2, 4))

    text_file_path2 = 'tests/test_labels/test_label2.txt'
    labels2 = handpose.dataset.read_labels(text_file_path2)

    assert (type(labels2) == torch.Tensor)
    assert (labels2.shape == (1, 2))

    with pytest.raises(Exception) as e:
        labels = handpose.dataset.read_labels('no_file_exists.txt')

def test_list_files():
    test_directory = 'tests/test_labels'

    directories = handpose.dataset.list_files(test_directory)

    assert(len(directories) == 2)
    assert(type(directories[0]) == str)

    with pytest.raises(Exception) as e:
        directories = handpose.dataset.list_files('no_dir_exists')