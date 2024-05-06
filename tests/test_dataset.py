import sys
import pytest 
import torch
sys.path.append('../')

import handpose

def test_read_label():
    text_file_path = 'tests/test_labels/test_label1.txt'
    labels = handpose.dataset.read_label(text_file_path)

    assert (type(labels) == torch.Tensor)
    assert (labels.shape == (2, 4))

    text_file_path2 = 'tests/test_labels/test_label2.txt'
    labels2 = handpose.dataset.read_label(text_file_path2)

    assert (type(labels2) == torch.Tensor)
    assert (labels2.shape == (1, 2))

    with pytest.raises(Exception) as e:
        labels = handpose.dataset.read_label('no_file_exists.txt')

def test_list_files():
    test_directory = 'tests/test_labels'

    directories = handpose.dataset.list_files(test_directory)

    assert(len(directories) == 2)
    assert(type(directories[0]) == str)

    with pytest.raises(Exception) as e:
        directories = handpose.dataset.list_files('no_dir_exists')

def test_split_label_tensor():
    tensor1 = torch.randn(2,5)
    tensor2 = torch.randn(1,5)

    tensor3 = torch.Tensor([[1,2,3,4], [5,6,7,8]])

    s1 = handpose.dataset.split_label_tensor(tensor1)
    s2 = handpose.dataset.split_label_tensor(tensor2)
    s3 = handpose.dataset.split_label_tensor(tensor3)

    assert(len(s1) == 2)
    assert(len(s2) == 1)
    assert(s1[0].shape == (1, 5))
    assert(s2[0].shape == (1, 5))
    torch.testing.assert_close(s3[0],torch.Tensor([[1,2,3,4]]))
    torch.testing.assert_close(s3[1],torch.Tensor([[5,6,7,8]]))