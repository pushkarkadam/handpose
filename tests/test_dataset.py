import sys
import os
import pytest 
import torch
import numpy as np
from torch.utils.data import DataLoader
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

def test_label_tensor():
    label = torch.Tensor([[1,0.5,0.5,0.5,0.5,0.5,0.6],[0,0.7,0.8,0.2,0.1, 0.2,0.3]])

    t1 = handpose.dataset.label_tensor(label, S=3, nc=2, nkpt=1, cell_relative=True, require_kpt_conf=False)
    t2 = handpose.dataset.label_tensor(label, S=3, nc=2, nkpt=1, cell_relative=True, require_kpt_conf=True)
    ch = torch.Tensor([[0,0,0],[0,1,0],[0,0,1]])

    assert(t1.shape == (9, 3, 3))
    assert(t2.shape == (10, 3, 3))
    torch.testing.assert_close(t1[0], ch)
    torch.testing.assert_close(t2[0], ch)

    label2 = torch.Tensor([[1,0.5,0.5,0.5,0.5,0.5,0.6, 0.5, 0.1],[0,0.7,0.8,0.2,0.1, 0.2,0.3, 0.5, 0.1]])
    t3 = handpose.dataset.label_tensor(label2, S=3, nc=2, nkpt=2, cell_relative=True, require_kpt_conf=True)
    assert(t3.shape == (13, 3, 3))

def test_polar_kpt():
    r1, alpha1 = handpose.dataset.polar_kpt(torch.Tensor([[0.6]]), torch.Tensor([[0.6]]), torch.Tensor([[0.4]]), torch.Tensor([[0.4]]), torch.Tensor([[0.2]]), torch.Tensor([[0.2]]))
    r2, alpha2 = handpose.dataset.polar_kpt(torch.Tensor([0.6]), torch.Tensor([0.6]), torch.Tensor([0.4]), torch.Tensor([0.4]), torch.Tensor([0.6]), torch.Tensor([0.2]))
    
    torch.testing.assert_close(torch.round(r1, decimals=4), torch.Tensor([[0.5657]]))
    torch.testing.assert_close(torch.round(alpha1, decimals=4), torch.Tensor([[0.625]]))
    torch.testing.assert_close(torch.round(r2, decimals=2), torch.Tensor([0.40]))
    torch.testing.assert_close(torch.round(alpha2, decimals=2), torch.Tensor([0.75]))

def test_truth_head():
    label = torch.Tensor([[1,0.5,0.5,0.5,0.5,0.5,0.6],[0,0.7,0.8,0.2,0.1, 0.2,0.3]])

    t1 = handpose.dataset.label_tensor(label, S=3, nc=2, nkpt=1, cell_relative=True, require_kpt_conf=False)
    t2 = handpose.dataset.label_tensor(label, S=3, nc=2, nkpt=1, cell_relative=True, require_kpt_conf=True)

    head1 = handpose.dataset.truth_head(t1, S=3, nc=2, nkpt=1, require_kpt_conf=False)
    head2 = handpose.dataset.truth_head(t2, S=3, nc=2, nkpt=1, require_kpt_conf=True)

    with pytest.raises(Exception) as e:
        head3 = handpose.dataset.truth_head(t1, S=3, nc=2, nkpt=1, require_kpt_conf=True)

    assert(type(head1) == dict)
    assert(type(head2) == dict)
    assert(list(head1.keys()) == ['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'kpt_polar', 'classes'])
    assert(list(head2.keys()) == ['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'kpt_polar', 'classes'])
    assert(head1['k_conf'] == dict())
    assert(list(head2['k_conf'].keys()) == ['k_conf_0'])
    assert(head2['k_conf']['k_conf_0'].shape == (1, 3, 3))
    assert(head1['x'].shape == (1, 3, 3))
    assert(head1['y'].shape == (1, 3, 3))
    assert(head1['w'].shape == (1, 3, 3))
    assert(head1['h'].shape == (1, 3, 3))
    assert(head2['conf'].shape == (1, 3, 3))
    assert(head2['classes'].shape == (2, 3, 3))
    assert(type(head2['kpt_polar']) == dict)
    assert(list(head2['kpt_polar'].keys()) == ['r_0', 'alpha_0'])
    assert(head2['kpt_polar']['r_0'].shape == (1, 3, 3))
    assert(head2['kpt_polar']['alpha_0'].shape == (1, 3, 3))

    label2 = torch.Tensor([[1,0.5,0.5,0.5,0.5,0.5,0.6, 0.5, 0.1],[0,0.7,0.8,0.2,0.1, 0.2,0.3, 0.5, 0.1]])
    t3 = handpose.dataset.label_tensor(label2, S=3, nc=2, nkpt=2, cell_relative=True, require_kpt_conf=True)

    head4 = handpose.dataset.truth_head(t3, S=3, nc=2, nkpt=2, require_kpt_conf=True)

    assert(list(head4['kpt'].keys()) == ["kx_0", "ky_0", "kx_1", "ky_1"])
    assert(list(head4['k_conf'].keys()) == ["k_conf_0", "k_conf_1"])

    label3 = torch.Tensor([[1,0.5,0.5,0.5,0.5,0.5,0.6, 0.5, 0.1, 0.5, 0.2],[0,0.7,0.8,0.2,0.1, 0.2,0.3, 0.5, 0.1, 0.5, 0.2]])
    t4 = handpose.dataset.label_tensor(label3, S=3, nc=2, nkpt=3, cell_relative=True, require_kpt_conf=True)
    head5 = handpose.dataset.truth_head(t4, S=3, nc=2, nkpt=3, require_kpt_conf=True)

    assert(list(head5['kpt'].keys()) == ["kx_0", "ky_0", "kx_1", "ky_1", "kx_2", "ky_2"])
    assert(list(head5['k_conf'].keys()) == ["k_conf_0", "k_conf_1", "k_conf_2"])

    assert(head5['kpt']['kx_0'].shape == (1, 3, 3))
    assert(head5['kpt_polar']['r_0'].shape == (1, 3, 3))


    t5 = handpose.dataset.label_tensor(label3, S=3, nc=2, nkpt=3, cell_relative=True, require_kpt_conf=False)
    head6 = handpose.dataset.truth_head(t5, S=3, nc=2, nkpt=3, require_kpt_conf=False)

    assert(list(head6['kpt'].keys()) == ["kx_0", "ky_0", "kx_1", "ky_1", "kx_2", "ky_2"])
    assert(head6['k_conf'] == dict())
    assert(list(head6['k_conf'].keys()) == [])

def test_dataset():
    sys.path.append('..')
    img_dir = 'data/dev/train/images'
    label_dir = 'data/dev/train/labels'
    training_data = handpose.dataset.HandDataset(img_dir, label_dir, S=7, nc=2, nkpt=21, cell_relative=True, require_kpt_conf=True)
    
    # test 1
    torch.manual_seed(0)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False, drop_last=True)

    train_features, data = next(iter(train_dataloader))

    head = data['head']

    assert(train_features.shape == (1, 3, 224, 224))
    assert(type(head) == dict)
    assert(head['x'].shape == (1, 1, 7, 7))

    kpts = []
    kpts_polar = []
    for i in range(21):
        kpts.append(f'kx_{i}')
        kpts.append(f'ky_{i}')
        kpts_polar.append(f'r_{i}')
        kpts_polar.append(f'alpha_{i}')

    assert(list(head['kpt'].keys()) == kpts)
    assert(list(head['kpt_polar'].keys()) == kpts_polar)

    # test 2
    torch.manual_seed(0)
    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=False, drop_last=True)

    train_features, data = next(iter(train_dataloader))

    head = data['head']

    assert(train_features.shape == (16, 3, 224, 224))
    assert(type(head) == dict)
    assert(head['x'].shape == (16, 1, 7, 7))

    kpts = []
    kpts_polar = []
    for i in range(21):
        kpts.append(f'kx_{i}')
        kpts.append(f'ky_{i}')
        kpts_polar.append(f'r_{i}')
        kpts_polar.append(f'alpha_{i}')

    assert(list(head['kpt'].keys()) == kpts)
    assert(list(head['kpt_polar'].keys()) == kpts_polar)