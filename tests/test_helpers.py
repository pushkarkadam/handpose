import sys
import pytest 
import torch
sys.path.append('../')

import handpose

def test_load_variables():
    file_path = 'tests/test_config.yaml'
    variables = handpose.helpers.load_variables(file_path)
    head = variables['head']
    head_items = list(head.keys())

    train = variables['train']
    train_items = list(train.keys())
    epoch_num = train["epochs"]
    optimizer = train["optimizer"]
    momentum = train["momentum"]

    assert(type(variables) == dict)
    assert(head_items == ["grid_size", "num_boxes", "num_classes", "num_keypoints"])
    assert(train_items == ["epochs", "optimizer", "momentum"])
    assert(type(epoch_num) == int)
    assert(type(optimizer) == str)
    assert(type(momentum) == float)

    # Testing for wrong file path
    # must throw an exception
    file_path2 = 'no_file_exists.yaml'
    with pytest.raises(Exception) as e:
        variables = handpose.helpers.load_variables(file_path2)

def test_xywh_to_xyxy():
    """Function to test xywh_to_xyxy()"""
    m = 2
    S = 7
    
    x = torch.ones((m, 1, S, S)) * 0.5
    y = torch.ones((m, 1, S, S)) * 0.5
    w = torch.ones((m, 1, S, S)) * 0.5
    h = torch.ones((m, 1, S, S)) * 0.5

    x_min, y_min, x_max, y_max = handpose.helpers.xywh_to_xyxy(x, y, w, h)

    assert(x_min.shape == (m, 1, S, S))
    assert(y_min.shape == (m, 1, S, S))
    assert(x_max.shape == (m, 1, S, S))
    assert(x_max.shape == (m, 1, S, S))

    assert(float(torch.min(x_min)) == 0.25)
    assert(float(torch.min(y_min)) == 0.25)
    assert(float(torch.min(x_max)) == 0.75)
    assert(float(torch.min(y_max)) == 0.75)

    x = torch.ones((m, 1, S, S)) * 0.9
    y = torch.ones((m, 1, S, S)) * 0.1
    w = torch.ones((m, 1, S, S)) * 0.5
    h = torch.ones((m, 1, S, S)) * 0.5

    x_min, y_min, x_max, y_max = handpose.helpers.xywh_to_xyxy(x, y, w, h)
    
    assert(x_min.shape == (m, 1, S, S))
    assert(y_min.shape == (m, 1, S, S))
    assert(x_max.shape == (m, 1, S, S))
    assert(x_max.shape == (m, 1, S, S))

    torch.testing.assert_close(torch.min(y_min).unsqueeze(0), torch.Tensor([1e-6]))
    torch.testing.assert_close(torch.min(x_max).unsqueeze(0), torch.Tensor([1 - 1e-6]))

def test_best_box1():
    """Tests best_box()"""

    m = 2
    S = 3
    B = 2
    nkpt = 21
    nkpt_dim = 3
    nc = 2
    require_kpt_conf = True
    tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    out_features = S * S * tensor_ch

    torch.manual_seed(0)
    pred = torch.sigmoid(torch.randn((m, out_features)))

    head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)

    best_head = handpose.helpers.best_box(head, 0.8)

    assert(best_head['conf'].shape == (m, 1, S, S))
    assert(best_head['x'].shape == (m, 1, S, S))
    assert(best_head['y'].shape == (m, 1, S, S))
    assert(best_head['w'].shape == (m, 1, S, S))
    assert(best_head['h'].shape == (m, 1, S, S))

    assert(len(list(best_head['k_conf'].keys())) == nkpt)
    assert(len(list(best_head['kpt'].keys())) == 2 * nkpt)

    assert(isinstance(best_head['obj_indices'], list))
    assert(isinstance(best_head['obj_indices'][0], torch.Tensor))

def test_best_box2():
    """Tests best_box()"""

    m = 16
    S = 7
    B = 4
    nkpt = 21
    nkpt_dim = 3
    nc = 2
    require_kpt_conf = True
    tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    out_features = S * S * tensor_ch

    torch.manual_seed(0)
    pred = torch.sigmoid(torch.randn((m, out_features)))

    head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)

    best_head = handpose.helpers.best_box(head, 0.8)

    assert(best_head['conf'].shape == (m, 1, S, S))
    assert(best_head['x'].shape == (m, 1, S, S))
    assert(best_head['y'].shape == (m, 1, S, S))
    assert(best_head['w'].shape == (m, 1, S, S))
    assert(best_head['h'].shape == (m, 1, S, S))

    assert(len(list(best_head['k_conf'].keys())) == nkpt)
    assert(len(list(best_head['kpt'].keys())) == 2 * nkpt)