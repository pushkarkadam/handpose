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
