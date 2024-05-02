import sys
import pytest 
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

