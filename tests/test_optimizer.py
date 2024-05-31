import sys
import os
import pytest 
import torch
import torch.optim as optim

sys.path.append('../')

import handpose

REPO = 'pytorch/vision:v0.17.1'

def test_optimizer_fn1():
    """Test for optimizer_fn() with ResNet18 and frozen weights"""
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = True
    weights = 'ResNet18_Weights.IMAGENET1K_V1'
    model_name = 'resnet18'
    freeze_weights = True

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        weights=weights,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    optimizer = handpose.optimizer.optimizer_fn(model, optim.SGD, **{'lr':0.01, 'momentum':0.9})

    assert isinstance(optimizer, optim.SGD)
    assert (optimizer.param_groups[0]['lr'] == 0.01)
    assert (optimizer.param_groups[0]['momentum'] == 0.9)
    assert (len(optimizer.param_groups[0]['params']) == 2)

def test_optimizer_fn2():
    """Test for optimizer_fn() with ResNet18 and unfrozen weights"""
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = True
    weights = 'ResNet18_Weights.IMAGENET1K_V1'
    model_name = 'resnet18'
    freeze_weights = False

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        weights=weights,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    optimizer = handpose.optimizer.optimizer_fn(model, optim.SGD, **{'lr':0.01, 'momentum':0.9})

    assert isinstance(optimizer, optim.SGD)
    assert (optimizer.param_groups[0]['lr'] == 0.01)
    assert (optimizer.param_groups[0]['momentum'] == 0.9)
    assert (len(optimizer.param_groups[0]['params']) == 62)

def test_optimizer_fn3():
    """Test for optimizer_fn() with AlexNet and unfrozen weights"""
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = True
    weights = 'AlexNet_Weights.IMAGENET1K_V1'
    model_name = 'alexnet'
    freeze_weights = False

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        weights=weights,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    optimizer = handpose.optimizer.optimizer_fn(model, optim.SGD, **{'lr':0.01, 'momentum':0.9})

    assert isinstance(optimizer, optim.SGD)
    assert (optimizer.param_groups[0]['lr'] == 0.01)
    assert (optimizer.param_groups[0]['momentum'] == 0.9)
    assert (len(optimizer.param_groups[0]['params']) == 16)