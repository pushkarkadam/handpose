import sys
import os
import pytest 
import torch

sys.path.append('../')

import handpose

REPO = 'pytorch/vision:v0.10.0'

def test_TransferNetwork1():
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = True
    pretrained = True
    model_name = 'resnet18'
    freeze_weights = True

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        pretrained=pretrained,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    assert isinstance(model, torch.nn.Module)
    assert (model.model.fc.out_features == test_out_features)
    
    params = [param for param in model.parameters()]

    assert (params[0].requires_grad == False)
    assert (params[-1].requires_grad == True)

def test_TransferNetwork2():
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = False
    pretrained=True
    model_name = 'resnet18'
    freeze_weights = True

    test_out_features = S * S * (B * (5 + 2 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        pretrained=pretrained,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    assert isinstance(model, torch.nn.Module)
    assert (model.model.fc.out_features == test_out_features)

    params = [param for param in model.parameters()]

    assert (params[0].requires_grad == False)
    assert (params[-1].requires_grad == True)

def test_TransferNetwork3():
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = True
    pretrained = True
    model_name = 'alexnet'
    freeze_weights = True

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        pretrained=pretrained,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    assert isinstance(model, torch.nn.Module)
    assert (model.model.classifier[-1].out_features == test_out_features)

    params = [param for param in model.parameters()]

    assert (params[0].requires_grad == False)
    assert (params[-1].requires_grad == True)

def test_TransferNetwork4():
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 224, 224)
    require_kpt_conf = True
    pretrained = True
    model_name = 'resnet18'
    freeze_weights = True

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        pretrained=pretrained,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    X = torch.randn((1, 3, 224, 224))
    pred = model(X)

    assert isinstance(pred, torch.Tensor)
    assert (pred.shape == (1, test_out_features))

def test_TransferNetwork5():
    S = 7
    B = 2
    nkpt = 21
    nc = 2
    input_size = (3, 448, 448)
    require_kpt_conf = True
    pretrained = True
    model_name = 'resnet18'
    freeze_weights = False

    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    model = handpose.network.TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        pretrained=pretrained,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    )

    X = torch.randn((1, 3, 448, 448))
    pred = model(X)

    assert isinstance(pred, torch.Tensor)
    assert (pred.shape == (1, test_out_features))

    params = [param for param in model.parameters()]

    # Testing not freezing the weights
    assert (params[0].requires_grad == True)
    assert (params[-1].requires_grad == True)