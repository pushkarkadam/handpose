import sys
import os
import pytest 
import torch

sys.path.append('../')

import handpose

REPO = 'pytorch/vision:v0.17.1'

def test_TransferNetwork1():
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
    weights = 'ResNet18_Weights.IMAGENET1K_V1'
    model_name = 'resnet18'
    freeze_weights = True

    test_out_features = S * S * (B * (5 + 2 * nkpt) + nc)

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
    weights = 'AlexNet_Weights.IMAGENET1K_V1'
    model_name = 'alexnet'
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

    X = torch.randn((1, 3, 448, 448))
    pred = model(X)

    assert isinstance(pred, torch.Tensor)
    assert (pred.shape == (1, test_out_features))

    params = [param for param in model.parameters()]

    # Testing not freezing the weights
    assert (params[0].requires_grad == True)
    assert (params[-1].requires_grad == True)

def test_network_head1():
    m = 16
    S = 19
    B = 4
    nkpt = 21
    nkpt_dim = 3
    nc = 2
    require_kpt_conf = True
    tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    out_features = S * S * tensor_ch

    torch.manual_seed(0)
    pred = torch.randn((m, out_features))

    head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)
    
    assert(list(head.keys()) == ['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'classes'])

    ch_dim = 0
    for k, v in head.items():
        if isinstance(v, torch.Tensor):
            _, ch_n, _, _ = v.shape
            ch_dim += ch_n
        else:
            for i,j in v.items():
                _, ch_n, _, _ = j.shape
                ch_dim += ch_n

    assert (ch_dim == tensor_ch)
    assert (len(list(head['k_conf'].keys())) == nkpt)
    assert (len(list(head['kpt'].keys())) == nkpt * 2)

def test_network_head2():
    m = 16
    S = 19
    B = 4
    nkpt = 21
    nkpt_dim = 3
    nc = 2
    require_kpt_conf = True
    tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    out_features = S * S * tensor_ch

    torch.manual_seed(0)
    pred = torch.randn(out_features)

    with pytest.raises(Exception) as e:
        head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)

def test_head_activations():
    """Tests head_activation() function"""
    m = 16
    S = 19
    B = 4
    nkpt = 21
    nkpt_dim = 3
    nc = 2
    require_kpt_conf = True
    tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    out_features = S * S * tensor_ch

    torch.manual_seed(0)
    pred = torch.randn((m, out_features))

    head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)

    head_act = handpose.network.head_activations(head)

    assert(list(head_act.keys()) == ['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'classes'])

    assert(head['x'].shape == head_act['x'].shape)
    assert(float(torch.max(head_act['x'])) <= 1.0)
    assert(float(torch.min(head_act['x'])) >= 0.0)

    head_classes = torch.sum(head_act['classes'][0], dim=0)
    ones = torch.ones(S, S)
    torch.testing.assert_close(head_classes, ones)