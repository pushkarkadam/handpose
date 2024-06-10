import sys
import os
import pytest 
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import collections

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

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_optimizer_initialization():
    model = SimpleModel()
    optimizer_cls = optim.SGD
    optimizer = handpose.optimizer.Optimizer(model=model, optimizer_cls=optimizer_cls, lr=0.01, momentum=0.9)

    assert isinstance(optimizer, handpose.optimizer.Optimizer)
    assert isinstance(optimizer.optimizer, optim.SGD)
    assert optimizer.optimizer.defaults['lr'] == 0.01
    assert optimizer.optimizer.defaults['momentum'] == 0.9

def test_optimizer_parameters():
    model = SimpleModel()
    optimizer_cls = optim.Adam
    optimizer = handpose.optimizer.Optimizer(model=model, optimizer_cls=optimizer_cls, lr=0.001)

    param_groups = optimizer.optimizer.param_groups
    for group in param_groups:
        for param in group['params']:
            assert param.requires_grad

def test_optimizer_step():
    model = SimpleModel()
    optimizer_cls = optim.SGD
    optimizer = handpose.optimizer.Optimizer(model=model, optimizer_cls=optimizer_cls, lr=0.01)

    # Forward pass
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Perform an optimization step
    optimizer.optimizer.step()

    # Check if the parameters were updated (i.e., gradients are no longer zero)
    for param in model.parameters():
        if param.grad is not None:
            assert torch.any(param.grad != 0)

def test_scheduler1():
    """Test for scheduler"""
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = handpose.optimizer.Scheduler(optimizer, lr_scheduler.StepLR, **{'step_size': 7, 'gamma':0.1}).scheduler

    assert(isinstance(scheduler, torch.optim.lr_scheduler.StepLR))
    assert(scheduler.state_dict()['step_size'] == 7)
    assert(scheduler.state_dict()['gamma'] == 0.1)

def test_scheduler1():
    """Test for scheduler"""
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = handpose.optimizer.Scheduler(optimizer, lr_scheduler.MultiStepLR, **{'milestones': [30, 80], 'gamma':0.1}).scheduler

    assert(isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR))
    assert(isinstance(scheduler.state_dict()['milestones'], collections.Counter))
    assert(scheduler.state_dict()['gamma'] == 0.1)