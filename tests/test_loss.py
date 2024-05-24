import torch
import copy 
import sys 

sys.path.append('../')

import handpose 


def test_box_loss1():
    """Tests box_loss()"""

    x = torch.Tensor([0.5]).reshape(1,1,1,1)
    y = torch.Tensor([0.5]).reshape(1,1,1,1)
    w = torch.Tensor([0.5]).reshape(1,1,1,1)
    h = torch.Tensor([0.5]).reshape(1,1,1,1)

    x_p = copy.deepcopy(x)
    y_p = copy.deepcopy(y)
    w_p = copy.deepcopy(w)
    h_p = copy.deepcopy(h)

    obj_conf = torch.Tensor([1]).reshape(1,1,1,1)

    box_truth = (x, y, w, h)
    box_pred = (x_p, y_p, w_p, h_p)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)

    assert(loss == torch.tensor(0))

def test_box_loss2():
    """Tests box_loss()"""

    x = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)
    y = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)
    w = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)
    h = torch.Tensor([[0.5, 0.5],[0.5, 0.5]]).reshape(1,1,2,2)

    x_p = copy.deepcopy(x)
    y_p = copy.deepcopy(y)
    w_p = copy.deepcopy(w)
    h_p = copy.deepcopy(h)

    obj_conf = torch.Tensor([[1,0],[0, 0]]).reshape(1,1,2,2)

    box_truth = (x, y, w, h)
    box_pred = (x_p, y_p, w_p, h_p)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)

    assert(loss == torch.tensor(0))

def test_box_loss3():
    """Tests box_loss()"""

    x = torch.ones((16,1,19,19))
    y = torch.ones((16,1,19,19))
    w = torch.ones((16,1,19,19))
    h = torch.ones((16,1,19,19))

    x_p = copy.deepcopy(x)
    y_p = copy.deepcopy(y)
    w_p = copy.deepcopy(w)
    h_p = copy.deepcopy(h)

    obj_conf = torch.ones((16,1,19,19))

    box_truth = (x, y, w, h)
    box_pred = (x_p, y_p, w_p, h_p)

    loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)

    assert(loss == torch.tensor(0))