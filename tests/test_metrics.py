import sys 
import torch 
import pytest 

sys.path.append('../')

import handpose


def test_intersection_over_union1():
    """Test for intersection_over_union()"""

    x1 = torch.Tensor([0.5]).reshape(1,1,1,1)
    y1 = torch.Tensor([0.5]).reshape(1,1,1,1)
    w1 = torch.Tensor([0.5]).reshape(1,1,1,1)
    h1 = torch.Tensor([0.5]).reshape(1,1,1,1)

    x2 = torch.Tensor([0.5]).reshape(1,1,1,1)
    y2 = torch.Tensor([0.5]).reshape(1,1,1,1)
    w2 = torch.Tensor([0.3]).reshape(1,1,1,1)
    h2 = torch.Tensor([0.6]).reshape(1,1,1,1)

    box1 = list(handpose.helpers.xywh_to_xyxy(x1, y1, w1, h1))
    box2 = list(handpose.helpers.xywh_to_xyxy(x2, y2, w2, h2))

    iou1 = handpose.metrics.intersection_over_union(box1, box1)

    assert(iou1.shape == (1,1,1,1))
    torch.testing.assert_close(torch.max(iou1), torch.tensor(1.000))

    iou2 = handpose.metrics.intersection_over_union(box1, box2)

    assert(iou2.shape == (1,1,1,1))
    torch.testing.assert_close(torch.max(iou2), torch.tensor(0.5357), rtol=1e-4, atol=1e-4)

def test_intersection_over_union2():
    """Test for intersection_over_union()"""

    m = 2
    S = 7

    x1 = torch.ones((m, 1, S, S)) * 0.5
    y1 = torch.ones((m, 1, S, S)) * 0.5
    w1 = torch.ones((m, 1, S, S)) * 0.5
    h1 = torch.ones((m, 1, S, S)) * 0.5

    x2 = torch.ones((m, 1, S, S)) * 0.5
    y2 = torch.ones((m, 1, S, S)) * 0.5
    w2 = torch.ones((m, 1, S, S)) * 0.3
    h2 = torch.ones((m, 1, S, S)) * 0.6

    box1 = list(handpose.helpers.xywh_to_xyxy(x1, y1, w1, h1))
    box2 = list(handpose.helpers.xywh_to_xyxy(x2, y2, w2, h2))

    iou = handpose.metrics.intersection_over_union(box1, box2)

    assert(iou.shape == (m, 1, S, S))
    torch.testing.assert_close(torch.max(iou), torch.tensor(0.5357), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch.min(iou), torch.tensor(0.5357), rtol=1e-4, atol=1e-4)
