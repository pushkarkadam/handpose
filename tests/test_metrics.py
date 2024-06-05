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

def test_non_max_suppression():
    """Tests non_max_suppression"""

    x = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    y = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    w = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    h = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)

    kpt_truth = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
                'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
                'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
                'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
                }

    kpt_conf_truth = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
                'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
                }

    conf_truth = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)

    classes_truth = torch.cat([torch.Tensor([[0,0,0],[0,0,0],[0,0,0]]).reshape(1,1,3,3), 
                            torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)], dim=1)

    head = {'x': x,
            'y': y,
            'w': w,
            'h': h,
            'conf': conf_truth,
            'kpt': kpt_truth,
            'k_conf': kpt_conf_truth,
            'classes': classes_truth    
            }

    best_head = handpose.helpers.best_box(head, iou_threshold=0.8)

    head_data = handpose.helpers.extract_head(best_head)

    nms_boxes = handpose.metrics.non_max_suppression(head_data, iou_threshold=0.5)

    assert isinstance(nms_boxes['nms_box_indices'], list)

    torch.testing.assert_close(nms_boxes['nms_box_indices'][0], torch.Tensor([0]).type(torch.int64))
    torch.testing.assert_close(nms_boxes['boxes'][0], torch.Tensor([[0.25, 0.25, 0.75, 0.75]]))

def test_mean_average_precision1():
    """Tests mean_average_precision()"""

    truth_boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32),
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
    ]

    truth_labels = [
        [torch.tensor(0)],
        [torch.tensor(1)]
    ]

    truth = {'boxes': truth_boxes,
            'labels': truth_labels
            }

    pred_boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32),
        []
    ]

    pred_score = [
        [torch.tensor(0.8)],
        []
    ]

    pred_labels = [
        [torch.tensor(0)],
        []
    ]

    pred = {'conf_score': pred_score,
            'labels': pred_labels,
            'boxes': pred_boxes
        }

    result = handpose.metrics.mean_average_precision(pred, truth, iou_threshold=0.5, num_classes=2)

    assert(result['mAP'] == 0.5)
    torch.testing.assert_close(result['confusion_matrix'], torch.tensor([[1,0],[0,0]], dtype=torch.int32))

def test_mean_average_precision2():
    """Tests mean_average_precision()"""

    truth_boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32),
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
    ]

    truth_labels = [
        [torch.tensor(0)],
        [torch.tensor(1)]
    ]

    truth = {'boxes': truth_boxes,
            'labels': truth_labels
            }

    pred_boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32),
        []
    ]

    pred_score = [
        [torch.tensor(0.8)],
        []
    ]

    pred_labels = [
        [torch.tensor(1)],
        []
    ]

    pred = {'conf_score': pred_score,
            'labels': pred_labels,
            'boxes': pred_boxes
        }

    result = handpose.metrics.mean_average_precision(pred, truth, iou_threshold=0.5, num_classes=2)

    assert(result['mAP'] == 0.0)
    torch.testing.assert_close(result['confusion_matrix'], torch.tensor([[0,1],[0,0]], dtype=torch.int32))

def test_mean_average_precision3():
    """Tests mean_average_precision()"""
    truth_boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32),
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
    ]

    truth_labels = [
        [torch.tensor(0)],
        [torch.tensor(1)]
    ]

    truth = {'boxes': truth_boxes,
            'labels': truth_labels
            }

    pred_boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75]], dtype=torch.float32),
        []
    ]

    pred_score = [
        [torch.tensor(0.8), torch.tensor(0.4)],
        []
    ]

    pred_labels = [
        [torch.tensor(0), torch.tensor(1)],
        []
    ]

    pred = {'conf_score': pred_score,
            'labels': pred_labels,
            'boxes': pred_boxes
        }

    result = handpose.metrics.mean_average_precision(pred, truth, iou_threshold=0.5, num_classes=2)

    assert(result['mAP'] == 0.5)
    torch.testing.assert_close(result['confusion_matrix'], torch.tensor([[1,1],[0,0]], dtype=torch.int32))