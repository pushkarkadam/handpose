import torch 
from handpose.helpers import *
import torchvision


def intersection_over_union(box1, box2, minimum=1e-6):
    """Calculates IOU over two input boxes.

    Parameters
    ----------
    box1: list
        A list of ``torch.Tensor``.
        ``[x_min, y_min, x_max, y_max]`` format.
    box2: list
        A list of ``torch.Tensor``.
        Similar to ``box1``.
    minimum: float, default ``1e-6``
        A minimum value to avoid ``0/0``.

    Returns
    -------
    torch.Tensor

    Examples
    --------
    >>> box1 = [torch.Tensor([[[[0.2500]]]]), torch.Tensor([[[[0.2500]]]]), torch.Tensor([[[[0.7500]]]]), torch.Tensor([[[[0.7500]]]])]
    >>> box1 = [torch.Tensor([[[[0.2500]]]]), torch.Tensor([[[[0.2500]]]]), torch.Tensor([[[[0.7500]]]]), torch.Tensor([[[[0.7500]]]])]
    >>> iou = intersection_over_union(box1, box2)
    
    """
    xmin = torch.max(box1[0], box2[0])
    ymin = torch.max(box1[1], box2[1])
    xmax = torch.min(box1[2], box2[2])
    ymax = torch.min(box1[3], box2[3])
    
    
    # A1 = (x_max - x_min) * (y_max - y_min)
    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # intersection weight and height
    wi = xmax - xmin
    hi = ymax - ymin
    
    Ai = torch.clamp(wi, 0, 1) * torch.clamp(hi, 0, 1)
    
    iou = Ai / (A1 + A2 - Ai + minimum)

    return iou

def non_max_suppression(data, iou_threshold=0.5):
    """Calculates non maximum suppression from the extracted data.

    Parameters
    ----------
    data: dict
        A dictionary of keys ``['conf_score', 'class_idx', 'x', 'y', 'w', 'h', 'kx', 'ky']`` from
        :func:`handpose.helpers.extract_head`.
    iou_threshold: float, default ``0.5``
        IoU threshold for NMS algorihtm.

    Returns
    -------
    dict

    Examples
    --------
    >>> x = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    >>> y = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    >>> w = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    >>> h = torch.Tensor([[0,0,0],[0,0.5,0],[0,0,0]]).reshape(1,1,3,3)
    >>> kpt_truth = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
                'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
                'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
                'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
                }
    >>> kpt_conf_truth = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
                'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
                }
    >>> conf_truth = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)
    >>> classes_truth = torch.cat([torch.Tensor([[0,0,0],[0,0,0],[0,0,0]]).reshape(1,1,3,3), 
                            torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)], dim=1)
    >>> head = {'x': x,
            'y': y,
            'w': w,
            'h': h,
            'conf': conf_truth,
            'kpt': kpt_truth,
            'k_conf': kpt_conf_truth,
            'classes': classes_truth    
            }
    >>> best_head = handpose.helpers.best_box(head, iou_threshold=0.8)
    >>> head_data = handpose.helpers.extract_head(best_head)
    >>> nms_boxes = handpose.metrics.non_max_suppression(head_data, iou_threshold=0.5)
    
    """
    m = len(data['conf_score'])
    
    nms_box_indices = []
    keep_boxes = []
    kx = []
    ky = []
    
    for i in range(m):
        n = len(data['conf_score'][i])
        boxes = []
        scores = []
        kx_i = []
        ky_i = []
        if n:
            for j in range(n):
                score = data['conf_score'][i][j]
                scores.append(score)
                
                x = data['x'][i][j]
                y = data['y'][i][j]
                w = data['w'][i][j]
                h = data['h'][i][j]

                kx_ = torch.Tensor(data['kx'][i][j])
                ky_ = torch.Tensor(data['ky'][i][j])

                kpt = kx_.shape[0]

                # Converting xywh to xyxy format
                xmin, ymin, xmax, ymax = xywh_to_xyxy(x, y, w, h)
                box = torch.Tensor([xmin, ymin, xmax, ymax])
                boxes.append(box)
                kx_i.append(torch.Tensor(kx_))
                ky_i.append(torch.Tensor(ky_))
                
            scores = torch.Tensor(scores).reshape(n)
            boxes = torch.cat(boxes, dim=0).reshape(n,4)
            kx_nms = torch.cat(kx_i, dim=0).reshape(n,kpt)
            ky_nms = torch.cat(ky_i, dim=0).reshape(n,kpt)
            
            nms_index = torchvision.ops.nms(boxes, scores, iou_threshold)
            nms_box_indices.append(nms_index)
            keep_boxes.append(boxes[nms_index])
            
            kx.append(kx_nms[nms_index])
            ky.append(ky_nms[nms_index])
            
        else:
            nms_box_indices.append([])
            keep_boxes.append([])
    
    return {"nms_box_indices": nms_box_indices,
            "boxes": keep_boxes,
            "kx": kx,
            "ky": ky
           }