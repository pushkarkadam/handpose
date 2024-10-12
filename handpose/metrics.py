import torch 
from handpose.helpers import *
import torchvision
from torchvision.ops import box_iou


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
    """Calculates non maximum suppression (NMS) from the extracted data.

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

# def mean_average_precision(pred, truth, iou_threshold=0.5, num_classes=1):
#     """
#     Calculate Mean Average Precision (mAP) and confusion matrix for object detection.

#     Parameters
#     ----------
#     pred: dict
#         A dictionary of prediction with keys ``['conf_score', 'labels', 'boxes']``
#     truth: dict
#         A dictionary of truth with keys ``['conf_score', 'labels', 'boxes']``
#     iou_threshold : float, default=0.5
#         IoU threshold to consider a prediction as true positive.
#     num_classes : int, default=1
#         Number of object classes.

#     Returns
#     -------
#     dict
#         Dictionary containing mAP and confusion matrix.
#     """
#     pred_boxes = pred['boxes']
#     pred_scores = pred['conf_score']
#     pred_labels = pred['labels']
#     true_boxes = truth['boxes']
#     true_labels = truth['labels']
    
#     average_precisions = []
#     confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

#     for c in range(num_classes):
#         true_positives = []
#         false_positives = []
#         num_true_objects = 0

#         for pred_box, pred_score, pred_label, true_box, true_label in zip(pred_boxes, pred_scores, pred_labels, true_boxes, true_labels):
#             if len(pred_box) == 0:
#                 continue

#             # Convert list of tensors to a single tensor
#             pred_label = torch.tensor([label.item() for label in pred_label])
#             true_label = torch.tensor([label.item() for label in true_label])

#             num_true_objects += (true_label == c).sum().item()
#             detected = []

#             for i, box in enumerate(pred_box):
#                 if pred_label[i] != c:
#                     continue

#                 if true_box.numel() == 0:
#                     false_positives.append(1)
#                     true_positives.append(0)
#                     continue
                
#                 ious = box_iou(box.unsqueeze(0), true_box)
#                 max_iou, max_index = ious.max(1)

#                 if max_iou >= iou_threshold and max_index not in detected and true_label[max_index] == c:
#                     true_positives.append(1)
#                     false_positives.append(0)
#                     detected.append(max_index)
#                 else:
#                     false_positives.append(0)
#                     true_positives.append(0)

#             # Update confusion matrix
#             for i, box in enumerate(pred_box):
#                 pred_class = pred_label[i].item()
#                 if pred_class != c:
#                     continue
#                 if i in detected:
#                     confusion_matrix[c, c] += 1
#                 else:
#                     if len(true_label) > 0:
#                         true_class = true_label[max_index].item()
#                         confusion_matrix[true_class, pred_class] += 1
#                     else:
#                         confusion_matrix[0, pred_class] += 1

#         if len(true_positives) == 0:
#             average_precisions.append(0)
#             continue

#         true_positives = torch.tensor(true_positives)
#         false_positives = torch.tensor(false_positives)

#         true_positives = torch.cumsum(true_positives, dim=0)
#         false_positives = torch.cumsum(false_positives, dim=0)

#         precisions = true_positives / (true_positives + false_positives + 1e-16)
#         recalls = true_positives / (num_true_objects + 1e-16)

#         precisions = torch.cat((torch.tensor([0]), precisions))
#         recalls = torch.cat((torch.tensor([0]), recalls))

#         for i in range(len(precisions) - 1, 0, -1):
#             precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])

#         indices = torch.where(recalls[1:] != recalls[:-1])[0]
#         average_precision = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
#         average_precisions.append(average_precision.item())

#     return {
#         'mAP': sum(average_precisions) / num_classes,
#         'confusion_matrix': confusion_matrix
#     }

def pr_curve(truth, pred, iou_threshold=0.5):
    """Returns precision and recall curve values.

    Parameters
    ----------
    truth: dict
        A dictionary of truth values
    pred: dict
        A dictionary of predicted values
    iou_threshold: float, default ``0.5``
        Intersection over union (IOU) threshold.

    Returns
    -------
    tuple
        A tuple of precision and recall values.
    
    """

    if type(truth) == dict and type(pred) == dict:
        pred_boxes = pred['boxes']
        true_boxes = truth['boxes']
    else:
        pred_boxes = pred
        true_boxes = truth

    TP = []
    FP = []
    conf = []
    num_true_det = len(true_boxes)
    
    for pred_box, true_box in zip(pred_boxes, true_boxes):
        # Checking if there was no detection
        if type(pred_box) == list:
            # For no detection, appending confidence as 0
            conf.append(0.0)
            FP.append(True)
            TP.append(False)
            continue
        for det in pred_box:
            # Calculating the iou score as the confidence
            iou_score = box_iou(det.unsqueeze(0), true_box)
            
            # Appending the iou score as the confidence
            conf.append(iou_score.item())

            if iou_score >= iou_threshold:
                TP.append(True)
                FP.append(False)
            else:
                FP.append(True)
                TP.append(False)

    # Sorting the confidence index in descending order
    conf_index = np.flip(np.array(conf).argsort())
    conf = np.array(conf)[conf_index]
    TP = np.array(TP)[conf_index]
    FP = np.array(FP)[conf_index]

    # precision curve
    prec_curve = TP.cumsum() / (TP.cumsum() + FP.cumsum())

    # recall curve
    recall_curve = TP.cumsum() / num_true_det

    return prec_curve, recall_curve