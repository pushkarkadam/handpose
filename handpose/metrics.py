import torch 


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