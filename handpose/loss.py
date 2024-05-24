import torch 



def box_loss(box_truth, box_pred, lambda_coord, epsilon=1e-6):
    """Box loss

    Calculates the box loss with ``x y w h`` bounding box coordinates
    of ground truth and prediction.

    .. math::
        L_{xy} = \\lambda_{coord} \\sum_{i=0}^{S^2} \\sum_{j=0}^{B} \\mathbbm{1}_{ij}^{\\text{obj}} \\left[ (x_i - \\hat{x}_i)^2 + (y_i - \\hat{y}_i)^2\\right]
        L_{wh} = \\lambda_{coord} \\sum_{i=0}^{S^2} \\sum_{j=0}^{B} \\mathbbm{1}_{ij}^{\\text{obj}} \\left[ \\left(\\sqrt{w_i + \\epsilon} - \\sqrt{\\hat{w}_i + \\epsilon} \\right)^2 + \\\\ \\left( \\sqrt{h_i + \\epsilon} - \\sqrt{\\hat{h}_i + \\epsilon}\\right)^2\\right]
        L_{box} = L_{xy} + L_{wh}

    Parameters
    ----------
    box_truth: tuple
        A tuple of ground truth bounding box coordinates ``(x, y, w, h)``.
    box_pred: tuple
        A tuple of ground truth bounding box coordinates ``(x_pred, y_pred, w_pred, h_pred)``.
    lamda_coord: float
        A multiplier used to scale the loss.
    epsilon: float, default ``1e-6``
        A term used to add to ``w`` and ``h`` to not make it too small.

    Returns
    -------
    torch.tensor
    
    """
    xt, yt, wt, ht = box_truth
    xp, yp, wp, hp = box_pred

    mse = torch.nn.MSELoss(reduction="sum")

    xd = mse(xt, xp)
    yd = mse(yt, yp)
    wd = mse(torch.sqrt(wt + epsilon), torch.sqrt(wp + epsilon))
    hd = mse(torch.sqrt(ht + epsilon), torch.sqrt(hp + epsilon))

    loss = lambda_coord * ((xd + yd) + (wd + hd))

    return loss