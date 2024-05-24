import torch 


def box_loss(box_truth, box_pred, lambda_coord, epsilon=1e-6):
    r"""Box loss

    Calculates the box loss with ``x y w h`` bounding box coordinates
    of ground truth and prediction.

    .. math::
        L_{xy} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right] \\
        L_{wh} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ \left(\sqrt{w_i + \epsilon} - \sqrt{\hat{w}_i + \epsilon} \right)^2 + \\ \left( \sqrt{h_i + \epsilon} - \sqrt{\hat{h}_i + \epsilon}\right)^2\right] \\
        L_{box} = L_{xy} + L_{wh}

    .. note::
        The indicator function :math:`1` used in the above equation performs the work of selecting the best bounding box :math:`j`
        for each grid cell :math:`i`.
        This selection process is done using 
        :func:`handpose.helpers.best_box`.

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

    Examples
    --------
    >>> import torch
    >>> x = torch.ones((16,1,19,19))
    >>> y = torch.ones((16,1,19,19))
    >>> w = torch.ones((16,1,19,19))
    >>> h = torch.ones((16,1,19,19))
    >>> x_p = copy.deepcopy(x)
    >>> y_p = copy.deepcopy(y)
    >>> w_p = copy.deepcopy(w)
    >>> h_p = copy.deepcopy(h)
    >>> box_truth = (x, y, w, h)
    >>> box_pred = (x_p, y_p, w_p, h_p)
    >>> loss = handpose.loss.box_loss(box_truth, box_pred, lambda_coord=0.5)
    tensor(0.)
    
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