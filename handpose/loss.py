import torch 
import copy


def box_loss(box_truth, box_pred, obj_conf, lambda_coord, epsilon=1e-6):
    r"""Box loss

    Calculates the box loss with ``x y w h`` bounding box coordinates
    of ground truth and prediction.

    .. math::
        L_{xy} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right] \\
        L_{wh} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ \left(\sqrt{w_i + \epsilon} - \sqrt{\hat{w}_i + \epsilon} \right)^2 + \\ \left( \sqrt{h_i + \epsilon} - \sqrt{\hat{h}_i + \epsilon}\right)^2\right] \\
        L_{box} = L_{xy} + L_{wh}

    .. note::
        The indicator function :math:`1` used in the above equation performs the work of
        selecting the best bounding box :math:`j`
        for each grid cell :math:`i`.
        The indicator 
        This selection process is done using 
        :func:`handpose.helpers.best_box`.

    Parameters
    ----------
    box_truth: tuple
        A tuple of ground truth bounding box coordinates ``(x, y, w, h)``.
    box_pred: tuple
        A tuple of ground truth bounding box coordinates ``(x_pred, y_pred, w_pred, h_pred)``.
    obj_conf: torch.Tensor
        A tensor which indicates whether the grid cell consists an object.
        This is the ``conf`` tensor in the ``head``.
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
    >>> obj_conf = torch.ones((16,1,19,19))
    >>> loss = handpose.loss.box_loss(box_truth, box_pred, obj_conf, lambda_coord=0.5)
    
    """
    xt, yt, wt, ht = box_truth
    xp, yp, wp, hp = box_pred

    # Works as an indicator function
    obj_indicator = obj_conf

    # Mean square error
    mse = torch.nn.MSELoss(reduction="sum")

    # Using indicator function by multiplying the box coordinates
    xd = mse(xt * obj_indicator, xp * obj_indicator)
    yd = mse(yt * obj_indicator, yp * obj_indicator)
    wd = mse(torch.sqrt(wt + epsilon) * obj_indicator, torch.sqrt(wp + epsilon) * obj_indicator)
    hd = mse(torch.sqrt(ht + epsilon) * obj_indicator, torch.sqrt(hp + epsilon) * obj_indicator)

    loss = lambda_coord * ((xd + yd) + (wd + hd))

    return loss

def conf_loss(conf_truth, conf_pred, lambda_noobj):
    r"""Confidence loss.

    Computes the confidence loss.

    .. math::
        L_{obj} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left( C_i - \hat{C_i} \right) \\
        L_{noobj} = \lambda_{noobj}\sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{noobj}} \left( C_i - \hat{C_i} \right) \\    
        L_{conf} = L_{obj} + \lambda_{noobj} L_{noobj}
    
    Parameters
    ----------
    conf_truth: torch.Tensor
        A tensor of confidence for ground truth.
    conf_pred: torch.Tensor
        A tensor of confidence for prediction.
    lambda_noobj: float
        A multiplier for noobj present in the grid cell.

    Returns
    -------
    torch.tensor

    Examples
    --------
    >>> conf_truth = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape((1,1,3,3))
    >>> conf_pred = torch.Tensor([[0.1,0.2,0.4],[0.8,0.9,0.7],[0.2,0.01,0.2]]).reshape((1,1,3,3))
    >>> loss = handpose.loss.conf_loss(conf_truth, conf_pred, 0.5)

    """

    # Equivalent of indicator function
    obj_indicator = copy.deepcopy(conf_truth)
    noobj_indicator = 1 - obj_indicator 
    
    ct = conf_truth
    cp = conf_pred
    mse = torch.nn.MSELoss(reduction="sum")

    obj_loss = mse(ct * obj_indicator, cp * obj_indicator)
    noobj_loss = mse(ct * noobj_indicator, cp * noobj_indicator)

    loss = obj_loss + lambda_noobj * noobj_loss

    return loss

def class_loss(classes_truth, classes_pred, obj_conf):
    r"""Class loss.

    .. math::

        L_{class} = \sum_{i = 0}^{S^2}
        1_i^{\text{obj}}
            \sum_{c \in \textrm{classes}}
                \left(
                    p_i(c) - \hat{p}_i(c)
                \right)^2
    
    
    Parameters
    ----------
    classes_truth: torch.Tensor
        A tensor of truth classes.
    classes_pred: torch.Tensor
        A tensor of prediction classes.
    obj_conf: torch.Tensor
        A tensor of object confidence from ground truth.

    Returns
    -------
    torch.tensor
    
    Examples
    --------
    >>> classes_truth = torch.cat([torch.ones(1,1,3,3), torch.zeros(1,1,3,3)], dim=1)
    >>> classes_pred = torch.cat([torch.ones(1,1,3,3), torch.zeros(1,1,3,3)], dim=1)
    >>> obj_conf = torch.ones((1,1,3,3))
    >>> loss = handpose.loss.class_loss(classes_truth, classes_pred, obj_conf)

    """
    ct = classes_truth
    cp = classes_pred
    obj_indicator = obj_conf

    # Binary Cross Entropy loss
    bce_loss = torch.nn.BCELoss()

    loss = bce_loss(cp * obj_indicator, ct * obj_indicator)

    return loss