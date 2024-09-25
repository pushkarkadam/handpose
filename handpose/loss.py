import torch 
import copy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    batch_size = xt.size(0)

    # Works as an indicator function
    obj_indicator = obj_conf.to(DEVICE)

    # Mean square error
    mse = torch.nn.MSELoss(reduction="sum")

    # Using indicator function by multiplying the box coordinates
    xd = mse(xt * obj_indicator, xp * obj_indicator)
    yd = mse(yt * obj_indicator, yp * obj_indicator)
    wd = mse(torch.sqrt(wt + torch.tensor(epsilon).to(DEVICE)) * obj_indicator, torch.sqrt(wp + torch.tensor(epsilon).to(DEVICE)) * obj_indicator)
    hd = mse(torch.sqrt(ht + torch.tensor(epsilon).to(DEVICE)) * obj_indicator, torch.sqrt(hp + torch.tensor(epsilon).to(DEVICE)) * obj_indicator)

    loss = torch.tensor(lambda_coord).to(DEVICE) * ((xd + yd) + (wd + hd))

    return loss / torch.tensor(batch_size).to(DEVICE)

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

    batch_size = ct.size(0)

    obj_loss = mse(ct * obj_indicator.to(DEVICE), cp * obj_indicator.to(DEVICE))
    noobj_loss = mse(ct * noobj_indicator.to(DEVICE), cp * noobj_indicator.to(DEVICE))

    loss = obj_loss + torch.tensor(lambda_noobj).to(DEVICE) * noobj_loss

    return loss / torch.tensor(batch_size).to(DEVICE)

def class_loss(classes_truth, classes_pred, obj_conf, lambda_class=0.05):
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
    bce_loss = torch.nn.BCELoss(reduction='sum')

    batch_size = ct.size(0)

    loss = bce_loss(cp * obj_indicator, ct * obj_indicator)

    return (torch.tensor(lambda_class).to(DEVICE) * loss) / torch.tensor(batch_size).to(DEVICE)

def class_loss_mse(classes_truth, classes_pred, obj_conf, lambda_class=1):
    r"""Class loss using regression.

    .. math::

        loss_{class} = \sum_{i = 0}^{S^2}
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
    
    """

    ct = classes_truth
    cp = classes_pred
    obj_indicator = obj_conf

    # Extracting batch size
    batch_size = ct.size(0)

    # Mean square error 
    mse = torch.nn.MSELoss(reduction="sum")

    loss = mse(ct * obj_indicator, cp * obj_indicator)
    
    return (torch.tensor(lambda_class).to(DEVICE) * loss) / torch.tensor(batch_size).to(DEVICE)

def kpt_loss(kpt_truth, kpt_pred, obj_conf, lambda_kpt=0.5):
    r"""Keypoint loss.

    Keypoint loss that uses mean square error.

    .. math::
        L_{kpt} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ \frac{ (kx_i - \hat{kx}_i)^2 + (ky_i - \hat{ky}_i)^2}{2} \right] \\

    Parameters
    ----------
    kpt_truth: dict
        A dictionary of truth keypoints
    kpt_pred: dict
        A dictionary of prediction keypoints
    obj_conf: torch.Tensor
        A torch tensor of size ``(m, 1, S, S)``
    lambda_kpt: float, default ``0.5``
        A multiplier.

    Returns
    -------
    torch.tensor

    Examples
    --------
    >>> kpt_truth = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
            }
    >>> kpt_pred = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
            }
    >>> obj_conf = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)
    >>> loss = kpt_loss(kpt_truth, kpt_pred, obj_conf, nkpt)
    
    """

    d = torch.tensor(0.0).to(DEVICE)

    nkpt = int(len(list(kpt_truth.keys())) / 2)
    nkpt_pred = int(len(list(kpt_truth.keys())) / 2)

    try:
        assert(nkpt == nkpt_pred)
    except Exception as e:
        print("\033[91m" + "Mismatch keypoints from truth and prediction.")
        print("\033[91m"+ f"Truth keypoints {nkpt} != Prediction keypoitns {nkpt_pred}")
        raise

    obj_indicator = obj_conf.to(DEVICE)
    mse = torch.nn.MSELoss(reduction="sum")

    batch_size = obj_conf.size(0)
    
    for i in range(nkpt):
        # truth tensor
        kx_truth = kpt_truth[f'kx_{i}']
        ky_truth = kpt_truth[f'ky_{i}']

        # prediction tensors
        kx_pred = kpt_pred[f'kx_{i}'] * obj_indicator
        ky_pred = kpt_pred[f'ky_{i}'] * obj_indicator

        dx = mse(kx_truth, kx_pred)
        dy = mse(ky_truth, ky_pred)

        d += (dx + dy)

    loss = torch.tensor(1).to(DEVICE) - torch.exp(-d)

    return (torch.tensor(lambda_kpt).to(DEVICE) * loss) / torch.tensor(batch_size).to(DEVICE)

def kpt_loss_euclidean(kpt_truth, kpt_pred, obj_conf, lambda_kpt=0.5):
    r"""Keypoint loss.

    Keypoint loss that uses mean square error.

    .. math::
        L_{kpt} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left[ \frac{ (kx_i - \hat{kx}_i)^2 + (ky_i - \hat{ky}_i)^2}{2} \right] \\

    Parameters
    ----------
    kpt_truth: dict
        A dictionary of truth keypoints
    kpt_pred: dict
        A dictionary of prediction keypoints
    obj_conf: torch.Tensor
        A torch tensor of size ``(m, 1, S, S)``
    lambda_kpt: float, default ``0.5``
        A multiplier.

    Returns
    -------
    torch.tensor

    Examples
    --------
    >>> kpt_truth = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
            }
    >>> kpt_pred = {'kx_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_0': torch.Tensor([[0,0,0],[0,0.5,0], [0,0,0]]).reshape(1,1,3,3),
             'kx_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
             'ky_1': torch.Tensor([[0,0,0],[0,0.2,0], [0,0,0]]).reshape(1,1,3,3),
            }
    >>> obj_conf = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3)
    >>> loss = kpt_loss(kpt_truth, kpt_pred, obj_conf, nkpt)
    
    """

    loss = torch.tensor(0.0).to(DEVICE)

    epsilon = torch.tensor(1e-6).to(DEVICE)

    nkpt = int(len(list(kpt_truth.keys())) / 2)
    nkpt_pred = int(len(list(kpt_truth.keys())) / 2)

    try:
        assert(nkpt == nkpt_pred)
    except Exception as e:
        print("\033[91m" + "Mismatch keypoints from truth and prediction.")
        print("\033[91m"+ f"Truth keypoints {nkpt} != Prediction keypoitns {nkpt_pred}")
        raise

    obj_indicator = obj_conf.to(DEVICE)
    mse = torch.nn.MSELoss(reduction="sum")

    batch_size = obj_conf.size(0)
    
    for i in range(nkpt):
        # truth tensor
        kx_truth = kpt_truth[f'kx_{i}']
        ky_truth = kpt_truth[f'ky_{i}']

        # prediction tensors
        kx_pred = kpt_pred[f'kx_{i}'] * obj_indicator
        ky_pred = kpt_pred[f'ky_{i}'] * obj_indicator

        dx = mse(kx_truth, kx_pred)
        dy = mse(ky_truth, ky_pred)

        loss += (torch.square(dx) + torch.square(dy))

    return (torch.tensor(lambda_kpt).to(DEVICE) * loss) / torch.tensor(batch_size).to(DEVICE)

def kpt_conf_loss(k_conf_truth, k_conf_pred, obj_conf, lambda_kpt_conf=1):
    r"""Calculates the loss of keypoint confidence.

    .. math::
        L_{obj} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{\text{obj}} \left( k_i - \hat{k_i} \right)
    
    Parameters
    ----------
    k_conf_truth: dict
        A dictionary of truth keypoint confidence.
    k_conf_pred: dict
        A dictionary of prediction keypoint confidence.
    obj_conf: torch.Tensor
        A torch tensor of size ``(m, 1, S, S)``
    lambda_kpt_conf: int, default ``1``
        A multiplier.

    Returns
    -------
    torch.tensor

    Examples
    --------
    >>> kpt_truth = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
             'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
            }
    >>> kpt_pred = {'k_conf_0': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3),
                 'k_conf_1': torch.Tensor([[0,0,0],[0,1,0], [0,0,0]]).reshape(1,1,3,3)
                } 
    >>> obj_conf = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(1,1,3,3) 
    >>> nkpt = 2
    >>> loss = kpt_conf_loss(kpt_truth, kpt_pred, obj_conf, nkpt)

    """
    loss = torch.tensor(0.0).to(DEVICE)

    nkpt = int(len(list(k_conf_truth.keys())))
    nkpt_pred = int(len(list(k_conf_pred.keys())))

    try:
        assert(nkpt == nkpt_pred)
    except Exception as e:
        print("\033[91m" + "Mismatch keypoints from truth and prediction.")
        print("\033[91m"+ f"Truth keypoints {nkpt} != Prediction keypoitns {nkpt_pred}")
        raise

    obj_indicator = obj_conf.to(DEVICE)
    mse = torch.nn.MSELoss(reduction="sum")

    batch_size = obj_conf.size(0)
    
    for i in range(nkpt):
        # truth tensor
        conf_truth = k_conf_truth[f'k_conf_{i}']
        conf_pred = k_conf_pred[f'k_conf_{i}'] * obj_indicator

        loss += mse(conf_truth, conf_pred)

    return (torch.tensor(lambda_kpt_conf).to(DEVICE) * loss) / torch.tensor(batch_size).to(DEVICE)

def loss_fn(truth, prediction, lambda_coord=5, lambda_noobj=0.5, epsilon=1e-6, lambda_kpt=0.5, lambda_kpt_conf=0.5):
    r"""Computes loss and returns a dictionary of all the losses.

    The losses returned are:
    
    - Total Loss
    - Box Loss
    - Class Loss
    - Keypoint Loss
    - Keypoint Confidence Loss

    
    Parameters
    ----------
    truth: dict
        A dictionary of ground truth.
    prediction: dict
        A dictionary of prediction.
    lambda_coord: float, default ``5``
        Coordinate multiplier.
    lambda_noobj: float, default ``0.5``
        No object multiplier.
    epsilon: float, default ``1e-6``
        A term used to add to ``w`` and ``h`` to not make it too small.
    lambda_kpt: float, default ``0.5``
        A multiplier keypoint loss.
    lambda_kpt_conf: float, default ``0.5``
        A multiplier for confidence loss.

    Returns
    -------
    dict

    Examples
    --------
    >>> x = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    >>> y = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    >>> w = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    >>> h = torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1)
    >>> kpt_truth = {'kx_0': torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1),
                 'ky_0': torch.Tensor([[0.5], [0.5]]).reshape(2,1,1,1),
                 'kx_1': torch.Tensor([[0.2], [0.2]]).reshape(2,1,1,1),
                 'ky_1': torch.Tensor([[0.2], [0.2]]).reshape(2,1,1,1),
                }
    >>> kpt_conf_truth = {'k_conf_0': torch.Tensor([[1],[1]]).reshape(2,1,1,1),
                 'k_conf_1': torch.Tensor([[1],[1]]).reshape(2,1,1,1)
                }
    >>> conf_truth = torch.Tensor([[1],[1]]).reshape(2,1,1,1)
    >>> classes_truth = torch.cat([torch.Tensor([[0],[0]]).reshape(2,1,1,1), 
                               torch.Tensor([[1],[1]]).reshape(2,1,1,1)], dim=1)
    >>> truth = {'x': x,
             'y': y,
             'w': w,
             'h': h,
             'conf': conf_truth,
             'kpt': kpt_truth,
             'k_conf': kpt_conf_truth,
             'classes': classes_truth  
            }
    >>> prediction = copy.deepcopy(truth)
    >>> lambda_coord = 5
    >>> lambda_noobj = 0.5
    >>> epsilon = 1e-6
    >>> lambda_kpt = 0.5 
    >>> lambda_kpt_conf = 0.5
    >>> loss(truth, prediction, lambda_coord, lambda_noobj, epsilon, lambda_kpt, lambda_kpt_conf)
    
    """
    # Extracting values for box loss
    box_truth = (truth['x'], truth['y'], truth['w'], truth['h'])
    box_pred = (prediction['x'], prediction['y'], prediction['w'], prediction['h'])

    # Extracting values for confidence loss
    conf_truth = truth['conf']
    conf_pred = prediction['conf']

    # Extracting values for class loss 
    classes_truth = truth['classes']
    classes_pred = prediction['classes']

    # Extracting values for keypoint loss
    kpt_truth = truth['kpt']
    kpt_pred = prediction['kpt']

    # Extracting values for keypoint confidence
    kpt_conf_truth = truth['k_conf']
    kpt_conf_pred = prediction['k_conf']

    # Box loss
    L_box = box_loss(box_truth=box_truth, box_pred=box_pred, obj_conf=conf_truth, lambda_coord=lambda_coord, epsilon=epsilon)
    
    # confidence loss 
    L_conf = conf_loss(conf_truth=conf_truth, conf_pred=conf_pred, lambda_noobj=lambda_noobj)

    # class loss 
    L_class = class_loss_mse(classes_truth=classes_truth, classes_pred=classes_pred, obj_conf=conf_truth)

    # keypoint loss
    # L_kpt = kpt_loss_euclidean(kpt_truth=kpt_truth, kpt_pred=kpt_pred, obj_conf=conf_truth, lambda_kpt=lambda_kpt)
    L_kpt = kpt_loss(kpt_truth=kpt_truth, kpt_pred=kpt_pred, obj_conf=conf_truth, lambda_kpt=lambda_kpt)

    # keypoint confidence loss 
    L_kpt_conf = kpt_conf_loss(k_conf_truth=kpt_conf_truth, k_conf_pred=kpt_conf_pred, obj_conf=conf_truth, lambda_kpt_conf=lambda_kpt_conf)

    # Adding all the losses
    loss = L_box + L_conf + L_class + L_kpt

    all_losses = {'total_loss': loss.to(DEVICE), 
                  'box_loss': L_box.to(DEVICE),
                  'conf_loss': L_conf.to(DEVICE),
                  'class_loss': L_class.to(DEVICE),
                  'kpt_loss': L_kpt.to(DEVICE),
                  'kpt_conf_loss': L_kpt_conf.to(DEVICE) 
                 }

    return all_losses