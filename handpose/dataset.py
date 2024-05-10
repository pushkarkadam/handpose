import os
import torch
import numpy as np

def read_label(file_path):
    """Reads the labels from text file.
    
    Parameters
    ----------
    file_path: str
        Path to the ``.txt`` file.

    Returns
    -------
    list

    Examples
    --------
    >>> labels = handpose.dataset.read_label('label.txt')
    
    """
    labels = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                nums = line.split()
                line_num = []
                for num in nums:
                    line_num.append(float(num))
                labels.append(line_num)
    except Exception as e:
        print(e)
        raise
        
    return torch.Tensor(labels)

def list_files(directory):
    """Reads the label files.

    Parameters
    ----------
    directory: str
        Path to the directory

    Returns
    -------
    list

    Examples
    --------
    >>> handpose.dataset.list_files('directory')
    
    """

    file_names = []

    try:
        files = os.listdir(directory)

        file_names = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        print(e)
        raise
        
    return file_names

def split_label_tensor(tensor):
    """Splits ``(N,n)`` tensor to ``(N,1,n)`` and returns a list of ``(1,n)`` tensors with ``N`` elements.

    For image that has multiple detection, the labels are on ``N`` lines.
    The read_labels function read these ``N`` lines and converts them into a tensor
    of size ``(N, n)`` where ``N`` is the number of detected objects and ``n`` is the annotation.
    These ``N`` detections are to be returned as a list of ``N`` element where each element is
    a tensor of size ``(1,n)``.

    Parameters
    ----------
    tensor: torch.Tensor
        A torch tensor with the labels.

    Returns
    -------
    list
        A list of split tensor labels.

    Examples
    --------
    >>> t = torch.randn(2,5)
    >>> split_label_tensor(t)
    
    """
    split_tensor = tensor.unsqueeze(1)

    labels = []

    for st in split_tensor:
        labels.append(st)

    return labels

def label_tensor(label, S, nc, nkpt, cell_relative=True, require_kpt_conf=True):
    """Converts to tensor.

    Parameters
    ----------
    label: torch.Tensor
        A tensor with ``(N,n)`` dimension
        where ``N`` is the number of detection
        and ``n`` are the annotations [class x y w h kx1 kx2 ...]
    S: int
        Size of the grid cell
    nc: int
        Number of classes
    nkpt: int
        Number of keypoints
    cell_relative: bool, default ``True``
        Boolean to select relative to cell coordinates for center of bounding box.
    require_kpt_conf: bool, default ``True``
        Boolean to select for the keypoint confidence visibility.
    
    Returns
    -------
    torch.Tensor
        A torch tensor of dimension ``(5 + 3*nkpt + nc, S, S)`` with keypoint visibility confidence flag
        and ``(5 + 2*nkpt + nc, S, S)`` for truth without keypoint visibility confidence flag.

    Examples
    --------
    >>> l = torch.Tensor([[1,0.5,0.5,0.5,0.5,0.5,0.6],[0,0.7,0.8,0.2,0.1, 0.2,0.3]])
    >>> t = label_tensor(l, S=3, nc=2, nkpt=1, cell_relative=True, kpt_conf=False)
    
    """
    # 4 --> (x, y, w, h)
    # 2 --> (kx, ky)
    channel_dims = 4 + 2 * nkpt

    # Ensuring the dimension inputs match the label input
    try:
        assert (channel_dims == label[...,1:].shape[-1])
    except Exception as e:
        print(e)
        raise

    # Checking if the keypoint visibility flag is needed
    if require_kpt_conf:
        # Extracting the dimension of the truth tensor
        truth_dims = (5 + 3*nkpt + nc, S, S)
    else:
        truth_dims = (5 + 2*nkpt + nc, S, S)

    # Creating a tensors of zeros with the expected dimensions
    truth_tensor = torch.zeros(truth_dims)

    labels = split_label_tensor(label)

    num_det = len(labels)

    for l in labels:
        # import pdb; pdb.set_trace()
 
        x = l[...,1:2]
        y = l[...,2:3]
        w = l[...,3:4]
        h = l[...,4:5]

        i = torch.floor(S * x).int()
        j = torch.floor(S * y).int()

        if cell_relative:
            x = x * S - i
            y = y * S - j

        keypoints = l[..., 5:]

        class_tensor = torch.zeros(1, nc)
        label_class = l[...,0].int()
        # one-hot encoding of label_class
        class_tensor[...,label_class] = 1

        # Objectness --> relates to the IOU
        obj_conf = torch.ones(1, 1)

        # keypoint visibility conf
        if require_kpt_conf:

            kpt_conf = torch.ones(1, nkpt)
    
            truth_label = torch.cat((obj_conf, x, y, w, h, kpt_conf, keypoints, class_tensor), dim=1)

        else:
            truth_label = torch.cat((obj_conf, x, y, w, h, keypoints, class_tensor), dim=1)

        try:
            assert(truth_label.shape[1] == truth_dims[0])
        except Exception as e:
            raise

        truth_label = truth_label.reshape(truth_dims[0], 1, 1)

        truth_tensor[:, i, j] = truth_label
    
    return truth_tensor

def polar_kpt(x, y, w, h, kx, ky):
    """Returns relative position of keypoints.

    Parameters
    ----------
    x: torch.Tensor
        x coordinate of bounding box.
    y: torch.Tensor
        y coordinate of bounding box.
    w: torch.Tensor
        Width of the bounding box.
    h: torch.Tensor
        Height of bounding box.
    kx: torch.Tensor
        x coordinate of keypoint.
    ky: torch.Tensor
        y coordinate of keypoint.

    Returns
    -------
    tuple
        A tuple of ``(r, alpha)`` which are polar coordiantes relative to center of bounding box ``(x, y)``.

    Examples
    --------
    >>> polar_kpt(torch.Tensor([0.6]), torch.Tensor([0.6]), torch.Tensor([0.4]), torch.Tensor([0.4]), torch.Tensor([0.2]), torch.Tensor([0.2]))
    (tensor([0.5657]), tensor([0.6250]))
    
    """
    
    kx_r = kx - x
    ky_r = ky - y

    # polar 
    r = torch.sqrt(kx_r**2 + ky_r**2)
    theta = torch.atan2(ky_r, kx_r)

    alpha = torch.where(theta < 0, theta + 2 *np.pi, theta)
    
    # normalising
    alpha = alpha / (2 * np.pi)

    return r, alpha

def truth_head(truth, S, nc, nkpt, require_kpt_conf=True, require_polar_kpt=True):
    """Returns the head for the network prediction
    with the values organised in a dictionary.

    Parameters
    ----------
    truth: torch.Tensor
        A ground truth tensor of size ``(N, S, S)`` where ``N`` is the channel.
        ``N`` depends upon whether the keypoint confidence is considered.

        .. math::
            N = 5 + nkpt + 2 \\times nkpt + nc \\\\
            N = 5 + 2 \\times nkpt + nc
    S: int
        The grid size.
    nc: int
        Number of classes
    nkpt: int
        Number of keypoints
    require_kpt_conf: bool, default ``True``
        Whether to include keypoints.

    Returns
    -------
    dict
        A dictionary with keys ``['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'kpt_polar', 'classes']``

    Examples
    --------
    >>> truth = torch.Tensor([[[0.0000, 0.0000, 0.0000],
         [0.0000, 1.0000, 0.0000],
         [0.0000, 0.0000, 1.0000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.5000, 0.0000],
         [0.0000, 0.0000, 0.1000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.5000, 0.0000],
         [0.0000, 0.0000, 0.4000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.5000, 0.0000],
         [0.0000, 0.0000, 0.2000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.5000, 0.0000],
         [0.0000, 0.0000, 0.1000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.5000, 0.0000],
         [0.0000, 0.0000, 0.2000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.6000, 0.0000],
         [0.0000, 0.0000, 0.3000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000]],
        [[0.0000, 0.0000, 0.0000],
         [0.0000, 1.0000, 0.0000],
         [0.0000, 0.0000, 0.0000]]])
    >>> truth_head(t, S=3, nc=2, nkpt=1, require_kpt_conf=True)
        
    """
    # Sanity check
    ch, _, _ = truth.shape
    try:
        if require_kpt_conf:
            assert(ch == (5 + nkpt + 2* nkpt + nc))
        else:
            assert(ch == (5 + 2 * nkpt + nc))
    except Exception as e:
        print(e)
        print('\033[91m' + "Dimension of truth tensor does not match with requirement.\nMake sure to check if keypoint confidence is required.\n")
        raise
        
    conf = truth[0:1, ...]
    x = truth[1:2, ...]
    y = truth[2:3, ...]
    w = truth[3:4, ...]
    h = truth[4:5, ...]
    classes = truth[-nc:,...]

    start = 5
    if require_kpt_conf:
        end = 5 + nkpt
        k_conf = truth[start:end,...]
        kpts = truth[end:end+(2*nkpt),...]
    else:
        end = 5 + 2*nkpt
        kpts = truth[start:end, ...]

    # Extracting keypoints
    kpt_dict = dict()
    kpt_polar_dict = dict()
    
    i = 0
    j = 0
    # Iterating over the number of keypoints (2 * nkpt)
    while i < (nkpt * 2):
        kx, ky = kpts[i:i+2,...]
        kpt_dict[f"kx_{j}"] = kx
        kpt_dict[f"ky_{j}"] = ky

        if require_polar_kpt:
            r, alpha = polar_kpt(x, y, w, h, kx, ky)
            kpt_polar_dict[f"r_{j}"] = r
            kpt_polar_dict[f"alpha_{j}"] = alpha
        
        i += 2
        j += 1

    k_conf_dict = dict()
    
    if require_kpt_conf:
        k = 0
        while k < nkpt:
            k_conf_dict[f"k_conf_{k}"] = k_conf[k:k+1,...]
            k += 1

    head = {"conf": conf,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "k_conf": k_conf_dict,
            "kpt": kpt_dict,
            "kpt_polar": kpt_polar_dict,
            "classes": classes
           }

    return head