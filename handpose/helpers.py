import yaml
import torch

def load_variables(file_path):
    """Loads variables from a YAML file.
    
    Parameters
    ----------
    file_path: str
        The path where the ``.yaml`` file is stored.

    Returns
    -------
    dict

    Examples
    --------
    >>> variables = handpose.helpers.load_variables('config.yaml')

    """
    try:
        with open(file_path, 'r') as file:
            variables = yaml.safe_load(file)
    except Exception as e:
        print(e)
        raise

    return variables

def xywh_to_xyxy(x, y, w, h, minimum = 1e-6):
    """Converts bounding box from ``[x y w h]`` to [x_min, y_min, x_max, y_max]`` format.
    
    Parameters
    ----------
    x: torch.Tensor
        x coordinate.
    y: torch.Tensor
        y coordinate.
    w: torch.Tensor
        Width of the box.
    h: torch.Tensor
        Height of the box.
    minimum: float, default ``1e-6``
        A minimum value that gives the respective output the value if the value goes in negative.
        This also helps in selecting the maximum value which is ``1-1e-6`` that keeps the coordinates
        within the image boundary.

    Returns
    -------
    x_min: torch.Tensor
        Minimum x coordinate of the box.
    y_min: torch.Tensor
        Minimum y coordinate of the box.
    x_max: torch.Tensor
        Maximum x coordinate of the box.
    y_max: torch.Tensor
        Maximum y coordinate of the box.

    Examples
    --------
    >>> x = torch.ones((1, 1, 3, 3)) * 0.5
    >>> y = torch.ones((1, 1, 3, 3)) * 0.5
    >>> w = torch.ones((1, 1, 3, 3)) * 0.5
    >>> h = torch.ones((1, 1, 3, 3)) * 0.5
    >>> x_min, y_min, x_max, y_max = xywh_to_xyxy(x, y, w, h)
    
    """

    # Maximum value so that it does not go at the edge of the image.
    maximum = 1 - minimum
    
    x_min = torch.clamp(x - w / 2, minimum, maximum)
    y_min = torch.clamp(y - h / 2, minimum, maximum)
    x_max = torch.clamp(x + w / 2, minimum, maximum)
    y_max = torch.clamp(y + h / 2, minimum, maximum)

    return x_min, y_min, x_max, y_max

def best_box(head, iou_threshold=0.5):
    """Returns best boxes.

    The best boxes are determined by confidence tensor.
    The confidence tensor represents the IOU in the prediction.
    Finding the maximum value of all the boxes indicates that the
    object is present and how accurate is the prediction of the object.
    The function returns an output similar to the output of ground truth.
    The similarity of the output means that the rendering function can be used.

    Parameters
    ----------
    head: dict
        A dictionary with the annotations.
    iou_threshold: float, default ``0.5``
        Threshold that is used to obtain index of the grid
        where the confidence that represents IOU is greater than
        the given threshold value.

    Returns
    -------
    dict
        A dictionary similar to the ones used by the ground truth.

    Examples
    --------
    >>> m = 2
    >>> S = 3
    >>> B = 2
    >>> nkpt = 21
    >>> nkpt_dim = 3
    >>> nc = 2
    >>> require_kpt_conf = True
    >>> tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    >>> out_features = S * S * tensor_ch
    >>> torch.manual_seed(0)
    >>> pred = torch.sigmoid(torch.randn((m, out_features)))
    >>> head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)
    >>> best_head = best_box(head, 0.8)
    
    """
    # Declaring keys to iterate over the dict
    box_keys = ['x', 'y', 'w', 'h']

    # Acquiring shape of the tensor
    m, B, S, _ = head['x'].shape

    # Maximum confidence and its index in the box (B) direction (dim=1)
    max_conf, index = torch.max(head['conf'], dim=1)

    # Assigning the values from head input amd max_conf
    best_head = {'conf': max_conf.unsqueeze(1), 'classes': head['classes']}

    # Iterating over the bounding box
    for key in box_keys:
        temp = torch.zeros((m, 1, S, S))

        for i in range(m):
            for j in range(S):
                for k in range(S):
                    temp[i, 0, j, k] = head[key][i, int(index[i, j, k]), j, k]
        best_head[key] = temp

    # keypoints keys
    kpt_keys = ['k_conf', 'kpt']

    # Iterating over the keypoints dictionary
    for kpt_key in kpt_keys:
        best_head[kpt_key] = dict()
        kpt_coord_keys = list(head[kpt_key].keys())
        kpt_i = head[kpt_key]
        for kpt_coord_key in kpt_coord_keys:
            temp_kpt = torch.zeros((m, 1, S, S))
            kpt_coord = kpt_i[kpt_coord_key]

            for i in range(m):
                for j in range(S):
                    for k in range(S):
                        temp_kpt[i, 0, j, k] = kpt_coord[i, int(index[i, j, k]), j, k]
            best_head[kpt_key][kpt_coord_key] = temp_kpt

    # Finding object indices
    conf = best_head['conf']
    obj_indices = []
    
    for i in range(m):
        temp = conf[i].squeeze(0)
        # Computing the indices of the grid where the confidence is
        # greater than the iou_threshold.
        indices = (temp > iou_threshold).nonzero()
        obj_indices.append(indices)

    best_head['obj_indices'] = obj_indices

    return best_head

def extract_head(best_head, cell_relative=True):
    """Extracts head data based on the indices

    The return dictionary contains a list of values for each image.
    If the image does not detect, then the list will be empty.
    For an image with multiple detection ``x, y, w, h`` will have multiple values.
    For multiple detection ``kx, ky`` will be a list of list with each detection being
    21 keypoints.

    Parameters
    ----------
    best_head: dict
        A dictionary with the keys ``['conf', 'classes', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'obj_indices']``.
    cell_relative: bool, default ``True``
        A boolean that checks if the ground truth bounding box centers were cell relative.

    Returns
    -------
    dict

    Examples
    --------
    >>> m = 2
    >>> S = 3
    >>> B = 2
    >>> nkpt = 21
    >>> nkpt_dim = 3
    >>> nc = 2
    >>> require_kpt_conf = True
    >>> tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    >>> out_features = S * S * tensor_ch
    >>> torch.manual_seed(0)
    >>> pred = torch.sigmoid(torch.randn((m, out_features)))
    >>> head = handpose.dataset.network_head(pred, require_kpt_conf, S, B, nkpt, nc)
    >>> best_head = best_box(head, 0.8)
    >>> data = extract_head(best_head)
    
    """
    # Creating a ditionary
    data = {'conf_score': [],
            'class_idx': [],
            'x': [],
            'y': [],
            'w': [],
            'h': [],
            'kx': [],
            'ky': []
           }

    # Object indices
    obj_indices = best_head['obj_indices']

    # Grid size
    m, _, S, _ = best_head['conf'].shape

    # confidence
    conf = best_head['conf']

    # class index maximum
    class_conf, class_idx = torch.max(best_head['classes'], dim=1)

    class_idx = class_idx.unsqueeze(1)
    class_conf = class_conf.unsqueeze(1)

    # confidence score
    conf_score = best_head['conf'] * class_conf
    
    # Number of keypoints
    nkpt = int(len(best_head['kpt'].keys()) / 2)

    for i, indices in enumerate(obj_indices):
        # Empty list to store values
        image_conf_score = []
        image_class_idx = []
        image_x = []
        image_y = []
        image_w = []
        image_h = []
        image_kx = []
        image_ky = []
        
        # Checking if the list of indices is empty
        if not indices.tolist():
            data['conf_score'].append([])
            data['class_idx'].append([])
            data['x'].append([])
            data['y'].append([])
            data['w'].append([])
            data['h'].append([])
            data['kx'].append([])
            data['ky'].append([])
            # continuing to next iteration
            continue
        
        for c in indices.tolist():
            # Extracting the index
            x, y = c

            # Extracting conf_score
            image_conf_score.append(conf_score[i,0,x,y])

            # Bounding box
            if cell_relative:
                x_img = (best_head['x'][i, 0, x, y] + x) / S
                y_img = (best_head['y'][i, 0, x, y] + y) / S
                image_x.append(x_img)
                image_y.append(y_img)
            else:
                image_x.append(best_head['x'][i,0,x, y])
                image_y.append(best_head['y'][i,0,x, y])
            image_w.append(best_head['w'][i,0,x, y])
            image_h.append(best_head['h'][i,0,x, y])

            # Classes index
            image_class_idx.append(class_idx[i, 0, x, y])
    
            # Keypoints
            kpt = best_head['kpt']
            temp_kx_list = []
            temp_ky_list = []
            for k_idx in range(nkpt):
                temp_x = kpt[f'kx_{k_idx}'][i,0,x, y]
                temp_y = kpt[f'ky_{k_idx}'][i,0,x, y]
                temp_kx_list.append(temp_x)
                temp_ky_list.append(temp_y)
    
            image_kx.append(temp_kx_list)
            image_ky.append(temp_ky_list)

        data['conf_score'].append(image_conf_score)
        data['class_idx'].append(image_class_idx)
        data['x'].append(image_x)
        data['y'].append(image_y)
        data['w'].append(image_w)
        data['h'].append(image_h)
        data['kx'].append(image_kx)
        data['ky'].append(image_ky)

    return data

def detect(image, model, require_kpt_conf, S, B, nkpt, nc, iou_threshold):
    r"""Detects on the input image.

    Performs different operations such as head extraction from the model from the single image input.

    Parameters
    ----------
    image: torch.Tensor
        A torch tensor image of size `(1, 3, H, W)`
    model: 
        A CNN model
    require_kpt_conf: bool
        Checks whether the keypoint confidence is required.
    S: int
        The grid size.
    B: int
        The number of prediction boxes.
    nkpt: int
        The number of landmarks.
    nc: int
        Number of classes to predict.
    iou_threshold: float
        The IOU (Intersection Over Union) threshold.

    Returns
    -------
    head: dict
        Head of the prediction
    head_nms: dict
        Head dictionary after Non-max suppression.
        
    """

    EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    image = image.to(DEVICE)

    pred = model(image)

    head = network_head(pred,
                        require_kpt_conf=require_kpt_conf,
                        S=S,
                        B=B,
                        nkpt=nkpt,
                        nc=nc
                       )
    # activations
    head = head_activations(head)

    head = best_box(head, iou_threshold=iou_threshold)

    head = extract_head(head)

    head_nms = non_max_suppression(head)

    return head, head_nms

def create_train_dir(save_model_path):
    """Creates a directory for training results.

    Parameters
    ----------
    save_model_path: str
        Path to store weights.
        
    """
    dirs = os.listdir(save_model_path)

    if not dirs:
        os.makedirs(os.path.join(save_model_path, 'train1'))
    else:
        train_nums = []

        for d in dirs:
            train_nums.append(int(re.findall('\d+', d)[0]))
        
        train_nums.sort()

        last_train_num = train_nums.pop()

        os.makedirs(os.path.join(save_model_path, 'train' + str(last_train_num + 1)))