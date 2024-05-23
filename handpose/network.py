import torch
from torchsummary import summary


class TransferNetwork(torch.nn.Module):
    def __init__(self, 
                 repo_or_dir, 
                 model_name, 
                 weights, 
                 S, 
                 B, 
                 nkpt, 
                 nc, 
                 input_size, 
                 require_kpt_conf, 
                 freeze_weights):
        """Transfer learning network.

        Arguments
        ---------
        repo_or_dir: str
            Repository where the model is stored.
            Example: ``'pytorch/vision:v0.17.1'``
        model_name: str
            A model from a list of models from pytorch.
        weights: str
            Pre-trained weights.
        S: int
            The grid size.
        B: int
            The number of prediction boxes.
        nkpt: int
            The number of landmarks.
        nc: int
            Number of classes to predict
        input_size: tuple
            The input size of the image.
        require_kpt_conf: bool
            Use keypoint confidence.
        model: torch.nn.Module
            A CNN model
        freeze_weights: bool
            Freezes weights of the loaded CNN.

        Methods
        -------
        summary()
            Summarizes the network using ``torchsummary.summary``.
        load_saved(network_path)
            Loads a sived network.
        
        """
    
        super(TransferNetwork, self).__init__()
        self.repo_or_dir = repo_or_dir
        self.model_name = model_name
        self.S = S
        self.B = B
        self.nkpt = nkpt
        self.nc = nc
        self.input_size = input_size
        self.require_kpt_conf = require_kpt_conf
        self.weights = weights
        self.freeze_weights = freeze_weights
        
        self.model = torch.hub.load(repo_or_dir, model_name, weights=weights)

        # Freezes weights for transfer learning
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        module_names = [name for name, _ in self.model.named_children()]

        # Find the name of the last layer (usually it's 'fc' for many models)
        last_layer_name = module_names[-1]

        nkpt_dim = 2
    
        # If keypoint conf is used, then (k_conf, kx, ky) --> 3 dims are used otherwise (kx, ky) --> 2 dims are used
        if self.require_kpt_conf:
            nkpt_dim = 3
    
        # Output features
        out_features = S * S * (B * (5 + nkpt_dim * nkpt) + nc)

        if last_layer_name == 'classifier':
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(in_features=num_features, out_features=out_features)

        elif last_layer_name == 'fc':
            num_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features=num_features, out_features=out_features)
        else:
            print('\033[91m' + "Network not suitable. Currently supports: ResNet, AlexNet, or VGG.")
            raise
            
    def forward(self, x):
        return self.model(x)

    def summary(self):
        """Returns the summary of the network."""
        return summary(self.model, input_size=self.input_size)

    def load_saved(self, network_path):
        """Loads the saved network.
        
        Parameters
        ----------
        network_path: str
            Network path and name. e.g. ``'~/path/to/data/network.pt'``
        
        """

        try:
            self.model.load_state_dict(torch.load(network_path))
        except Exception as e:
            print(e)
            raise

    def save(self, network_path):
        """Saves the network.
        
        Parameters
        ----------
        network_path: str
            Network path and name. e.g. ``'~/path/to/data/network.pt'``
        
        """
        torch.save(self.model.state_dict(), network_path)

def network_head(pred, 
                 require_kpt_conf,
                 S,
                 B,
                 nkpt,
                 nc
                ):
    """Returns the head of the network.

    For bounding box ``x y w h``, each of these tensor is of size ``(m, B, S, S)``.
    All the boxes values are stored in their respective type of annotation.
    
    Parameters
    ----------
    pred: torch.Tensor 
        A tensor of size ``(m, N)`` 
        where ``m`` is the number of samples
        and ``N`` is the number of features.

        .. math::
            N = S \\times S \\times (B \\times (5 + nkpt_{dim} \\times nkpt) + nc) \\\\
    require_kpt_conf: bool
        Use keypoint confidence.
    S: int
        The grid size.
    B: int
        The number of prediction boxes.
    nkpt: int
        The number of landmarks.
    nc: int
        Number of classes to predict
        
    Returns
    -------
    dict
        A dictionary with keys ``['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'classes']``
    
    Examples
    --------
    >>> pred = torch.randn((2, 7 * 7 * (2 * (5 + 3 * 21) + 2)))
    >>> head = network_head(pred=pred, require_kpt_conf=True, S=7, B=2, nkpt=21, nc=2)
    
    """

    # Selecting the keypoint tensor dimension based on whether keypoint confidence are considered
    nkpt_dim = 2

    if require_kpt_conf:
        nkpt_dim = 3

    try:
        m, _ = pred.shape
    except Exception as e:
        print(e)
        raise

    tensor_ch = B * (5 + nkpt_dim * nkpt) + nc
    
    # Reshaping
    tensor = pred.reshape((m, tensor_ch, S, S))
    
    start = 0
    end = start+B
    conf = tensor[:,start:end, ...]

    start = end
    end = start+B
    x = tensor[:, start:end, ...]

    start = end
    end = start + B
    y = tensor[:, start:end, ...]

    start = end
    end = start + B
    w = tensor[:, start:end, ...]

    start = end
    end = start + B
    h = tensor[:, start:end, ...]
    
    start = end
    end = start + nc
    class_scores = tensor[:, start:end,... ]

    start = end
    kpt_tensor = tensor[:, start:, ...]

    kpt_conf_dict = dict()
    kpt_dict = dict()

    i = 0
    j = 0
    while i < (nkpt * 2):
        start = i
        end = i + B
        kx = kpt_tensor[:,start:end, ...]
        ky = kpt_tensor[:,end:end+B, ...]

        kpt_dict[f'kx_{j}'] = kx
        kpt_dict[f'ky_{j}'] = ky

        i += 2
        j += 1
        

    if require_kpt_conf:
        kpt_conf_tensor = kpt_tensor[:,-nkpt*B:,...]
        k = 0
        l = 0
        while k < (nkpt * B):
            kpt_conf_dict[f'k_conf_{l}'] = kpt_conf_tensor[:,k:k+B,...]
            k += B
            l += 1

    head = {'conf': conf,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'k_conf': kpt_conf_dict,
            'kpt': kpt_dict,
            'classes': class_scores
           }
    
    return head

def head_activations(head):
    """Returns the same dictionary with the tensors through activation functions.

    Parameters
    ----------
    head: dict
        A dictionary of keys ``['conf', 'x', 'y', 'w', 'h', 'k_conf', 'kpt', 'classes']``

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
    >>> pred = torch.randn((m, out_features))
    >>> head = handpose.network.network_head(pred, require_kpt_conf, S, B, nkpt, nc)
    >>> head_act = head_activations(head)
    
    """

    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=1)
    
    conf = sigmoid(head['conf'])
    x = sigmoid(head['x'])
    y = sigmoid(head['y'])
    w = sigmoid(head['w'])
    h = sigmoid(head['h'])

    k_conf = dict()
    for k, v in head['k_conf'].items():
        k_conf[k] = sigmoid(v)

    kpt = dict()
    for k, v in head['kpt'].items():
        kpt[k] = sigmoid(v)

    classes = softmax(head['classes'])

    head_act = {'conf': conf,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'k_conf': k_conf,
                'kpt': kpt,
                'classes': classes
               }
    return head_act