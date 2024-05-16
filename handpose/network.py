import torch
from torchsummary import summary


class TransferNetwork(torch.nn.Module):
    def __init__(self, repo_or_dir, model_name, pretrained, S, B, nkpt, nc, input_size, require_kpt_conf):
        """Transfer learning network.

        Arguments
        ---------
        repo_or_dir: str
            Repository where the model is stored.
        model_name: str
            A model from a list of models from pytorch.
        pretrained: bool
            Use pre-trained weights.
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
        self.pretrained = pretrained
        
        self.model = torch.hub.load(repo_or_dir, model_name, pretrained=pretrained)

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