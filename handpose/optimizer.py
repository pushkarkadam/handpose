import torch
import warnings


def optimizer_fn(model, optimizer, **kwargs):
    r"""Returns an optimizer.

    Select optimizer from different algorithms in `Optimizer`_.

    .. _Optimizer: https://pytorch.org/docs/stable/optim.html#algorithms

    .. deprecated::
        You can now use
        :func:`handpose.metric.Optimizer`
        for creating an optimizer.

    Parameters
    ----------
    model: torch.nn.Module
        A CNN model.
    optimizer: torch.optim
        An optimizer.
    **kwargs
        Hyper parameters for optimizer.

    Returns
    -------
    torch.optim

    Examples
    --------
    >>> optimizer_fn(model, optim.SGD, **{'lr':0.01, 'momentum':0.9})
    
    """
    warnings.warn(
        r"`optimizer_fn()` is deprecated and will be removed in future version. Please use the `handpose.optimizer.Optimizer` class instead.",
        DeprecationWarning
    )

    # Using parameters whose weights are not frozen
    params = [param for param in model.parameters() if param.requires_grad]

    opt = optimizer(params, **kwargs)

    return opt

class Optimizer(torch.nn.Module):
    """
    A wrapper class for optimizers to be used with a PyTorch model.

    This class simplifies the creation of an optimizer by automatically
    filtering out model parameters that do not require gradients.

    Select optimizer from different algorithms in `Optimizer`_.

    .. _Optimizer: https://pytorch.org/docs/stable/optim.html#algorithms


    Parameters
    ----------
    model : torch.nn.Module
        The neural network model whose parameters will be optimized.
    optimizer_cls : type
        The optimizer class from `torch.optim` (e.g., `torch.optim.SGD`).
    **kwargs : dict
        Additional keyword arguments to be passed to the optimizer.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model to be optimized.
    optimizer : torch.optim.Optimizer
        The optimizer instance that will be used to update the model's parameters.

    Methods
    -------
    __init__(model, optimizer_cls, **kwargs)
        Initializes the Optimizer instance with a given model and optimizer class.

    Examples
    --------
    >>> model = MyModel()
    >>> optimizer = Optimizer(model=model, optimizer_cls=torch.optim.SGD, lr=0.01, momentum=0.9)
    >>> optimizer.optimizer.step()  # Perform an optimization step
    """
    
    def __init__(self, model, optimizer_cls, **kwargs):
        """
        Initializes the Optimizer instance.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be optimized.
        optimizer_cls : type
            The optimizer class from `torch.optim` (e.g., `torch.optim.SGD`).
        **kwargs : dict
            Additional keyword arguments to be passed to the optimizer.

        Examples
        --------
        >>> model = MyModel()
        >>> optimizer = Optimizer(model=model, optimizer_cls=torch.optim.SGD, lr=0.01, momentum=0.9)
        >>> optimizer.optimizer.zero_grad()
        """
        super().__init__()
        self.model = model

        # Filter parameters that require gradients
        params = filter(lambda p: p.requires_grad, model.parameters())
        
        # Initialize the optimizer with the filtered parameters
        self.optimizer = optimizer_cls(params, **kwargs)

class Scheduler(torch.nn.Module):
    """A wrapper class for scheduler.
    
    This class simplifies the creation of a scheduler while giving the flexibility
    of using different types of scheduler mentioned in torch `Scheduler`_.

    .. _Scheduler: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Parameters
    ----------
    optimizer: torch.optim
        Optimizer class.
    lr_scheduler: torch.optim
        A scheduler type.
    **kwargs: dict
        Additional keywork arguments to be passed to the scheduler.

    Attributes
    ----------
    optimizer: torch.optim
        Optimizer class.
    lr_scheduler: torch.optim
        A scheduler type.
    
    Examples
    --------
    >>> model = Model()
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = handpose.optimizer.Scheduler(optimizer, lr_scheduler.MultiStepLR, **{'milestones': [30, 80], 'gamma':0.1}).scheduler
    
    """
    def __init__(self, optimizer, lr_scheduler, **kwargs):
        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.scheduler = self.lr_scheduler(self.optimizer, **kwargs)