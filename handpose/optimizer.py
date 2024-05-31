def optimizer_fn(model, optimizer, **kwargs):
    """Returns an optimizer.

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

    # Using parameters whose weights are not frozen
    params = [param for param in model.parameters() if param.requires_grad]

    opt = optimizer(params, **kwargs)

    return opt