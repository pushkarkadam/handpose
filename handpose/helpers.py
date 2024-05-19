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