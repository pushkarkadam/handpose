import os
import torch

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
    of size (N, n) where ``N`` is the number of detected objects and ``n`` is the annotation.
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