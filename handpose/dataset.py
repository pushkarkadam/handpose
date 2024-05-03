import os

def read_labels(file_path):
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
    >>> labels = handpose.dataset.read_labels('label.txt')
    
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
        
    return labels

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
        
    return file_names