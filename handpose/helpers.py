import yaml

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

    return variables