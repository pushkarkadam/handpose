import sys
import pytest 
sys.path.append('../')

import handpose

def test_read_labels():
    text_file_path = 'tests/test_labels/test_label1.txt'
    labels = handpose.dataset.read_labels(text_file_path)

    assert(len(labels) == 2)
    assert(len(labels[0]) == 4)
    assert(type(labels[0][0]) == float)

    text_file_path = 'no_file_exists.txt'
    with pytest.raises(Exception) as e:
        labels = handpose.dataset.read_labels(text_file_path2)

def test_list_files():
    test_directory = 'tests/test_labels'

    directories = handpose.dataset.list_files(test_directory)

    assert(len(directories) == 2)
    assert(type(directories[0]) == str)