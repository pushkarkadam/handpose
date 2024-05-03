import sys
import pytest 
sys.path.append('../')

import handpose

def test_read_labels():
    text_file_path = 'tests/test_label.txt'
    labels = handpose.dataset.read_labels(text_file_path)

    assert(len(labels) == 2)
    assert(len(labels[0]) == 4)
    assert(type(labels[0][0]) == float)

    text_file_path = 'no_file_exists.txt'
    with pytest.raises(Exception) as e:
        labels = handpose.dataset.read_labels(text_file_path2)