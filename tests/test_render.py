import sys
import pytest 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append('../')

import handpose 

def test_render_pose():
    img_dir = 'data/dev/train/images'
    label_dir = 'data/dev/train/labels'

    training_data = handpose.dataset.HandDataset(img_dir, label_dir, S=7, nc=2, nkpt=21, cell_relative=True, require_kpt_conf=True)
    torch.manual_seed(0)
    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=False, drop_last=True)

    train_features, data = next(iter(train_dataloader))

    img_name = data['image_name']
    head = data['head']

    rendered_images = handpose.render.render_pose(train_features, head, is_relative=True, show_keypoint_label=True)

    assert(type(rendered_images) == list)
    assert(rendered_images[0].shape == (224, 224, 3))