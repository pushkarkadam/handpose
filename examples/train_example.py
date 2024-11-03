import torch 
import sys 

sys.path.append('../')
from handpose import *


if (len(sys.argv) != 2):
    print("Usage: python train_example.py ~/path/to/config.yaml")
    sys.exit(1)

config_file = sys.argv[1]

start_training(config_file)