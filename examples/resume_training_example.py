import torch 
import sys 

sys.path.append('../')
from handpose import *

if (len(sys.argv) != 3):
    print("Usage: python resume_trainin_example.py ~/path/to/train_dir ~/path/to/config.yaml")
    sys.exit(1)
    
train_path = sys.argv[1]
config_file = sys.argv[2]

resume_training(train_path, config_file)
