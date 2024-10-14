import torch 
import sys 

sys.path.append('../')
from handpose import *


def main() -> int:
    train_network('config.yaml')

if __name__ == '__main__':
    sys.exit(main())