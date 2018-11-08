import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import tqdm
import time
import os.path
import pdb
import argparse
import sys
import os
from shapes_loader import *
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import *

def train(params):
    
    writer = SummaryWriter(log_dir='runs/'+params.runid)

    train_loader = DataLoader(ShapesLoader(mode='training',dataset_path='./datasets/synthetic_shapes_v6/'),batch_size=params.batch,shuffle=True)
def main():
    parser = argparse.ArgumentParser(description='Specialized Superpoint')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--batch',default=64,type=int)
    
    params = parser.parse_args()

    params.runid = time.asctime(time.localtime())
    
    if params.train:
        train(params)
        
    if params.validate:
        validate(params)
        
    if params.test:
        test(params)

        
if __name__ == '__main__':    
    main()
    