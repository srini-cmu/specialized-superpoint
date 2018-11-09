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
from base_model import *
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import *

def train(params):
    
    writer = SummaryWriter(log_dir='runs/'+params.runid)

    if params.dev:
        mode = 'validation'
    else:
        mode = 'training'
        
    train_loader = DataLoader(ShapesLoader(mode=mode,dataset_path='./datasets/synthetic_shapes_v6/'),
                              batch_size=params.batch,shuffle=True,collate_fn=collate_fn)
    
    model = SuperPointNet()
    optimizer = torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=params.weightdecay)
    criterion = nn.CrossEntropyLoss() #reduction='elementwise_sum')
    
    for batch_idx, (imgs,pix_locs,didx) in enumerate(train_loader):
        
        #bnum = imgs.shape[0]
        #h,w = imgs[0].shape

        ipt,desc = model(imgs.float().unsqueeze(1))
        #ipt bnum x 65 x hc x wc
        bnum, dims, hc, wc = ipt.shape
        
        #for pixels in each image, divide by eight to find which sector it belongs to
        #modulo by 8 to find the value in each sector
        dim_locs = [(torch.tensor(pix_locs[i])/8).int() for i in range(len(pix_locs))]
        dim_val = [(torch.tensor(pix_locs[i])%8).int() for i in range(len(pix_locs))]
        
        #create the encoded vector
        pts = torch.ones(bnum,hc,wc)*64 #initialize to dustbin values which will be cleared if there is a true value
            
        #TODO: Make this faster
        #for each image in the batch
        for b in range(len(dim_locs)):
            
            #for each pixel location
            for i in range(len(dim_locs[b])):
                pts[b,dim_locs[b][i][0],dim_locs[b][i][1]] = 8*dim_val[b][i][0] +  dim_val[b][i][1]
        
        pts = pts.long()
        #pdb.set_trace()

        #probs bnum x 65 x hc x w
        loss = criterion(ipt,pts)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('Loss:{0:.3f}'.format(loss.item()))
              
        #probs = softmax(ipt)

        #probs bnum x 64 x hc x wc
        #probs = probs[:,:-1,:,:]
        
        #probs bnum x h x w
        #probs = probs.reshape(bnum,h,w)
        
        
        

    
def main():
    parser = argparse.ArgumentParser(description='Specialized Superpoint')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--dev',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--weightdecay',default=1e-6,type=float)
    parser.add_argument('--lr',default=1e-3,type=float)
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
    