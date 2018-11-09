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
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model,epoch,dist,runid):
    path = os.getcwd()
    outdir = os.path.join(path,str(runid))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    filename1 = outdir + '/e_'+str(epoch)+'_a_{0:.4f}'.format(dist)+'.model'
    net = model.cpu()
    torch.save(net,filename1)
    
    filename2 = outdir + '/e_'+str(epoch)+'_a_{0:.4f}'.format(dist)+'.dict'
    torch.save(net.state_dict(), filename2)
    
    print('\t\t\t\t\t\t\tSaving model to: {0} and dict to: {1}'.format(filename1,filename2))

    model.to(DEVICE)

    
def train(params):
    
    writer = SummaryWriter(log_dir='runs/'+params.runid)

    if params.dev:
        mode = 'validation'
    else:
        mode = 'training'
        
    train_loader = DataLoader(ShapesLoader(mode=mode,dataset_path='./datasets/synthetic_shapes_v6/'),
                              batch_size=params.batch,shuffle=True,collate_fn=collate_fn)
    
    model = SuperPointNet()
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=params.weightdecay)
    
    criterion = nn.CrossEntropyLoss() #reduction='elementwise_sum')
        
    for e in range(params.epoch):
        
        epoch_loss = 0
        img_count = 0
        
        for batch_idx, (imgs,pix_locs,didx) in enumerate(train_loader):

            ipt,desc = model(imgs.float().unsqueeze(1).to(DEVICE))
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
            loss = criterion(ipt.to(DEVICE),pts.long().to(DEVICE))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            
            print('\x1b[2K\rEpoch {0} Loss:{1:.3f} Batch idx:{2}'.format(e+1,loss.item(),batch_idx),end='\r')
            img_count += bnum

        print('\x1b[2K\rEnd of epoch {0} Loss:{1:.5f}'.format(e+1,epoch_loss))
        
        save_model(model,e,epoch_loss,params.runid)
        #probs = softmax(ipt)

        #probs bnum x 64 x hc x wc
        #probs = probs[:,:-1,:,:]
        
        #probs bnum x h x w
        #probs = probs.reshape(bnum,h,w)
        
def validate(params):
    
    writer = SummaryWriter(log_dir='runs/'+params.runid)

    val_loader = DataLoader(ShapesLoader(mode='validation',dataset_path='./datasets/synthetic_shapes_v6/'),
                              batch_size=params.batch,collate_fn=collate_fn)
    
    if params.model is None:
        print('Need a model to run the validation')

    
    else:
        model = torch.load(params.model)
        pdb.set_trace()
        model = model.to(DEVICE)
        
    model.eval()
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=params.weightdecay)
    
    criterion = nn.Softmax(dim=1) #reduction='elementwise_sum')
        
    threshold = 0.02

    for batch_idx, (imgs,pix_locs,didx) in enumerate(val_loader):

        ipt,desc = model(imgs.float().unsqueeze(1).to(DEVICE))
        #ipt bnum x 65 x hc x wc
        bnum, dims, hc, wc = ipt.shape

        ipt_sm = criterion(ipt)
        #ignore the dustbin entry
        ipt_sm = ipt_sm[:,:-1,:,:]
        
        #find the max entry and confidence
        idx_conf,idx_locs = ipt_sm.max(dim=1)
        
        idx_mask = idx_conf > threshold
        #convert this to pixel location
        #for each image in the batch
        for b in range(bnum):
            #img = Image.fromarray(data, 'RGB')
            print('Image:',didx[b])
            print('Ground truth pixels:')
            print(pix_locs[b])
            
            print('Estimated pixels:')
            for x in range(hc):
                for y in range(wc):
                    if idx_mask[b,x,y] == 1:
                        #location within the block
                        block_x = idx_locs[b,x,y]/8
                        block_y = idx_locs[b,x,y]%8
                        
                        #location in the image
                        px = x * block_x
                        py = y * block_y
                        print('x:{0} y:{1}'.format(px,py))
                        

    
def main():
    parser = argparse.ArgumentParser(description='Specialized Superpoint')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--dev',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--epoch',default=1000,type=int)
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
    