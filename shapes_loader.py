import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import time
import os.path
import pdb
import argparse
import sys
import os
from PIL import Image

class ShapesLoader(torch.utils.data.Dataset):
    
    def __init__(self,mode='training',dataset_path='./datasets/synthetic_shapes_v6/'):

        if 'training' not in mode and 'test' not in mode and 'validation' not in mode:
            print('Invalid mode. Allowed values: training, test, validation')
            sys.exit(-1)
        
        self.mode = mode
        self.dataset_path = dataset_path
        print('Looking into dataset dir for '+mode+' data')
        
        tbegin = time.time()
        
        self.shape_dirs = os.listdir(dataset_path)

        self.imgs = []
        self.pts = []
        self.dset = []
        
        for shape in self.shape_dirs:
            print('Loading images and points from '+shape)
            pt_path = self.dataset_path+'/'+shape+'/points/'+self.mode+'/'
            shape_path = self.dataset_path+'/'+shape+'/images/'+self.mode+'/'
            num_files = len(os.listdir(shape_path))
                            
            self.imgs += [np.array(Image.open(shape_path+str(idx)+'.png')) for idx in range(num_files)]
            self.pts += [np.array(np.load(pt_path+str(idx)+'.npy')) for idx in range(num_files)]
            self.dset += [shape+'_'+str(idx) for idx in range(num_files)]
        tend = time.time()
        
        print('Finished loading the synthetic shapes dataset. Took:{0:.3f} s'.format(tend-tbegin))   
      
        
        
    def __len__(self):
        return len(self.pts)
    
    def __getitem__(self,idx):
        #print('Idx:{0} Img:{1}x{2} Pts:{3}'.format(idx,self.imgs[idx].shape[0],self.imgs[idx].shape[1],len(self.pts[idx])))
        return self.imgs[idx],self.pts[idx],self.dset[idx]
                            
    def get_img(self,shape_idx,idx):    
        img = self.dataset_path+'/'+self.shape_dirs[shape_idx]+'/images/'+self.mode+'/'+str(idx)+'.png'
        return np.array(Image.open(img))

    def get_pts(self,shape_idx,idx):    
        pt = self.dataset_path+'/'+self.shape_dirs[shape_idx]+'/points/'+self.mode+'/'+str(idx)+'.npy'
        return np.load(pt)
    
    
def collate_fn(batch_list):
    imgs,pixloc,didx = zip(*batch_list)
    imgs = torch.tensor(np.asarray([img for img in imgs]),dtype=torch.float)
    return imgs,pixloc,didx
