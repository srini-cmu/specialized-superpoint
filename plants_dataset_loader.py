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

from homography import *

class PlantDatasetLoader(torch.utils.data.Dataset):
    
    def __init__(self,dataset_path,plant_imgs,downsample_percent=1.,homographies=None):

        self.dataset_path = dataset_path
        self.plant_imgs = plant_imgs
        self.dwn_smpl = downsample_percent
        self.homographies = homographies
        
        self.dim1 = int(1600 * self.dwn_smpl) #476
        self.dim2 = int(1200 * self.dwn_smpl) #720
        print('Setting image resolution to {}x{}'.format(self.dim1,self.dim2))
        
    def __len__(self):
        return len(self.plant_imgs)
    
    def warp_img(self,img):
        Htransform,_ = sample_homography(torch.Tensor(np.array(img).shape))
        invH = invert_homography(Htransform)
        warped = img.transform(size=img.size, method=Image.PERSPECTIVE, data=invH[0].numpy(), resample=Image.BILINEAR)
        warped = np.array(warped)
        return warped,invH
    
    def __getitem__(self,idx):

        filename = self.dataset_path + self.plant_imgs[idx]
        img = Image.open(filename).resize((self.dim1,self.dim2),Image.ANTIALIAS).convert('L') #load image as grayscale
        original = np.array(img)
        
        if self.homographies is not None:
            #generate homography and warp image
            warped,invH = self.warp_img(img)
            #load the homography adaptation points
            h = np.load(self.dataset_path + self.homographies[idx])
            return original,warped,invH,h,filename
        else:
            return original,filename
                                
    
def plant_dataset_collate_fn(batch_list):
    imgs,warps,homographies,homo_adapt,fnames = zip(*batch_list)
    imgs = torch.cat([torch.tensor(img,dtype=torch.float).unsqueeze(0) for img in imgs])
    warps = torch.cat([torch.tensor(warp,dtype=torch.float).unsqueeze(0) for warp in warps])
    homographies = torch.cat([torch.tensor(h) for h in homographies])
    return imgs,warps,homographies,homo_adapt,fnames

def homography_collate_fn(batch_list):
    imgs,fnames = zip(*batch_list)
    imgs = torch.cat([torch.tensor(img,dtype=torch.float).unsqueeze(0) for img in imgs])
    return imgs,fnames
