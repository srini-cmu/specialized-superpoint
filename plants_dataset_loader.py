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
    
    def __init__(self,dataset_path,plant_imgs,downsample_percent=1.):

        self.dataset_path = dataset_path
        self.plant_imgs = plant_imgs
        self.dwn_smpl = downsample_percent
        
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
        img = Image.open(filename).resize((int(476*self.dwn_smpl),int(720*self.dwn_smpl)),Image.ANTIALIAS).convert('L') #load image as grayscale
        original = np.array(img)
        warped,invH = self.warp_img(img)
        return original,warped,invH,filename
                                
    
def plant_dataset_collate_fn(batch_list):
    imgs,warps,homographies,fnames = zip(*batch_list)
    imgs = torch.cat([torch.tensor(img,dtype=torch.float).unsqueeze(0) for img in imgs])
    warps = torch.cat([torch.tensor(warp,dtype=torch.float).unsqueeze(0) for warp in warps])
    homographies = torch.cat([torch.tensor(h) for h in homographies])
    return imgs,warps,homographies,fnames
