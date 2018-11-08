import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
import pdb
import sys
    

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc

'''
        
class VGGBlock(nn.Module):
    
    def __init__(self,inch,outch):
        super(VGGBlock,self).__init__()
        
        self.conv = nn.Conv2d(inch, outch, 3, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(outch)
        self.relu = nn.ReLU()

    def forward(self,inp):
        inp = self.conv(inp)
        inp = self.bn(inp)
        inp = self.relu(inp)

        return inp
    
class VGGBackBone(nn.Module):
    
    def __init(self,in_size):
        super(BaseModel,self).__init__()
        
        self.vgg1 = VGGBlock(in_size,64)
        self.pool1 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        self.vgg2 = VGGBlock(64,64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.vgg3 = VGGBlock(64,128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.vgg4 = VGGBlock(128,128)
        self.vgg5 = VGGBlock(128,128)
        
    def forward(self,inp):
        
        inp = self.vgg1(inp)
        inp = self.pool1(inp)
        
        inp = self.vgg2(inp)
        inp = self.pool2(inp)
        
        inp = self.vgg3(inp)
        inp = self.pool3(inp)
        
        inp = self.vgg4(inp)
        inp = self.vgg5(inp)
        
        return inp
    
    
class BaseModel(nn.Module):
    
    def __init__(self):
        
 
'''
