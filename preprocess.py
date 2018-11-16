import cv2
import numpy as np
import sys
import os
import os.path
import argparse
import pdb
import time
from matplotlib import pyplot as plt


class Histogram:
    
    def __init__(self,params):
        
        if params.chist:
            self.contrast_limiting = True
            self.hist = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        else:
            self.contrast_limiting = False
        
        
    def equalize(self,img):
        
        if params.chist:
            return self.hist.apply(img)
        else:
            return cv2.equalizeHist(img)
        


def load_image_paths(path):
    
    files = []
    for f in os.listdir(path):
        
        if '.png' in f and '_preprocessed' not in f:
            files.append((path,f))
    
        elif os.path.isdir(os.path.join(path,f)):
            files.extend(load_image_paths(os.path.join(path,f)))
            
    print('Read {0} images from dir {1}'.format(len(files),path))
    return files
                         

    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Preprocessor for the plant dataset')
    parser.add_argument('--chist',action='store_true',help='Performs contrast limiting histogram equalization')
    parser.add_argument('--imgpath',help='Location to the images directory')
    parser.add_argument('--outdir',default='./')
    
    params = parser.parse_args()

    if params.imgpath is None:
        print('Need the location of the images to do any processing!')
        sys.exit(-1)
        

    runid = str(time.time()) + '_preprocessed'
    #create out directory 
    outdir = os.path.join(params.outdir,runid)
    os.mkdir(outdir)
    print('Writing to directory:{}'.format(outdir))
          
    img_paths = load_image_paths(params.imgpath)
    
    hist = Histogram(params)
    
    counter = 0
    #equalize histogram
    for img_f in img_paths:
    
        counter += 1
        print('\x1b[2K\rHistogram equalization:{0:40s} {1:0.2f}%'.format(img_f[1],counter*100./len(img_paths)),end='\r')
        img = cv2.imread(os.path.join(img_f[0],img_f[1]),0)
        
        hist_img = hist.equalize(img)
    
        
        cv2.imwrite(os.path.join(outdir,img_f[1]),hist_img)
        
    print('\nFinished processing all the images')