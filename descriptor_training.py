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
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import *
from PIL import Image
import matplotlib.pyplot as plt

from plants_dataset_loader import *
from base_model import *
from homography import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def df(x):
    print(x.shape,end='')
    print(' Diff:',x.requires_grad,end='')
    print(' Mean:',x.mean().item(),end='')
    print(' Max:',x.max().item(),end='')
    print(' Min:',x.min().item())
    

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

def log_settings(params):
    
    writer = SummaryWriter(log_dir='runs/'+str(params.runid))

    params_str = ''
    for key,value in vars(params).items():
        params_str = params_str + str(key)+':'+str(value)+'\n'

    writer.add_text('Initial Settings',params_str)
    
    #params.writer = writer
    

mp = 1
mn = 0.2
ld = 250
# should use the raw descriptor output from the network
def descriptor_loss(desc, warped_desc, homographies):
    batch_size = desc.shape[0]
    Hc, Wc = desc.shape[-2:]
    
    p_hw = torch.stack(torch.meshgrid((torch.arange(Hc), torch.arange(Wc))), dim=-1).float()
    p_hw = p_hw * 8 + 8 // 2
    warped_p_hw, bound, mask = warp_point(homographies, p_hw)
    p_hw = p_hw.view(1,Hc,Wc,1,1,2).float()
    warped_p_hw = warped_p_hw.view(batch_size,1,1,Hc,Wc,2).float()
    s = torch.le(torch.norm(p_hw-warped_p_hw,p=2,dim=-1), 8).float().to(DEVICE)

    desc = desc.view((batch_size,Hc,Wc,1,1,-1))
    warped_desc = warped_desc.view((batch_size,1,1,Hc,Wc,-1))
    dot_prod = torch.sum(desc*warped_desc, dim=-1)

    loss = ld * s * torch.clamp(mp - dot_prod, min=0.) + (1-s) * torch.clamp(dot_prod - mn, min=0.)

    mask_tensor = torch.Tensor(mask / 255).to(DEVICE)
    mask_split = mask_tensor.split(8, 2) # dim 2
    mask_stack = [st.reshape(mask_tensor.shape[0],Hc,1,8*8) for st in mask_split]
    mask_out = torch.cat(mask_stack,2)
    
    mask_out = torch.cumprod(mask_out, dim=3)[:,:,:,-1]
    mask_out = mask_out.reshape(batch_size, 1, 1, Hc, Wc)
    
    normalization = torch.sum(mask_out) * (Hc * Wc)
    loss = torch.sum(loss * mask_out) / normalization
    return loss


def feature_point_loss(ipt, pix_locs):

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
            if dim_locs[b][i][0] >= hc or dim_locs[b][i][0] < 0 or dim_locs[b][i][1] >= wc or dim_locs[b][i][1] < 0:
                continue
            pts[b,dim_locs[b][i][0],dim_locs[b][i][1]] = 8*dim_val[b][i][0] +  dim_val[b][i][1]

    pts = pts.long()
    #pdb.set_trace()

    #probs bnum x 65 x hc x w
    loss = F.cross_entropy(ipt.to(DEVICE),pts.long().to(DEVICE))
    return loss

def warp_pseudo_gt_pts(homography,gt_pts):
    
    with torch.no_grad():
        batch = len(gt_pts)
        #homography [batch x 8]
        #mat_homo [batch x 3 x 3]
        mat_homo = flat2mat(homography)
        res_pts = []

        for b in range(batch):
            pts = gt_pts[b]
            pts = torch.cat((torch.from_numpy(pts).float(),torch.ones(len(pts),1)),dim=-1)

            res = torch.matmul(mat_homo[b],pts.transpose(0,1))
            res = res/res[2,:].unsqueeze(0)
            
            '''
            if (res[0,:].numpy() > 160).any():
                print('X out of bound ')
                print(res[0,:])
            if (res[1,:].numpy() > 120).any():
                print('Y out of bound ')
                print(res[1,:])
            '''
            res_pts.append(res[:2,:].transpose(0,1))
        return res_pts

    

def generate_homography(params):
    
    plant_imgs = [f for f in os.listdir(params.dset_path) if f.endswith('.jpg')]
    plant_imgs = plant_imgs[:int(len(plant_imgs)/2)]
    
    print('Loaded {} images for training'.format(len(plant_imgs)))
              
    train_loader = DataLoader(PlantDatasetLoader(dataset_path=params.dset_path,plant_imgs=plant_imgs,
                                                 downsample_percent=params.downsample_percent),
                              batch_size=params.batch,shuffle=False,collate_fn=homography_collate_fn)
    
    base_detector = SuperPointNet()
    base_detector.load_state_dict(torch.load(params.base_model_dict))
    base_detector.to(DEVICE)
    base_detector.eval()
            
    begin = time.time()
        
    for batch_idx, (imgs,fnames) in enumerate(train_loader):
            
        #get pix locs from homography adaptation
        batch = imgs.shape[0]
        for b in range(batch):
            
            while(True):
                pix_locs = homo_adapt_helper(imgs[b].numpy(),base_detector)
                if len(pix_locs) > 0:
                    break
                else:
                    print('Rerunning homography!!!!')
                    
            #write it to file
            np.save(fnames[b][:-4]+'_homography.npy',pix_locs)

            percent_complete = params.batch * batch_idx * 100. / len(plant_imgs)
            print('\x1b[2K\rElapsed:{0:.2f} min '.format((time.time()-begin)/60.)+
                  'Batch {0:.2f} (idx:{1}) '.format(percent_complete,batch_idx) +
                  'Writing: {}'.format(fnames[b][:-4]+'_homography.npy'),end='\r')


    
def train(params):
    
    writer = SummaryWriter(log_dir='runs/'+params.runid)


    plant_imgs = [f for f in os.listdir(params.dset_path) if f.endswith('.jpg')]
    plant_imgs = plant_imgs[:int(len(plant_imgs)/2)]

    homographies = [f for f in os.listdir(params.dset_path) if f.endswith('.npy')]
    homographies = homographies[:int(len(homographies))]

    print('Loaded {} images and {} homographies for training'.format(len(plant_imgs),len(homographies)))
          
    if params.dev:
        #just choose 100 images for dev set
        plant_imgs = plant_imgs[:10]
        
    train_loader = DataLoader(PlantDatasetLoader(dataset_path=params.dset_path,plant_imgs=plant_imgs,
                                                 downsample_percent=params.downsample_percent,
                                                homographies=homographies),
                              batch_size=params.batch,shuffle=True,collate_fn=plant_dataset_collate_fn)
    
    model = SuperPointNet()
    #load base model weights
    model.load_state_dict(torch.load(params.base_model_dict))
    model.to(DEVICE)
    model.train()
    
    base_detector = SuperPointNet()
    base_detector.load_state_dict(torch.load(params.base_model_dict))
    base_detector.to(DEVICE)
    base_detector.eval()
        
    optimizer = torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=params.weightdecay)
    
    balance_factor = 1.
    
    for e in range(params.resume_epoch,params.epoch):
        
        epoch_loss = 0
        epoch_img_floss = 0
        epoch_warp_floss = 0
        epoch_desc_loss = 0
        img_count = 0
        
        begin = time.time()
        
        for batch_idx, (imgs,warps,homo_tfs,homo_adapt_pts,fnames) in enumerate(train_loader):
            
            #ipt bnum x 65 x hc x wc
            bnum = imgs.shape[0]
            
            ipt_imgs,desc_imgs = model(imgs.float().unsqueeze(1).to(DEVICE))
            ipt_warps,desc_warps = model(warps.float().unsqueeze(1).to(DEVICE))
            
            #Calculate the descriptor loss
            desc_loss = descriptor_loss(desc_imgs,desc_warps,homo_tfs)
                        
            img_feature_point_loss = feature_point_loss(ipt_imgs,homo_adapt_pts)
            warp_pix_locs = warp_pseudo_gt_pts(homo_tfs,homo_adapt_pts)
            warp_feature_point_loss = feature_point_loss(ipt_warps,warp_pix_locs)
            
            loss = balance_factor * desc_loss + img_feature_point_loss + warp_feature_point_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_img_floss += img_feature_point_loss.item()
            epoch_warp_floss += warp_feature_point_loss.item()
            epoch_desc_loss += desc_loss.item()
            epoch_loss += loss.item()
            
            percent_complete = params.batch * batch_idx * 100. / len(plant_imgs)
            print('\x1b[2K\rEpoch {0} Loss:{1:.3f} '.format(e+1,loss.item())+
                  'Elapsed:{0:.2f} min '.format((time.time()-begin)/60.)+
                  'Batch {0:.2f}% (idx:{1})'.format(percent_complete,batch_idx),end='\r')
            img_count += bnum

        writer.add_scalar('train/loss',epoch_loss,e+1)
        writer.add_scalar('train/img_floss',epoch_img_floss,e+1)
        writer.add_scalar('train/warp_floss',epoch_warp_floss,e+1)
        writer.add_scalar('train/desc_loss',epoch_desc_loss,e+1)
        
        elapsed = (time.time() - begin)/60.
        print('\x1b[2K\rEnd of epoch {0} Loss:{1:.5f} Took:{2:.2f} min'.format(e+1,epoch_loss,elapsed))
        
        save_model(model,e,epoch_loss,params.runid)
        

        
    
def main():
    parser = argparse.ArgumentParser(description='Specialized Superpoint')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--dev',action='store_true')
    #parser.add_argument('--test',action='store_true')
    #parser.add_argument('--validate',action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--base-model-dict',default='e_400_a_10.8703.dict')
    parser.add_argument('--dset-path',default='datasets/plant_data/train')
    parser.add_argument('--high-res',action='store_false') #will use high-res by default
    parser.add_argument('--outfile',default='validation_output')
    parser.add_argument('--epoch',default=1000,type=int)
    parser.add_argument('--resume-epoch',default=0,type=int)
    parser.add_argument('--weightdecay',default=1e-6,type=float)
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--batch',default=64,type=int)
    parser.add_argument('--runid')
    parser.add_argument('--downsample-percent',default=0.1,type=float)
    parser.add_argument('--generate-homography',action='store_true')
    
    params = parser.parse_args()

    if params.runid is  None:
        params.runid = '{0:.0f}'.format(time.time())
    
    log_settings(params)
    
    if params.high_res:
        params.dset_path = params.dset_path+'/high_res/'
    else:
        params.dset_path = params.dset_path+'/low_res/'
        
    if params.generate_homography:
        generate_homography(params)
        
    if params.train:
        train(params)
        

        
if __name__ == '__main__':    
    main()
    