
import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
from skimage import io
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
import time
import cfg
from tqdm import tqdm
from utils import *
from einops import rearrange
import models.sam.utils.transforms as samtrans

import shutil
import tempfile

import matplotlib.pyplot as plt
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].cuda()
            masks = pack['mask'].cuda()

            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']
            
            showp = pt

            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda()
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.float().cuda()
            
            '''Train'''
            for n, value in net.image_encoder.named_parameters():
                if "Adapter" not in n:
                    value.requires_grad = False

            imge = net.image_encoder(imgs)

            with torch.no_grad():
                # imge= net.image_encoder(imgs)
                se, de = net.prompt_encoder(points=pt, boxes=None, masks=None, )
                
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
              )

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    mix_res = (0,0,0,0)
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            masksw = pack['mask'].to(dtype = torch.float32).cuda()
            orig_size = pack['orig_size']
            
            if 'pt' not in pack:
                imgsw, pt, masksw = generate_click_prompt(imgsw, masksw)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            imgs = imgsw
            masks = masksw
            showp = pt
            mask_type = torch.float32

            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda()
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            imgs = imgs.to(dtype = mask_type).cuda()
            '''val'''
            with torch.no_grad():
                imge= net.image_encoder(imgs)
                se, de = net.prompt_encoder(points=pt, boxes=None, masks=None)

                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
                
                tot += lossfunc(pred, masks)

                '''vis images'''
                if args.vis:
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

                temp = eval_seg(pred, masks, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def test_sam(args, val_loader, net: nn.Module):
    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            masksw = pack['mask'].to(dtype = torch.float32).cuda()
            orig_size = pack['orig_size']
            
            if 'pt' not in pack:
                imgsw, pt, masksw = generate_click_prompt(imgsw, masksw)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = imgsw
            masks = masksw
            showp = pt
            mask_type = torch.float32

            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda()
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            imgs = imgs.to(dtype = mask_type).cuda()
            '''test'''
            with torch.no_grad():
                imge= net.image_encoder(imgs)
                se, de = net.prompt_encoder(points=pt, boxes=None, masks=None)

                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )

            # save images
            orig_size = orig_size.numpy().tolist()[0]
            pred = (F.sigmoid(pred)>0.5).float()
            file_name = pack['mask_path'][0]
            out_name = os.path.join(args.out_dir, file_name)

            prediction = np.uint8(pred.cpu().numpy().squeeze()*255)
            pred = np.array(transforms.Resize((orig_size))(Image.fromarray(prediction).convert('L')))
            Image.fromarray(pred).save(out_name)
            pbar.update()