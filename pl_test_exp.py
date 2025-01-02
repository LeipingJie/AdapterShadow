import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from kornia.morphology import erosion
import cv2

import cfg
from models_exp.sam import SamPredictor, sam_model_registry

from dataloader import ShadowDataLoader, DataFiles
from utils import *

from PIL import Image
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def test_sam(args, val_loader, net: nn.Module, out_dir=None):
    # create output directory
    if out_dir is None:
        out_dir = f'./logs_{args.exp_name}/version_0'
        # os.makedirs(args.out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    prefix = args.dataset_name
    coarse_dir = os.path.join(out_dir, prefix+'_cmasks')
    mask_dir = os.path.join(out_dir, prefix+'_masks')
    cmp_dir = os.path.join(out_dir, prefix+'_cmps')
    pt_dir = os.path.join(out_dir, prefix+'_vis')
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)
    # eval mode
    net.eval()

    kernel = torch.ones(5,5).cuda()
    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch

    accuracy = 0.0
    _TP = 0.0
    _TN = 0.0
    _Np = 0.0
    _Nn = 0.0
    total = 0

    total_t = 0.0
    n_imgs = len(val_loader)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            masksw = pack['mask'].to(dtype = torch.float32).cuda()
            orig_img = pack['orig_image']

            orig_size = pack['orig_size']
            
            file_name = pack['mask_path'][0]
            mask_out_name = os.path.join(mask_dir, file_name)
            coarse_out_name = os.path.join(coarse_dir, file_name)
            # if os.path.exists(mask_out_name):
            #     continue
            
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

            imgs = imgs.to(dtype = mask_type).cuda()
            delta_t = 0.0
            with torch.no_grad():
                if args.type=='gen_mask':
                    # mask from CNN
                    pred_mask = net.mask_generator(imgs, imgs.shape[-2:])
                    pred_mask_erosion = erosion(pred_mask, kernel)
                    # upsample
                    mask_size = (args.mask_size, args.mask_size)
                    if pred_mask_erosion.shape[-2:]!=mask_size:
                        pred_mask_erosion = F.interpolate(pred_mask_erosion, mask_size, mode='bilinear', align_corners=True)
                        
                    se, de = net.prompt_encoder(points=None, boxes=None, masks=pred_mask_erosion, )

                elif args.type=='gt_pt':
                    if point_labels[0] != -1:
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda()
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                        pt = (coords_torch, labels_torch)

                    se, de = net.prompt_encoder(points=pt, boxes=None, masks=None, )

                elif args.type=='gen_pt':
                    # topk points
                    t = time.time()
                    if args.vpt:
                        pred_mask, vpt_prompts = net.mask_generator(imgs[:, :, ::4, ::4] if args.small_size else imgs, imgs.shape[-2:], args.vpt)
                    else:
                        pred_mask = net.mask_generator(imgs, imgs.shape[-2:])
                    m_b, m_c, m_h, m_w = pred_mask.shape

                    # topk points
                    if args.sample=='topk':
                        '''
                        pred_mask_clone = pred_mask.clone().view(m_b, m_c, -1)
                        pts = []
                        _, indices = torch.topk(pred_mask_clone, args.npts, 2)
                        hs = torch.div(indices, m_w, rounding_mode='floor')
                        ws = indices % m_w
                        coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                        labels_torch = torch.ones(m_b, args.npts)
                        pt = (coords_torch, labels_torch)

                        se, de = net.prompt_encoder(points=pt, boxes=None, masks=None, )
                        '''
                        pred_mask_clone = torch.sigmoid(pred_mask.clone().view(m_b, m_c, -1))
                        values, indices = torch.topk(pred_mask_clone, args.npts, 2)
                        hs = torch.div(indices, m_w, rounding_mode='floor')
                        ws = indices % m_w
                        coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                        labels_torch = torch.ones(m_b, args.npts)

                        if args.use_neg_points:
                            values, neg_indices = torch.topk(pred_mask_clone, args.npts, 2, largest=False)
                            hs = torch.div(neg_indices, m_w, rounding_mode='floor')
                            ws = neg_indices % m_w
                            neg_coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                            neg_labels_torch = torch.zeros(m_b, args.npts)
                            coords_torch = torch.cat((coords_torch, neg_coords_torch), dim=1)
                            labels_torch = torch.cat((labels_torch, neg_labels_torch), dim=1)
                    elif args.sample=='grid':
                        amp2d = nn.AdaptiveMaxPool2d((args.grid_out_size, args.grid_out_size), return_indices=True)
                        values, indices = amp2d(pred_mask)
                        values = values.flatten(1)
                        indices = indices.flatten(2)
                        hs = torch.div(indices, m_w, rounding_mode='floor')
                        ws = indices % m_w
                        coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                        labels_torch = torch.gt(values, args.grid_thres).float()
                    

                    pt = (coords_torch, labels_torch)
                    se, de = net.prompt_encoder(points=pt, boxes=None, masks=None, )

                    delta_t += (time.time() - t)

                elif args.type=='gen_mask_pt':
                    # topk points
                    pred_mask = net.mask_generator(imgs)

                    # mask
                    pred_mask_erosion = erosion(pred_mask, kernel)
                    mask_size = (args.mask_size, args.mask_size)
                    if pred_mask_erosion.shape[-2:]!=mask_size:
                        pred_mask_erosion = F.interpolate(pred_mask_erosion, mask_size, mode='bilinear', align_corners=True)

                    # points
                    m_b, m_c, m_h, m_w = pred_mask.shape
                    pred_mask_clone = pred_mask.clone().view(m_b, m_c, -1)
                    pts = []
                    _, indices = torch.topk(pred_mask_clone, args.npts, 2)
                    hs = torch.div(indices, m_w, rounding_mode='floor')
                    ws = indices % m_w
                    coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                    labels_torch = torch.ones(m_b, args.npts)
                    pt = (coords_torch, labels_torch)

                    se, de = net.prompt_encoder(points=pt, boxes=None, masks=pred_mask_erosion, )
                
                t = time.time()
                imge= net.image_encoder(imgs, vpt_prompts if args.vpt else None)
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
                delta_t += (time.time() - t)
                total_t += delta_t

            # save images
            orig_size = orig_size.numpy().tolist()[0]
            pred = (torch.sigmoid(pred)>0.5).float()
            out_name = os.path.join(cmp_dir, file_name)

            # save pred coarse images
            # pred_mask_gen = (torch.sigmoid(pred_mask)>0.5).float()
            pred_mask_gen = torch.sigmoid(pred_mask).float()
            pred_mask_gen2 = np.uint8(pred_mask_gen.cpu().numpy().squeeze()*255)
            pred_mask_gen2 = np.array(transforms.Resize((orig_size))(Image.fromarray(pred_mask_gen2).convert('L')))
            Image.fromarray(pred_mask_gen2).save(coarse_out_name)

            # save points
            pt_img = cv2.imread(coarse_out_name)
            pts, labels = pt
            b, n = labels.shape

            for i in range(n):
                color = (0, 255, 0) if labels[0, i]==0 else (0, 0, 255)
                center = pts[0, i].cpu().numpy()/np.array([args.image_size, args.image_size])*pt_img.shape[:2]
                p = center.astype(int).tolist()[::-1]
                cv2.circle(pt_img, center=p, radius=3, color=color, thickness=-1)
                cv2.imwrite(os.path.join(pt_dir, file_name), pt_img)
                
            # save pred images
            prediction = np.uint8(pred.cpu().numpy().squeeze()*255)
            pred = np.array(transforms.Resize((orig_size))(Image.fromarray(prediction).convert('L')))
            
            Image.fromarray(pred).save(mask_out_name)

            # pbar.update()
            
            # rgb_img = Image.fromarray((imgs*255).cpu().squeeze().permute(1,2,0).numpy().astype(np.uint8))
            # rgb_img = np.array(transforms.Resize((orig_size))(rgb_img))
            orig_img = (orig_img.squeeze()*255).numpy().astype(np.uint8)
            masksw = np.array(transforms.Resize((orig_size))(Image.fromarray((masksw>0.5).cpu().squeeze().numpy()).convert('L')))

            h, w = orig_size
            n_tp, n_tn, n_p, n_n, _ = cal_acc(pred, masksw)
            _TP += n_tp
            _TN += n_tn
            _Np += n_p
            _Nn += n_n
            total += h*w

            '''
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.title.set_text('rgb')
            # ax.imshow(imgsw.squeeze().cpu().permute(1,2,0).numpy())
                    
            ax.imshow(orig_img)
            if args.type=='gen_pt':
                # plot selected points
                pts = pt[0].cpu().squeeze()
                for pt in pts:
                    # print(pt[0], args.image_size, orig_size[0])
                    ax.plot(int(pt[0]/args.image_size*orig_size[0]), int(pt[1]/args.image_size*orig_size[1]), label='x', color='r')

            ax = fig.add_subplot(222)
            ax.title.set_text('gt')
            ax.imshow(masksw, cmap='gray')
            
            ax = fig.add_subplot(223)
            ax.title.set_text('mask_gen')
            ax.imshow(pred_mask_gen2, cmap='gray')


            ax = fig.add_subplot(224)
            ax.title.set_text('final')
            ax.imshow(pred, cmap='gray')
            plt.savefig(out_name)
            plt.close()
            '''

        
            pbar.update()

        TP, TN, Np, Nn = _TP, _TN, _Np, _Nn
        accuracy = (TP+TN) * 1.0 / total
        ber = (1 - 0.5*(TP/Np+TN/Nn))*100
        shadow_ber = (1 - TP/Np)*100
        noshadow_ber = (1 - TN/Nn)*100
        str_result = 'Accuracy: %f, BER: %f, Shadow Ber: %f, Non-Shadow Ber: %f'%(accuracy, ber, shadow_ber, noshadow_ber)
        print(str_result)


    print('Parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    print('Average time: ', total_t/n_imgs)
    return mask_dir

args = cfg.parse_args('test')

log_dir = f'./logs_{args.exp_name}/version_0'

args.weights = os.path.join(f'{log_dir}/checkpoints', 'sdnet-epoch=24-ber=2.765.ckpt')

print('weights: ', args.weights)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt)

# freeze original sam 
for n, value in net.image_encoder.named_parameters():
    if "Adapter" not in n:
        value.requires_grad = False
    # else:
    #     print(n)

if args.freeze_backbone:
    for p in net.mask_generator.efficient_encoder.parameters():
        p.requires_grad = False

# load pretrained model
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
checkpoint = torch.load(checkpoint_file, map_location='cpu')
start_epoch = checkpoint['epoch']

state_dict = checkpoint['state_dict']
checkpoint_ = {}
for k, v in state_dict.items():
    if not k.startswith('model.'):
        continue

    k = k[6:] # remove 'model.'
    checkpoint_[k] = v

net.load_state_dict(checkpoint_)

net = net.cuda()
###################################
# dataset
###################################
shadow_dataset = ShadowDataLoader(args)
test_loader = shadow_dataset.get_dataloader('test')

# begain valuation
mask_out_name = test_sam(args, test_loader, net)

# assert 1==0



    
