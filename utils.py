import os, time, random, logging, shutil
from datetime import datetime
import dateutil.tz
from skimage.transform import resize

import torch
import torchvision
from torch.autograd import Function
import torchvision.utils as vutils

import numpy as np

from skimage.color import rgb2lab
from skimage.transform import resize

def calc_BER(gt, pred, interpolate=True):
    if interpolate:
        pred = (resize(pred/pred.max(), gt.shape)>0.1).astype(np.int)

    gt = (gt>0.1).astype(np.int)
    N_p = np.sum(gt)
    N_n = np.sum(np.logical_not(gt))

    TP = np.sum(np.logical_and(gt, pred))
    TN = np.sum(np.logical_and(np.logical_not(gt), np.logical_not(pred)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))
    return TN, TP, N_n, N_p, ber_


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix, exist_ok=True)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'runs')
    os.makedirs(log_path, exist_ok=True)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    path_dict['sample_path'] = sample_path

    return path_dict


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
    
    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if points != None:
            for i in range(b):
                if args.thd:
                    p = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return


def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w = msk.size()
    msk = msk[:,0,:,:]
    
    pt_list_s = []
    msk_list_s = []
    for j in range(b):
        msk_s = msk[j,:,:]
        indices = torch.nonzero(msk_s)
        if indices.size(0) == 0:
            # generate a random array between [0-h, 0-h]:
            random_index = torch.randint(0, h, (2,)).to(device = msk.device)
            new_s = msk_s
        else:
            random_index = random.choice(indices)
            label = msk_s[random_index[0], random_index[1]]
            new_s = torch.zeros_like(msk_s)
            # convert bool tensor to int
            new_s = (msk_s == label).to(dtype = torch.float)
            # new_s[msk_s == label] = 1
        pt_list_s.append(random_index)
        msk_list_s.append(new_s)
    pts = torch.stack(pt_list_s, dim=0)
    msks = torch.stack(msk_list_s, dim=0)
    pt_list.append(pts)
    msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def calc_BER(pred, gt, interpolate=True):
    if interpolate:
        pred = (resize(pred/pred.max(), gt.shape)>0.1).astype(np.int)

    gt = (gt>0.1).astype(np.int)
    N_p = np.sum(gt)
    N_n = np.sum(np.logical_not(gt))

    TP = np.sum(np.logical_and(gt, pred))
    TN = np.sum(np.logical_and(np.logical_not(gt), np.logical_not(pred)))

    # print('===> ', TN, TP, N_n, N_p, pred.shape, gt.shape)
    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))
    return TN, TP, N_n, N_p, ber_

def cal_acc(prediction, label, thr = 128):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(float)
    label_tmp = label.astype(float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    Union = np.sum(prediction_tmp) + Np - TP

    return TP, TN, Np, Nn, Union

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)
    
def set_seed(seed=20230712):
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# def backup_code(save_dir, dirs_to_save = ['models_dense', 'models_nf', 'models_fuse', 'models_eff', 'models_df', 'models']):
def backup_code(save_dir, dirs_to_save = ['models_exp']):
    fs = os.listdir('./')
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, 'code')
    os.makedirs(save_dir, exist_ok=True)

    # save dirs
    for dts in dirs_to_save:
        d = os.path.join(save_dir, dts)
        if os.path.exists(d):
            shutil.rmtree(d)

        shutil.copytree(dts, d)

    # save files
    for f in fs:
        if not os.path.isdir(f) and (f.endswith('.py') or f.endswith('.sh')):
            shutil.copy(f, os.path.join(save_dir, f))
