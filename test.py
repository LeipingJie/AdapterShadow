import os
import torch

import cfg
from models.sam import SamPredictor, sam_model_registry
from dataloader import ShadowDataLoader, DataFiles
import function
from utils import calc_BER

from PIL import Image
import numpy as np
from tqdm import tqdm

args = cfg.parse_args('test')

log_dir = f'./logs_{args.exp_name}/version_0'

args.weights = os.path.join(f'{log_dir}/checkpoints', 'last.ckpt')
print('weights: ', args.weights)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt)

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
function.test_sam(args, test_loader, net)

# start calculating ber
print('>>> start calculating ber ...')
accuracy = 0.0
_TP = 0.0
_TN = 0.0
_Np = 0.0
_Nn = 0.0
total = 0

df_obj = DataFiles(args)
test_files = df_obj.get_test()

for _, mask_path in tqdm(test_files):
    gt_mask = (np.array(Image.open(mask_path))>0).astype(np.uint8)*255
    h, w = gt_mask.shape

    pred_path = os.path.join(args.out_dir, os.path.basename(mask_path))
    pred_img = (np.array(Image.open(pred_path))>0).astype(np.uint8)*255

    n_tp, n_tn, n_p, n_n, _ = calc_BER(gt_mask, pred_img)
    _TP += n_tp
    _TN += n_tn
    _Np += n_p
    _Nn += n_n
    total += h*w

TP, TN, Np, Nn = _TP, _TN, _Np, _Nn
accuracy = (TP+TN) * 1.0 / total
ber = (1 - 0.5*(TP/Np+TN/Nn))*100
shadow_ber = (1 - TP/Np)*100
noshadow_ber = (1 - TN/Nn)*100
str_result = 'Accuracy: %f, BER: %f, Shadow Ber: %f, Non-Shadow Ber: %f'%(accuracy, ber, shadow_ber, noshadow_ber)
print(str_result)


    