import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime

import cfg
from utils import *
from models.sam import SamPredictor, sam_model_registry
import function
from dataloader import ShadowDataLoader

args = cfg.parse_args()

set_seed()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt).cuda()

if args.pretrain:
    pass

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

###################################
# dataset
###################################
shadow_dataset = ShadowDataLoader(args)
train_loader = shadow_dataset.get_dataloader('train')
val_loader = shadow_dataset.get_dataloader('val')

'''checkpoint path and tensorboard'''
writer = SummaryWriter(log_dir=os.path.join(args.path_helper['log_path'], args.net, datetime.now().isoformat()))
checkpoint_path = os.path.join(args.path_helper['ckpt_path'], '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
for epoch in range(args.epochs):
    # if args.mod == 'sam_adpt':
    if True:
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == args.epochs-1:
            tol, (eiou, edice) = function.validation_sam(args, val_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False

writer.close()