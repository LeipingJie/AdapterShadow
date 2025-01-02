import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from monai.losses import DiceCELoss
from kornia.morphology import erosion

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cfg import parse_args
from utils import *
# from models.sam import SamPredictor, sam_model_registry
# from models_eff.sam import SamPredictor, sam_model_registry
# from models_fuse.sam import SamPredictor, sam_model_registry
# from models_nf.sam import SamPredictor, sam_model_registry
# from models_df.sam import SamPredictor, sam_model_registry
from models_dense.sam import SamPredictor, sam_model_registry
# from models_cat_fusion.sam import SamPredictor, sam_model_registry

from dataloader import ShadowDataLoader
from utils import calc_BER, cal_acc
from PIL import Image

class WeightedCELoss(nn.Module):
    def __init__(self):
        super(WeightedCELoss, self).__init__()
        self.epsilon = 1e-10

    def forward(self, pred, gt):
        if pred.shape!=gt.shape:
            pred = F.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)

        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt) * 1.0 + self.epsilon
        count_neg = torch.sum(1. - gt) * 1.0 + self.epsilon
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-5

    def forward(self, logits, target, alpha=None, reduction='mean'):
        if logits.shape!=target.shape:
            logits= F.interpolate(logits, target.shape[-2:], mode='bilinear', align_corners=True)

        if alpha==None:
            alpha = self.alpha

        logits = logits.flatten(1)
        target = target.flatten(1)
        pt = torch.clip(torch.sigmoid(logits), self.eps, 0.99999)
        loss = -alpha*(1-pt)**self.gamma*target*torch.log(pt) - (1-alpha)*pt**self.gamma*(1-target)*torch.log(1-pt)
        if reduction=='mean':
            return torch.mean(loss)
        elif reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


def focal_loss(device=None, alpha=8.0/9, gamma=2.0):
    return FocalLoss(alpha, gamma).to(device) if device is not None else FocalLoss(alpha, gamma).cuda()


def weight_ce_loss(device=None):
    return WeightedCELoss().to(device) if device is not None else WeightedCELoss().cuda()


class SAMNetFramework(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hparams.learning_rate = args.lr
        self.save_hyperparameters()
        self.model = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt)
        # self.loss_fun = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if self.args.loss_type=='focal':
            self.loss_fun = focal_loss()
        elif self.args.loss_type=='wce':
            self.loss_fun = weight_ce_loss()

        self.register_buffer("kernel", torch.ones(5, 5))
        # freeze original sam 
        for n, value in self.model.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            # else:
            #     print(n)

        # assert 1==0
        # three tyeps, gt_pt, gen_mask, gen_pt
        self.type = args.type
        assert self.type in ['gt_pt', 'gen_mask', 'gen_pt', 'gen_mask_pt']

    def forward(self, x):
        return self.model(x)

    def _calculate_train_loss(self, batch, batch_idx):
        imgs = batch['image']
        masks = batch['mask']

        if 'pt' not in batch:
            imgs, pt, masks = generate_click_prompt(imgs, masks)
        else:
            pt = batch['pt']
            point_labels = batch['p_label']

        b_size, c, w, h = imgs.size()
        longsize = w if w >=h else h

        imgs = imgs.float()

        if self.type=='gen_mask':
            # mask from CNN
            pred_mask, cnn_features = self.model.mask_generator(imgs)
            pred_mask_erosion = erosion(pred_mask, self.kernel)
            # upsample
            pred_mask_erosion = F.interpolate(pred_mask_erosion, \
            (self.args.mask_size, self.args.mask_size), mode='bilinear', align_corners=True)
            se, de = self.model.prompt_encoder(points=None, boxes=None, masks=pred_mask_erosion, )

        elif self.type=='gt_pt':
            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            with torch.no_grad():
                cnn_features = None
                se, de = self.model.prompt_encoder(points=pt, boxes=None, masks=None, )

        elif self.type=='gen_pt':
            # topk points
            pred_mask, cnn_features = self.model.mask_generator(imgs)
            m_b, m_c, m_h, m_w = pred_mask.shape
            pred_mask = pred_mask.flatten(2)
            
            _, indices = torch.topk(pred_mask, self.args.npts, 2)
            hs = torch.div(indices, m_w, rounding_mode='floor')
            ws = indices % m_w
            coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
            labels_torch = torch.ones(m_b, self.args.npts)

            if self.args.use_neg_points:
                _, neg_indices = torch.topk(pred_mask, self.args.npts, 2, largest=False)
                hs = torch.div(neg_indices, m_w, rounding_mode='floor')
                ws = indices % m_w
                neg_coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                neg_labels_torch = torch.zeros(m_b, self.args.npts)
                coords_torch = torch.cat((coords_torch, neg_coords_torch), dim=1)
                labels_torch = torch.cat((labels_torch, neg_labels_torch), dim=1)

            pt = (coords_torch, labels_torch)
            se, de = self.model.prompt_encoder(points=pt, boxes=None, masks=None, )
            pred_mask = pred_mask.view(m_b, m_c, m_h, m_w)

        elif self.type=='gen_mask_pt':
            # topk points
            pred_mask, cnn_features = self.model.mask_generator(imgs)

            # mask
            pred_mask_erosion = erosion(pred_mask, self.kernel)
            mask_size = (self.args.mask_size, self.args.mask_size)
            if pred_mask_erosion.shape[-2:]!=mask_size:
                pred_mask_erosion = F.interpolate(pred_mask_erosion, mask_size, mode='bilinear', align_corners=True)

            # points
            m_b, m_c, m_h, m_w = pred_mask.shape
            pred_mask_clone = pred_mask.clone().view(m_b, m_c, -1)
            _, indices = torch.topk(pred_mask_clone, self.args.npts, 2)
            hs = torch.div(indices, m_w, rounding_mode='floor')
            ws = indices % m_w
            coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
            labels_torch = torch.ones(m_b, self.args.npts)
            pt = (coords_torch, labels_torch)

            se, de = self.model.prompt_encoder(points=pt, boxes=None, masks=pred_mask_erosion, )
        
        # cnn feature
        if self.args.plug_features_fusion:
            imge = self.model.image_encoder(imgs, None)
            # fuse
            imge = self.model.vit_cnn_fusion(imge, cnn_features[0])
        else:
            imge = self.model.image_encoder(imgs, cnn_features, pred_mask)
            
        pred, _ = self.model.mask_decoder(
            image_embeddings=imge,
            image_pe=self.model.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
            cnn_decoder_features=cnn_features[-1] if self.args.plug_decoder_feature else None
        )

        loss = self.loss_fun(pred, masks)# + self.loss_fun(pred_mask_erosion, masks)
        if self.type!='gt_pt':
            loss += self.loss_fun(pred_mask, masks)

        self.log('train/alpha', self.model.mask_decoder.alpha)
        self.log('train/train_loss', loss)
        return loss

    def _calculate_val_loss_acc(self, batch):
        imgs = batch['image']
        masks = batch['mask']
        
        # print('+++> ', masks.shape, batch['mask_path'])
        if 'pt' not in batch:
            imgsw, pt, masksw = generate_click_prompt(imgs, masks)
        else:
            pt = batch['pt']
            point_labels = batch['p_label']

        imgs = imgs.float()
        '''val'''
        with torch.no_grad():
            if self.type=='gt_pt':
                cnn_features = None
                if point_labels[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                se, de = self.model.prompt_encoder(points=pt, boxes=None, masks=None)

            elif self.type=='gen_mask':
                # mask from CNN
                pred_mask, cnn_features = self.model.mask_generator(imgs)
                pred_mask_erosion = erosion(pred_mask, self.kernel)
                pred_mask_erosion = F.interpolate(pred_mask_erosion, \
                (self.args.mask_size, self.args.mask_size), mode='bilinear', align_corners=True)
                se, de = self.model.prompt_encoder(points=None, boxes=None, masks=pred_mask_erosion)

            elif self.type=='gen_pt':
                # topk points
                pred_mask, cnn_features = self.model.mask_generator(imgs)
                m_b, m_c, m_h, m_w = pred_mask.shape
                pred_mask = pred_mask.flatten(2)

                _, indices = torch.topk(pred_mask, self.args.npts, 2)
                hs = torch.div(indices, m_w, rounding_mode='floor')
                ws = indices % m_w
                coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                labels_torch = torch.ones(m_b, self.args.npts)

                if self.args.use_neg_points:
                    _, neg_indices = torch.topk(pred_mask, self.args.npts, 2, largest=False)
                    hs = torch.div(neg_indices, m_w, rounding_mode='floor')
                    ws = indices % m_w
                    neg_coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                    neg_labels_torch = torch.zeros(m_b, self.args.npts)
                    coords_torch = torch.cat((coords_torch, neg_coords_torch), dim=1)
                    labels_torch = torch.cat((labels_torch, neg_labels_torch), dim=1)

                pt = (coords_torch, labels_torch)
                se, de = self.model.prompt_encoder(points=pt, boxes=None, masks=None, )
                pred_mask = pred_mask.view(m_b, m_c, m_h, m_w)
            
            elif self.type=='gen_mask_pt':
                # topk points
                pred_mask, cnn_features = self.model.mask_generator(imgs)

                # mask
                pred_mask_erosion = erosion(pred_mask, self.kernel)
                mask_size = (self.args.mask_size, self.args.mask_size)
                if pred_mask_erosion.shape[-2:]!=mask_size:
                    pred_mask_erosion = F.interpolate(pred_mask_erosion, mask_size, mode='bilinear', align_corners=True)

                # points
                m_b, m_c, m_h, m_w = pred_mask.shape
                pred_mask = pred_mask.view(m_b, m_c, -1)
                pts = []
                _, indices = torch.topk(pred_mask, self.args.npts, 2)
                hs = torch.div(indices, m_w, rounding_mode='floor')
                ws = indices % m_w
                coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                labels_torch = torch.ones(m_b, self.args.npts)
                pt = (coords_torch, labels_torch)

                se, de = self.model.prompt_encoder(points=pt, boxes=None, masks=pred_mask_erosion, )
            
            # cnn feature
            if self.args.plug_features_fusion:
                imge = self.model.image_encoder(imgs, None)
                # fuse
                imge = self.model.vit_cnn_fusion(imge, cnn_features[0])
            else:
                imge = self.model.image_encoder(imgs, cnn_features, pred_mask if self.args.plug_image_mask else None)

            pred, _ = self.model.mask_decoder(
                image_embeddings=imge,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
                cnn_decoder_features=cnn_features[-1] if self.args.plug_decoder_feature else None
            )
            
        loss = self.loss_fun(pred, masks)
        self.log('val/val_loss', loss)
        
        h, w = masks.shape[-2:]
        pred = (torch.sigmoid(pred)>0.5).float()
        pred = np.uint8(pred.cpu().numpy().squeeze()*255)
        pred_img_np = np.array(transforms.Resize((h, w))(Image.fromarray(pred).convert('L')))
        # pred_img_np = pred.squeeze(0).permute(1, 2, 0).cpu().squeeze().float().numpy()
        mask_img_np = np.uint8(masks.squeeze().cpu().squeeze().float().numpy()*255)

        # print('===> ', pred.shape, masks.shape, imgs.shape, mask_img_np.shape, pred_img_np.shape)
        total = w*h
        n_tn, n_tp, n_n, n_p, ber = cal_acc(pred_img_np, mask_img_np, self.args.thres)
        return loss, n_tn, n_tp, n_n, n_p, total, ber
        
        
    def training_step(self, batch, batch_idx):
        train_loss = self._calculate_train_loss(batch, batch_idx)
        info = {'loss' : train_loss}
        return info


    def validation_step(self, batch, batch_idx):
        val_loss, n_tn, n_tp, n_n, n_p, total, ber = self._calculate_val_loss_acc(batch)
        info = {'loss':val_loss, 'n_tn':n_tn, 'n_tp': n_tp, 'n_n': n_n,  'n_p':n_p, 'total':total, 'ber':ber}
        return info


    def validation_epoch_end(self, outputs):
        TN = torch.stack([torch.FloatTensor([x['n_tn']]) for x in outputs]).sum()
        TP = torch.stack([torch.FloatTensor([x['n_tp']]) for x in outputs]).sum()
        Nn = torch.stack([torch.FloatTensor([x['n_n']]) for x in outputs]).sum()
        Np = torch.stack([torch.FloatTensor([x['n_p']]) for x in outputs]).sum()
        total = torch.stack([torch.FloatTensor([x['total']]) for x in outputs]).sum()

        accuracy = (TP+TN) * 1.0 / total
        ber = (1 - 0.5*(TP/Np+TN/Nn))*100
        ber_shadow = (1 - TP / Np) * 100
        ber_unshadow = (1 - TN / Nn) * 100

        print(f'>>>> all: {ber}, shadow: {ber_shadow}, non-shadow: {ber_unshadow}')

        self.log('accuracy', accuracy, prog_bar=True)
        self.log('ber', ber, prog_bar=True)
        self.log('shadow', ber_shadow, prog_bar=True)
        self.log('non-shadow', ber_unshadow, prog_bar=True)


    def configure_optimizers(self):
        if self.args.lr_decay == 'constant':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            return optimizer
        elif self.args.lr_decay == '1cycle':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.args.wd)
            lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                        self.args.lr,
                                                        epochs=self.args.epochs,
                                                        steps_per_epoch=self.args.steps_per_epoch,
                                                        cycle_momentum=True,
                                                        base_momentum=0.85,
                                                        max_momentum=0.95,
                                                        last_epoch=-1,
                                                        pct_start=self.args.pct_start,
                                                        div_factor=self.args.div_factor,
                                                        final_div_factor=self.args.final_div_factor)
            return [optimizer], [lr_scheduler]
        else:
            pass

def main(hparams):
    ##################################################################
    # dataset
    ##################################################################
    # shadow_dataset = ShadowDataLoader(hparams)
    # train_loader = shadow_dataset.get_dataloader('train')
    # val_loader = shadow_dataset.get_dataloader('val')

    # scale_size = (hparams.scale_size, hparams.scale_size)
    # crop_size = (hparams.crop_size, hparams.crop_size)
    # dm = ISTD_Dataset(data_dir=hparams.root_istd, batch_size=hparams.bs, num_workers=hparams.n_workers, return_name=True, scale_size=scale_size, crop_size=crop_size)
    # dm.setup()

    dm = ShadowDataLoader(hparams)
    hparams.steps_per_epoch = len(dm.train_dataloader())

    _save_dir, _name = os.getcwd(), f'logs_{hparams.exp_name}'
    ##################################################################
    # save code
    ##################################################################
    code_save_dir = os.path.join(_save_dir, _name)
    backup_code(code_save_dir)
    
    ##################################################################
    # call backs
    ##################################################################

    # learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # logger
    if hparams.logger == 'tensorboard':
        logger_instance = pl_loggers.TensorBoardLogger(save_dir=_save_dir, version=None, name=_name)
    elif hparams.logger == 'wandb':
        logger_instance = pl_loggers.WandbLogger(project='sam_shadow')
    
    # checkpoint saver
    checkpoint_callback = ModelCheckpoint(
        monitor='ber',
        filename='sdnet-{epoch:02d}-{ber:.3f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )

    model = SAMNetFramework(hparams)
    trainer = Trainer(
        max_epochs=hparams.epochs,
        gpus=len(params.gpus.split(',')),
        accelerator='gpu',
        default_root_dir=_name,
        logger=logger_instance,
        callbacks=[checkpoint_callback, lr_monitor],
        progress_bar_refresh_rate=1,
        precision=16 if hparams.amp else 32, 
        check_val_every_n_epoch=1,
        #overfit_batches=10
    )

    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    params = parse_args()
    seed_everything(params.seed)
    # set_seed(params.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus

    main(hparams=params)