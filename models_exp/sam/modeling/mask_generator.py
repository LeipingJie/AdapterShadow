import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient import get_efficientnet

model_infos = {
    'b5':[24,40,64,176,2048], 'b4':[24,32,56,160,1792], 'b3':[24,32,48,136,1536], 
    'b2':[16,24,48,120,1408], 'b1':[16,24,40,112,1280], 'b0':[16,24,40,112,1280],
}

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class EfficientEncoder(nn.Module):
    def __init__(self, backbone):
        super(EfficientEncoder, self).__init__()
        self.original_model = get_efficientnet(backbone)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class Decoder(nn.Module):
    def __init__(self, n_channel_features):
        super(Decoder, self).__init__()
        features = n_channel_features[-1]

        self.conv2 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + n_channel_features[-2], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + n_channel_features[-3], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + n_channel_features[-4], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + n_channel_features[-5], output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return torch.tanh(out)

class SDNet(nn.Module):
    def __init__(self, backbone='b0', vpt=False, small=False):
        super(SDNet, self).__init__()
        self.backbone = backbone
        # encoder
        n_channel_features = model_infos[backbone]
        self.efficient_encoder = EfficientEncoder(backbone)

        # decoder
        features = 128

        self.conv = nn.Conv2d(n_channel_features[-1], features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + n_channel_features[-2], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + n_channel_features[-3], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + n_channel_features[-4], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + n_channel_features[-5], output_features=features // 16)

        n_dim_align, ch_base = 32, 64
        self.align1 = nn.Sequential(
            nn.Conv2d(features // 2, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align2 = nn.Sequential(
            nn.Conv2d(features // 4, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align3 = nn.Sequential(
            nn.Conv2d(features // 8, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align4 = nn.Sequential(
            nn.Conv2d(features // 16, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )

        self.predict_mask = nn.Sequential(
            nn.Conv2d(n_dim_align, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

        if vpt:
            self.vpt = vpt
            self.img_encoder_feature = nn.Sequential(
                nn.Conv2d(32, 768, 1, stride=4 if small else 32)
            )
        
    def forward(self, rgb, size, return_prompts=False):
        features = self.efficient_encoder(rgb)

        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        ##############################################    
        # upsampling
        ##############################################

        x_d0 = self.conv(x_block4)
        # layer01
        x_d1 = self.up1(x_d0, x_block3)

        # layer02
        x_d2 = self.up2(x_d1, x_block2)
        x_align2 = self.align2(x_d2)
        
        # layer03
        x_d3 = self.up3(x_d2, x_block1)
        x_align3 = self.align3(x_d3)
        
        # layer04
        x_d4 = self.up4(x_d3, x_block0)
        x_align4 = self.align4(x_d4) # [2, 32, 512, 512]
        pred_mask = self.predict_mask(x_align4)

        # print('*0* ', x_d0.shape)
        # print('*1* ', x_d1.shape)
        # print('*2* ', x_d2.shape)
        # print('*3* ', x_d3.shape)
        # print('*4* ', x_d4.shape)
        # print('*m* ', pred_mask.shape)
        # assert 1==0

        if return_prompts and self.vpt:
            # b, c, h, w = pred_mask.shape
            # dim_vit = 64
            # vpt_prompts = pred_mask.reshape(b, c, int(h/dim_vit), dim_vit, int(w/dim_vit), dim_vit).permute(0, 1, 2, 4, 3, 5).flatten(1,3).permute(0, 2, 3, 1)
            vpt_prompts = self.img_encoder_feature(x_align4)

        pred_mask = F.interpolate(pred_mask, size, mode='bilinear', align_corners=True)
        if return_prompts:
            # print('>>>>', rgb.shape, pred_mask.shape, x_align4.shape, vpt_prompts.shape)
            return pred_mask, vpt_prompts.permute(0, 2, 3, 1)
        else:
            return pred_mask

    def name(self):
        return 'sam_model_mask_gen'