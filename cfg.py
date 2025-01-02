import argparse

def parse_args(mode='train'):
    assert mode in ['train', 'test']    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-gpus', type=str, default='3', help='use gpu or not')
    parser.add_argument('-gpus', type=str, default='0,1,2,3', help='use gpu or not')
    # parser.add_argument('-gpus', type=str, default='4,5,6,7', help='use gpu or not')
    parser.add_argument('-sam_ckpt', default='./checkpoint/sam/sam_vit_b_01ec64.pth', help='sam checkpoint address')
    
    if mode=='train':
        parser.add_argument('-seed', type=int, default=42, help='fix seed') # 42, 20230718
        parser.add_argument('--dist', action='store_true', help='whether use distribution training')
        parser.add_argument('-epochs', type=int, default=40, help='epochs')
        parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
        # lr decay
        parser.add_argument('-lr_decay', type=str, default='constant', choices=['constant', '1cycle'], help='how learning rate decays')

        parser.add_argument('-wd', default=5e-4, type=float, help='weight decay')
        parser.add_argument('-pct_start', default=0.10, type=float, help='The percentage of the cycle (in number of steps) spent increasing the learning rate. Default: 0.3')
        parser.add_argument('-div_factor', default=10, type=float, help="Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25")
        parser.add_argument('-final_div_factor', default=30, type=float, help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4")

        parser.add_argument('-loss_type', type=str, default='focal', choices=['focal', 'wce'], help='type of loss')

    parser.add_argument('-val_freq',type=int, default=2, help='interval between each validation')

    parser.add_argument('-exp_name', type=str, required=True, help='net type')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-dataset_name', type=str, default='sbu', choices=['sbu', 'sbu_new', 'ucf', 'istd', 'cuhk'], help='dataset')
    parser.add_argument('-root_sbu', default='/home/mail/2017m7/m730602003/dataset/shadow/SBU-shadow', type=str, help='sbu dataset root directory')
    parser.add_argument('-root_sbu_new', default='/home/mail/2017m7/m730602003/dataset/shadow/SBUTestNew', type=str, help='sbu dataset root directory')
    parser.add_argument('-root_ucf', default='/home/mail/2017m7/m730602003/dataset/shadow/UCF', type=str, help='ucf dataset root directory')
    parser.add_argument('-root_istd', default='/home/mail/2017m7/m730602003/dataset/shadow/ISTD_Dataset', type=str, help='istd dataset root directory')
    parser.add_argument('-root_cuhk', default='/home/mail/2017m7/m730602003/dataset/shadow/CUHKshadow', type=str, help='cuhk dataset root directory')

    parser.add_argument('-backbone', default='b5', type=str, choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'resnext'], help='backbone for efficientnet')
    parser.add_argument('-logger', default='tensorboard', type=str, choices=['wandb', 'tensorboard'], help='type of logger')
    # amp
    parser.add_argument('-amp', action='store_true', help='whether use mixed precision')

    # adapter
    parser.add_argument('-use_neg_points', action='store_true', help='whether use negative points')
    parser.add_argument('-plug_dense_adapter', action='store_true', help='whether use dense adaptor in image encoder')
    parser.add_argument('-plug_image_adapter', action='store_true', help='whether use adaptor in image encoder')
    parser.add_argument('-plug_image_mask', action='store_true', help='whether use adaptor in image encoder')
    parser.add_argument('-plug_encoder_feature', action='store_true', help='whether use cnn feature in vit encoder')
    parser.add_argument('-plug_decoder_feature', action='store_true', help='whether use cnn feature in mask predictor')
    parser.add_argument('-plug_features_fusion', action='store_true', help='whether use cnn feature in mask predictor')
    parser.add_argument('-plug_idx', type=list, default=[2, 4, 6, 8, 10], help='layers to plug in features')

    parser.add_argument('-freeze_backbone', action='store_true', help='whether freeze backbone')
    parser.add_argument('-vpt', action='store_true', help='visual prompt tuning (shallow)')
    parser.add_argument('-small_size', action='store_true', help='smaller image size for mask generator')

    parser.add_argument('-skip_adapter', action='store_true', help='whether skip on all')
    parser.add_argument('-lora', action='store_true', help='whether use lora')
    parser.add_argument('-multi_branch', action='store_true', help='whether use branch')
    parser.add_argument('-mb_ratio', type=float, default=0.25, help='whether use branch')
    
    parser.add_argument('-sample', type=str, default='topk', choices=['topk', 'grid'], help='sampling methods')
    parser.add_argument('-npts', type=int, default=1, help='number of generated points to prompt')
    parser.add_argument('-grid_out_size', type=int, default=4, help='grid size when using grid sampling')
    parser.add_argument('-grid_thres', type=float, default=0.9, help='threshold for judging positive or negative')

    # only several transformer layers
    parser.add_argument('-ratio_bt', default=1.0, type=float, help='ratio of transformer blocks')


    parser.add_argument('-a1', action='store_true', help='MHA adapter')
    parser.add_argument('-a2', action='store_true', help='FFN adapter')
    parser.add_argument('-all', action='store_true', help='both MHA and FFN adapter')
    parser.add_argument('-tba', action='store_true', help='Transformer block adapter')
    parser.add_argument('-local_vit', action='store_true', help='local_vit')
    parser.add_argument('-down_ratio', default=0.25, type=float, help='downsample ratio for Adapter or LocalAdapater')

    parser.add_argument('-bs', default=2 if mode=='train' else 1, type=int, help='batch size')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-image_size', default=1024, type=int, help='resize width and height for input image')
    parser.add_argument('-mask_size', default=256, type=int, help='resize width and height for input image')
    parser.add_argument('--mode', default=mode, type=str, help='current mode')

    parser.add_argument('-type', type=str, default='gen_pt', choices=['gt_pt', 'gen_mask', 'gen_pt', 'gen_mask_pt'], help='type for prompt')

    # eval
    parser.add_argument('-weights', type=str, default=0, help='the weights file you want to test')
    parser.add_argument('-out_dir', type=str, default='results', help='output directory to save predictions')
    parser.add_argument('-thres', type=int, default=128, help='threshold for calculating ber')

    '''
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, required=True, help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', type=str, required=True, help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=100,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=256, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', default=None , help='sam checkpoint address')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    parser.add_argument('-data_path', type=str, default='../data', help='The path of segmentation data')
    
    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    '''

    opt = parser.parse_args()

    return opt
