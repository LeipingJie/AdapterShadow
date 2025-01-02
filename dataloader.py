import os
import random

from pathlib import Path
import numpy as np
import torch
import torch.utils.data.distributed
import torchvision.transforms.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from shutil import copyfile
from PIL import Image

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms(mode):
    return transforms.Compose([ToTensor(mode=mode)])

class DataFiles():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        root_dataset = ''
        if args.dataset_name == 'sbu':
            root_dataset = args.root_sbu
            root_dataset = Path(root_dataset)
            train_dir = root_dataset.joinpath('SBUTrain4KRecoveredSmall', 'ShadowImages')
            train_imgs = train_dir.rglob('*.jpg')
            self.train_jpgs = [(p, root_dataset.joinpath('SBUTrain4KRecoveredSmall', 'ShadowMasks', p.name[:-3]+'png')) for p in train_imgs]
            
            test_dir = root_dataset.joinpath('SBU-Test', 'ShadowImages')
            test_imgs = test_dir.rglob('*.jpg')
            self.test_jpgs = [(p, root_dataset.joinpath('SBU-Test', 'ShadowMasks', p.name[:-3]+'png')) for p in test_imgs]

            print('==> using SBU, train {%d}, test{%d}'%(len(self.train_jpgs), len(self.test_jpgs)))

        elif args.dataset_name == 'sbu_new':
            root_dataset = args.root_sbu_new
            root_dataset = Path(root_dataset)
            self.train_jpgs = []
            test_dir = root_dataset.joinpath('ShadowImages')
            test_imgs = test_dir.rglob('*.jpg')
            self.test_jpgs = [(p, root_dataset.joinpath('ShadowMasks', p.name[:-3]+'png')) for p in test_imgs]
        
        # ucf only for testing
        elif args.dataset_name == 'ucf':
            root_dataset = Path(args.root_ucf)
            test_dir = root_dataset.joinpath('InputImages')
            test_imgs = test_dir.rglob('*.jpg')
            self.test_jpgs = [(p, root_dataset.joinpath('GroundTruth', p.name[:-3]+'png')) for p in test_imgs]
        
        elif args.dataset_name == 'istd':
            root_dataset = Path(args.root_istd)
            train_dir = root_dataset.joinpath('train/train_A')
            train_imgs = train_dir.rglob('*.png')
            self.train_jpgs = [(p, root_dataset.joinpath('train/train_B', p.name[:-3]+'png')) for p in train_imgs]

            test_dir = root_dataset.joinpath('test/test_A')
            test_imgs = test_dir.rglob('*.png')
            self.test_jpgs = [(p, root_dataset.joinpath('test/test_B', p.name[:-3]+'png')) for p in test_imgs]

        elif args.dataset_name == 'cuhk':
            root_dataset = Path(args.root_cuhk)
            
            train_img_txt = open(os.path.join(args.root_cuhk, 'train.txt'))
            self.train_jpgs = []
            for img_list in train_img_txt:
                x = img_list.split()
                # train/mask_ADE/ADE_train_00001071.png
                self.train_jpgs.append((root_dataset.joinpath(x[0]), (root_dataset.joinpath(x[1]))))

            train_img_txt.close()
            
            test_img_txt = open(os.path.join(args.root_cuhk, 'test.txt'))
            self.test_jpgs = []
            for img_list in test_img_txt:
                x = img_list.split()
                self.test_jpgs.append((root_dataset.joinpath(x[0]), (root_dataset.joinpath(x[1]))))

            test_img_txt.close()

    def get_pairs(self):
        if self.dataset_name == 'ucf' or self.dataset_name == 'itsd':
            return None, self.test_jpgs

        return self.train_jpgs, self.test_jpgs
    
    def get_train(self):
        return self.train_jpgs

    def get_test(self):
        return self.test_jpgs
        
def random_click(mask, point_labels=1, inout=1):
    indices = np.argwhere(mask>0.0)
    n = len(indices)
    # print('>>>>', mask.shape, indices.shape, n)
    if n == 0:
        h, w = mask.shape
        return np.array([h//2, w//2])
    else:
        return indices[np.random.randint(n)]

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, filenames, transform=None):
        self.args = args
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.filenames = filenames
        self.overfitting = False

    def __len__(self):
        if self.overfitting:
            return 4
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.filenames[idx]
        if self.overfitting:
            name = 'lssd2645'
            image_path = Path(f'/home/mail/2017m7/m730602003/dataset/shadow/SBU-shadow/SBUTrain4KRecoveredSmall/images/{name}.jpg')
            mask_path = Path(f'/home/mail/2017m7/m730602003/dataset/shadow/SBU-shadow/SBUTrain4KRecoveredSmall/labels/{name}.png')
        else:
            image_path, mask_path = sample[0], sample[1]

        # image = cv2.imread(str(image_path))[:,:,::-1]/255.
        image = np.array(Image.open(str(image_path))).astype(float) / 255.
        orig_image = image.copy()
        mask = np.array(Image.open(str(mask_path)).convert('L')).astype(float)[:,:,np.newaxis] / 255.
        # mask = cv2.imread(str(mask_path), 0)[:,:,np.newaxis]/255.
        orig_size = np.array(mask.shape[:-1])

        if self.mode=='train':
            # random crop
            image, mask = self.random_crop(image, mask)
            # random flip 
            image, mask = self.random_flip(image, mask)

        # resize 
        if self.mode=='train':
            image, mask = self.resize(image, self.args.image_size, mask, self.args.mask_size)
            # image, mask = self.brightness(image, mask)
        else:
            # image, mask = self.resize(image, self.args.image_size, self.args.mask_size, mask)
            image = self.resize(image, self.args.image_size)

        pt = random_click(mask, point_labels=1, inout=1)
        
        name = image_path.name.split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}

        sample = {
            'image': image.astype(float), 
            'mask': mask.astype(float)
        }

        if self.mode!='train':
            sample['image_path'] = image_path.name
            sample['mask_path'] = mask_path.name

        if self.transform:
            sample = self.transform(sample)

        sample['pt'] = pt
        sample['p_label'] = 1
        sample['image_meta_dict'] = image_meta_dict
        sample['orig_size'] = orig_size
        if self.mode!='train':
            sample['orig_image'] = orig_image
        
        return sample
    
    def brightness(self, image, mask, value=0.2):
        image = torch.from_numpy(image)
        low, high = 1 - value, 1 + value
        brightness_factor = random.uniform(low, high)
        image = F.adjust_brightness(image.permute(2,0,1), brightness_factor).permute(1,2,0)
        return image.numpy(), mask
    
    ### data augmentation
    def random_flip(self, image, mask):
        # random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            mask = (mask[:, ::-1, :]).copy()

        return image, mask

    def random_crop(self, img, mask, height=None, width=None):
        # random crop
        h, w, c = img.shape
        #print(h, w, c)
        if height is None:
            height = int(h*0.9)
        if width is None:
            width = int(w*0.9)

        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        mask = mask[y:y + height, x:x + width, :]

        return img, mask
    
    def resize(self, img, isize, mask=None, msize=None):
        if mask is None:
            return cv2.resize(img, (isize, isize))
        else:
            return cv2.resize(img, (isize, isize)), cv2.resize(mask, (msize, msize))
    

class ShadowDataLoader(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_pairs, self.test_pairs = DataFiles(args).get_pairs()
        # print(f'===> train: {len(self.train_pairs)}, test: {len(self.test_pairs)}')

    def get_dataloader(self, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(self.args, mode, self.train_pairs, transform=preprocessing_transforms(mode))
            data = DataLoader(self.training_samples, self.args.bs,
                                   shuffle=True,
                                   num_workers=self.args.bs,
                                   pin_memory=True)
            return data
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(self.args, mode, self.test_pairs, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)
            return data
        else:
            print('mode should be one of \'train, test\'. Got {}'.format(mode))

        return 

    def train_dataloader(self):
        mode = 'train'
        self.training_samples = DataLoadPreprocess(self.args, mode, self.train_pairs, transform=preprocessing_transforms(mode))        
        data = DataLoader(self.training_samples, self.args.bs, shuffle=True, num_workers=self.args.bs, pin_memory=True)
        return data

    def val_dataloader(self):
        mode = 'val'
        self.testing_samples = DataLoadPreprocess(self.args, mode, self.test_pairs, transform=preprocessing_transforms(mode))
        data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=0, pin_memory=True)
        return data


class Brightness(object):
    def __init__(self):
        self.colorjitter = transforms.ColorJitter(brightness=0.5)

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        image = self.colorjitter(image)
        return {'image': image, 'mask': mask}

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        # image = self.normalize(image)

        mask = self.to_tensor(sample['mask'])
        if self.mode == 'train':
            return {'image': image, 'mask': mask}
        else:
            return {'image': image, 'mask': mask, 'image_path': sample['image_path'], 'mask_path': sample['mask_path']}

    def to_tensor(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            if pic.ndim==3:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
                return img
            elif pic.ndim==2:
                img = torch.from_numpy(pic).unsqueeze(0)
                return img
        
        return pic

class Brightness(object):
    def __init__(self):
        self.colorjitter = transforms.ColorJitter(brightness=0.5)

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        image = self.colorjitter(image)
        return {'image': image, 'mask': mask}
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset', fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--dataset_name', default='sbu', type=str, choices=['sbu', 'cuhk'], help='tag for one experiment')
    parser.add_argument('--root_cuhk', default='D:/dataset/shadow/SBU/SBU-shadow', type=str, help='dataset root directory')
    parser.add_argument('--root_sbu', default='D:/dataset/shadow/SBU/SBU-shadow', type=str, help='dataset root directory')
    parser.add_argument('--bs', default=1, type=int, help='dataset root directory')
    parser.add_argument('--dist', default=0, type=int, help='dataset root directory')
    parser.add_argument('--input_size', default=416, type=int, help='resize width and height for input image')
    args = parser.parse_args()
    
    shadow_dataloader = ShadowDataLoader(args)
    train_loader = shadow_dataloader.get_dataloader('train')
    import matplotlib.pyplot as plt
    for idx, sample in enumerate(train_loader):
        print(sample['image'].shape, sample['mask'].shape)
        print(sample['image'].min(), sample['mask'].min())
        print(sample['image'].max(), sample['mask'].max())
        if idx>3:
            break
        else:
            continue
        plt.subplot(121)
        plt.imshow(sample['image'][0].permute(1,2,0).numpy())
        plt.subplot(122)
        plt.imshow(sample['mask'][0].permute(1,2,0).numpy())
        plt.show()
