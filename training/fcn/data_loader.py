from torch.utils import data
import torch
import numpy as np

import PIL

import glob

class Cloud_Data(data.Dataset):
    class_names = []
    with open('class_names.list', 'r') as f:
        for line in f:
            class_names.append(line.strip())

    class_names = np.array(class_names)
    print(class_names)
    '''
    class_names = np.array([
        'sky',
        'cloud',
    ])
    '''

    def __init__(self, root, image_set='train', backbone='vgg', transform=False):
        self._transform = transform
        self.backbone = backbone
        self.image_set = image_set
        if backbone == 'vgg':
            self.mean = np.array([104.00698793, 116.66876762,
                                  122.67891434])  # BGR
            self.std = np.array([1., 1., 1.])
        else:
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])

        train = glob.glob(root+'/converged/selected300/*.png')
        lbl = glob.glob(root+'/GTmaps/*.png')
        val = glob.glob(root+'/converged/test/*.png')

        self.files = {'train': train, 'val': val, 'lbl':lbl}

    def __len__(self):
        return len(self.files[self.image_set])

    def __getitem__(self, index):
        data_file = self.files[self.image_set][index]
        #load image
        img_file = data_file
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.float64)

        #img = img.transpose(2, 0, 1)
        #load label
        key = img_file.split('/')[-1].split('.')[0]

        def find_lbl():
            for item in self.files['lbl']:
                if key in item:
                   return item

        lbl_file = find_lbl()
        lbl = PIL.Image.open(lbl_file).convert('P')
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = 1
        #lbl[lbl == 255] = -1

        if self._transform:
            return self.transform(img,lbl)
        else:
            return img, lbl


    def transform(self, img, lbl):
        if self.backbone == 'vgg':
            img = np.array(img, dtype=np.uint8)
            lbl = np.array(lbl, dtype=np.uint8)
            img = img[:, :, ::-1]  # RGB -> BGR
            img = np.array(img, dtype=np.float64)
        else:
            img = np.array(img, dtype=np.float64)
            img /= 255.
        lbl = np.array(lbl, dtype=np.float64)
        #lbl[lbl == 255] = -1  # Ignore contour
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= self.std
        img += self.mean
        if self.backbone == 'resnet':
            img *= 255
        img = img.astype(np.uint8)
        if self.backbone == 'vgg':
            img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


def get_loader(opts):
    from data_loader import Pascal_Data
    import os
    kwargs = {'num_workers': 4} if 'cuda' in str(opts.cuda) else {}
    if opts.mode in ['train', 'demo']:
        modes = ['train', 'val']
    else:
        modes = [opts.mode, opts.mode]
    train_loader = data.DataLoader(Cloud_Data(opts.root_dataset,
                                               image_set=modes[0],
                                               backbone=opts.backbone,
                                               transform=True),
                                   batch_size=1,
                                   shuffle=True,
                                   **kwargs)
    val_loader = data.DataLoader(Cloud_Data(opts.root_dataset,
                                             image_set=modes[1],
                                             backbone=opts.backbone,
                                             transform=True),
                                 batch_size=1,
                                 shuffle=False,
                                 **kwargs)
    return train_loader, val_loader
