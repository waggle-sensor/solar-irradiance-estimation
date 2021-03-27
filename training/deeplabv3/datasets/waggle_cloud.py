from torch.utils import data
import torch
import numpy as np

import PIL

import glob
from random import sample

import os

class WaggleSegmentation(data.Dataset):
    def __init__(self, opts, image_set='train', transform=False):
        self.opts = opts

        self._transform = transform
        self.image_set = image_set
        self.image_type = opts.dataset

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])


        ## images  labels
        if self.image_type == 'waggle_cloud':
            image_dir = os.path.join(opts.input_path, 'images/*')
            label_dir = os.path.join(opts.input_path, 'labels/*')

            image = glob.glob(image_dir)
            lbl = glob.glob(label_dir)
        elif self.image_type == 'voc':
            import shutil

            image_dir = os.path.dirname(opts.input_file).split('I')[0] + 'JPEGImages/'
            label_dir = os.path.dirname(opts.input_file).split('I')[0] + 'SegmentationClass/'

            image = []
            lbl = []
            with open(opts.input_file, 'r') as f:
                for line in f:
                    image.append(os.path.join(image_dir, line.strip() + '.jpg'))
                    lbl.append(os.path.join(label_dir, line.strip() + '.png'))
        elif self.image_type == 'cityscape':
            print('under construction')
            exit(1)


        tnum = [i for i in range(len(image))]
        tdata = int(len(image)*0.8)
        tsample = sample(tnum, tdata)

        train = []
        val = []
        for i in tnum:
            if i in tsample:
                train.append(image[i])
            else:
                val.append(image[i])


        self.files = {'train': train, 'val': val, 'lbl':lbl}

    def __len__(self):
        return len(self.files[self.image_set])

    def __getitem__(self, index):
        data_file = self.files[self.image_set][index]
        #load image
        img_file = data_file
        img = PIL.Image.open(img_file)
        if self.opts.resize != None:
            img = img.resize((self.opts.resize, self.opts.resize))
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
        if self.opts.resize != None:
            lbl = lbl.resize((self.opts.resize, self.opts.resize))
        lbl = np.array(lbl, dtype=np.int32)


        if self.image_type == 'waggle_cloud':
            lbl[lbl <= 175] = 0
            lbl[lbl > 175] = 1
        elif self.image_type == 'voc':
            lbl[lbl == 255] = -1
        else:
            raise Exception('data_loader for this image type is not ready')


        '''
        ### check mask images when training
        pathsplit = lbl_file.split('/')[-1].split('.')
        output_path = ('{}OUT_{}.png'.format('./output/mask_training/', pathsplit[0]))

        #print(pathsplit, output_path)

        result = PIL.Image.fromarray(lbl * 255).convert('RGB')
        result.save(output_path)
        #### end
        '''

        if self._transform:
            #print('transform', img.shape, lbl.shape)
            return self.transform(img, lbl)
        else:
            #print('untransform', img.shape, lbl.shape)
            return self.untransform(img, lbl)
            #return img, lbl


    def transform(self, img, lbl):
        img = np.array(img, dtype=np.float64)
        img /= 255.
        lbl = np.array(lbl, dtype=np.float64)
        lbl[lbl == 255] = -1  # Ignore contour
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()


        return img, lbl

    def untransform(self, img, lbl):
        img = np.array(img, dtype=np.float64)
        lbl = np.array(lbl, dtype=np.float64)
        lbl = np.array(lbl, dtype=np.int8)

        img = img.transpose(2, 1, 0)


        #img = img.transpose(1, 2, 0)
        #img *= self.std
        #img += self.mean

        #img *= 255
        #img = img.astype(np.uint8)
        #lbl = lbl.numpy()

        return img, lbl


