import torch
import torch.nn as nn
from torchvision import transforms

import network
import utils

import os
from PIL import Image
import numpy as np
import argparse
import glob

class DeepLab:
    def __init__(self, args):
        opts = vars(args)
        print(opts)
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }



        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model_map[opts['model']](num_classes=opts['num_classes'], output_stride=opts['output_stride'])

        if opts['separable_conv'] == 'True' and 'plus' in opts['model']:
            network.convert_to_separable_conv(model.classifier)

        def set_bn_momentum(model, momentum=0.1):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum

        set_bn_momentum(self.model.backbone, momentum=0.01)

        checkpoint = torch.load(opts['ckpt'], map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()


        if not os.path.exists(opts['output']):
            os.makedirs(opts['output'])


        # create a color pallette, selecting a color for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.colors = torch.as_tensor([i for i in range(opts['num_classes'])])[:, None] * self.palette
        self.colors = (self.colors % 255).numpy().astype("uint8")



    def run(self, input_path):
        """Do validation and return specified samples"""
        input_image = Image.open(input_path)
        input_image = input_image.resize((self.opts['resize'], self.opts['resize']))


        preprocess = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ])
        input_tensor = preprocess(input_image)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        #print(input_tensor.shape)


        input_image = input_tensor.to(self.device, dtype=torch.float32)
        score = self.model(input_image)
        output = score[0]
        output_predictions = output.argmax(0)


        pathsplit = input_path.split('/')
        output_path = ('{}/OUT_{}'.format(self.opts['output'], pathsplit[-1]))

        # plot the semantic segmentation predictions of 21 classes in each color
        #r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r = Image.fromarray(output_predictions.byte().cpu().numpy())
        r.putpalette(self.colors)
        r.convert('RGB').save(output_path)




        ## calculate ratio
        np_output = np.asarray(r)

        cloud = 0
        for i in range(len(np_output)):
            for j in range(len(np_output[0])):
                if np_output[i][j] == 1:  # cloud
                    cloud += 1

        total = np_output.shape[0] * np_output.shape[1]
        ratio = round((cloud / total), 5)

        '''
        ## Calculate Intersection over Union (IoU)
        #np_label = np.resize(np_label, (np_output.shape[0], np_output.shape[1]))

        iou = 0
        for i in range(len(np_label)):
            for j in range(len(np_label[0])):
               if np_label[i][j] > 0 and np_output[i][j] == 1:
                   iou += 1

        iou_cal = round((iou / cloud), 5)

        #print(cloud, iou)

        return ratio, iou_cal
        '''
        print(ratio)
        return ratio

