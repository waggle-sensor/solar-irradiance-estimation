import torch
import torch.nn as nn

from torchvision import transforms

import deeplab.network
import deeplab.utils

import os

from PIL import Image

import numpy as np

import argparse
import glob


class ASPP_Main:
    def __init__(self, opts):

        self.denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])


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

        self.model = model_map[opts['model']](num_classes=opts['n_classes'], output_stride=opts['output_stride'])

        if opts['separable_conv'] == 'True' and 'plus' in opts['model']:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)

        checkpoint = torch.load(opts['checkpoint'], map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()


        if not os.path.exists(opts['output']):
            os.makedirs(opts['output'])
        if not os.path.exists(opts['score']):
            os.makedirs(opts['score'])


        # create a color pallette, selecting a color for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.colors = torch.as_tensor([i for i in range(opts['n_classes'])])[:, None] * self.palette
        self.colors = (self.colors % 255).numpy().astype("uint8")



    def run(self, input_path, label):
        # print(input_path)
        """Do validation and return specified samples"""
        input_image = Image.open(input_path)
        input_image = input_image.resize((self.opts['resize'], self.opts['resize']))


        preprocess = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ])
        input_tensor = preprocess(input_image)
        input_tensor = torch.unsqueeze(input_tensor, 0)

        input_image = input_tensor.to(self.device, dtype=torch.float32)
        score = self.model(input_image)

        output = score[0]
        output_predictions = output.argmax(0)

        scores = output_predictions.cpu().numpy().reshape(-1)
        returning_scores = scores

        if self.opts['save'] == 'True':
            pathsplit = input_path.split('/')[-1].split('.')[0] + '.txt'
            output_path = ('{}/OUT_{}'.format(self.opts['score'], pathsplit))
            #print(type(scores), scores.shape)
            np.savetxt(output_path, scores, delimiter=',')

            pathsplit = input_path.split('/')
            output_path = ('{}/OUT_{}'.format(self.opts['output'], pathsplit[-1]))

            # plot the semantic segmentation predictions of 21 classes in each color
            #r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r = Image.fromarray(output_predictions.byte().cpu().numpy())
            r.putpalette(self.colors)
            r.convert('RGB').save(output_path)

        return returning_scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_stride', type=int, default=16, help='output stride size', choices=[8, 16])
    parser.add_argument('--n_classes', type=int, default=21, help='number of classes')
    parser.add_argument('--model', type=str, help='base network name', choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                                                                'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                                                                'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'])
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--separable_conv', type=bool, default=False, help='use separable conv? yes or no, any parameter will be encoded as True')
    parser.add_argument('--resize', type=int, default=300, help='resize image size')

    parser.add_argument('--output', type=str, default='./output', help='base output dir for result images')
    parser.add_argument('--score', type=str, default='./score', help='base output dir for result scores')

    parser.add_argument('--input', type=str, help='input path')
    parser.add_argument('--save', type=str, default='False')

    args = parser.parse_args()
    opts = vars(args)


    if args.input == None:
        parser.print_help()
        exit(0)


    aspp = ASPP_Main(opts)
    if 'jpg' in args.input or 'png' in args.input or 'jpeg' in args.input:
        #print('a')
        aspp.run(args.input, None)
    elif 'txt' in args.input:
        import shutil
        #print('b')
        count = 0
        dir = os.path.dirname(args.input).split('I')[0] + 'JPEGImages/'
        print(dir)
        with open(args.input, 'r') as f:
            for line in f:
                count += 1
                path = os.path.join(dir, line.strip() + '.jpg')
                shutil.copy2(path, './images')
                aspp.run(path, None)
                if count == 100:
                    exit(0)
    else:
        inputs = glob.glob(os.path.join(args.input, '*'))
        inputs = sorted(inputs)
        print(len(inputs))

        count = 0
        for i in inputs:
            count += 1
            print(count, i)
            aspp.run(i, None)




