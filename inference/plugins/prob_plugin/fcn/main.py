import os
import os.path as osp

import torch
from torchvision import transforms
from PIL import Image

from importlib import import_module

import numpy as np

class FCN_Main:
    def __init__(self, args):
        self.args = args

        model_module = import_module('fcn.models.{}.fcn{}'.format(args['backbone'], args['fcn']))
        self.model = model_module.FCN(n_class=args['n_classes'])

        #print(args['model'])
        if torch.cuda.is_available():
            checkpoint = torch.load(args['model'])
        else:
            checkpoint = torch.load(args['model'], map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if args['save'] == 'True':
            if not osp.exists(args['output']):
                os.makedirs(args['output'])
            if not os.path.exists(args['score']):
                os.makedirs(args['score'])

        # create a color pallette, selecting a color for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.colors = torch.as_tensor([i for i in range(args['n_classes'])])[:, None] * self.palette
        self.colors = (self.colors % 255).numpy().astype("uint8")


    def run(self, file_path):
        input_image = Image.open(file_path)
        input_image = input_image.resize((self.args['resize'], self.args['resize']))
        preprocess = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
           input_batch = input_batch.to('cuda')
           self.model.to('cuda')
        with torch.no_grad():
            output = self.model(input_batch)[0]

        output_predictions = output.argmax(0)
        #print('fcn', output[1].shape)

        scores = output[1].detach().cpu().numpy().reshape(-1)
        maxs = max(scores)
        mins = min(scores)
        scores = [(i-mins)/(maxs-mins) for i in scores]
        
        returning_scores = scores

        if self.args['save'] == 'True':
            pathsplit = file_path.split('/')
            output_path = ('{}/OUT_{}'.
                            format(self.args['score'],
                            pathsplit[-1].split('.')[0] + '.txt'))
            np.savetxt(output_path, scores, delimiter=',')

            output_path = ('{}/OUT_{}'.format(self.args['output'], pathsplit[-1]))
            # plot the semantic segmentation predictions of 21 classes in each color
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(self.colors)
            r.convert('RGB').save(output_path)

        return returning_scores
