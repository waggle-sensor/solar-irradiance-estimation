import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .unet import UNet


class Unet_Main:
    def __init__(self, args):
        self.args = args

        self.net = UNet(n_channels=3, n_classes=args['n_classes'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(args['model'], map_location=self.device))

        if not os.path.exists(args['output']):
            os.makedirs(args['output'])
        if not os.path.exists(args['score']):
            os.makedirs(args['score'])

    def preprocess(self, pil_img, scale, n_classes=2):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        newW, newH = 300, 300
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)


        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans.astype(float)



    def predict_img(self,
                    pathsplit, full_img,
                    scale_factor=1,
                    out_threshold=0.5):

        self.net.eval()
        img = torch.from_numpy(self.preprocess(full_img, scale_factor))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)


        with torch.no_grad():
            output = self.net(img)

            probs = torch.sigmoid(output)
            probs = probs.squeeze(0)

            scores = probs.detach().cpu().numpy().reshape(-1)
            maxs = max(scores)
            mins = min(scores)
            scores = [(i-mins)/(maxs-mins) for i in scores]

            ## save scores and images
            if self.args['save'] == 'True':
                output_path = ('{}/OUT_{}'.
                                format(self.args['score'],
                                pathsplit[-1].split('.')[0] + '.txt'))
                np.savetxt(output_path, scores, delimiter=',')


            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(full_img.size[1]),
                    transforms.ToTensor()
                ]
            )

        return scores


    def run(self, image_path, np_label):
        img = Image.open(image_path)
        mask = self.predict_img(pathsplit, full_img=img,
                                scale_factor=self.args['scale'],
                                out_threshold=self.args['threshold'])

        if self.args['save'] == 'True':
            pathsplit = image_path.split('/')
            out_file = ('{}OUT_{}'.format(self.args['output'], pathsplit[-1]))
            result = Image.fromarray((mask * 255).astype(np.uint8))
            result.save(out_file)

        return mask



