import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from dataset import BasicDataset

import glob

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_output_filenames(in_files, args):
    out_files = []

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for f in in_files:
        pathsplit = f.split('/')
        out_files.append('{}OUT_{}'.format(args.output, pathsplit[-1]))

    return out_files



'''
    ('-d', '--mode', dest='mode', type=str, default='train', help='Mode to run the U-Net, train or infer')
    ('--n_classes', type=int, default=1, help='Number of classes in the segmentation')
    ('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    ('-m', '--model', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')
    ('-i', '--input', metavar='INPUT', default='/data/test/', help='Folder to read input images', required=True)
    ('-o', '--output', metavar='OUTPUT', default='/data/output/', help='Folder to save ouput images')
    ('-n', '--no-save', action='store_true', default=False, help='Do not save the output masks')
    ('-t', '--mask-threshold', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
'''

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    in_files = glob.glob(args.input + 'images/*')
    if len(in_files) < 1:
        raise Exception('There is no images in /data/test/images/')

    out_files = get_output_filenames(in_files, args)

    net = UNet(n_channels=3, n_classes=args.n_classes)

    if args.model == None:
        raise Exception('Path to model is necessary')
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = Image.fromarray((mask * 255).astype(np.uint8))
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))
