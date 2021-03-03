import json
import argparse
from types import SimpleNamespace

from fcn.main import FCN_Main
from pls.main import PLS_Main
from unet.main import Unet_Main

import glob

import numpy as np
from PIL import Image

import datetime

def ground_truth(label_path):
    label = Image.open(label_path).convert('L')
    np_label = np.asarray(label)

    cloud = 0
    for i in range(len(np_label)):
        for j in range(len(np_label[0])):
            if np_label[i][j] >= 175:  # white
                cloud += 1

    total = np_label.shape[0] * np_label.shape[1]
    ratio = round((cloud / total), 5)

    return ratio, np_label


def find_label(key):
    for i in labels:
        if key in i:
            return i


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-fc', '--fcn_config', type=str, help='path to fcn configuration list')
    parser.add_argument('-uc', '--unet_config', type=str, help='path to unet configuration list')
    parser.add_argument('-pc', '--pls_config', type=str, help='path to plsregression configuration list')
    parser.add_argument('-ii', '--input_image', type=str, help='path to an input image')
    parser.add_argument('-il', '--input_label', type=str, help='path to an input label')
    parser.add_argument('-im', '--massive', type=str, help='path to an input folder')
    parser.add_argument('-ml', '--massive_labels', type=str, help='y/n if there are labels for the massive images')
    parser.add_argument('--image_type', type=str, default='voc', help='segmentation class coloring type')

    args = parser.parse_args()

    if args.input_image != None and args.input_label == None:
        print('[warning]: path to the input label is not provided')

    if args.fcn_config == None and args.unet_config == None and args.pls_config == None:
        parser.print_help()
        exit(0)

    print(datetime.datetime.now())

    ## read configuration files
    opts = SimpleNamespace()

    if args.fcn_config != None:
        with open(args.fcn_config, 'r') as f:
            fcn_configurations = json.load(f)
        opts.fcn_cfg = fcn_configurations
        print('FCN loaded')

    if args.pls_config != None:
        with open(args.pls_config, 'r') as f:
            pls_configurations = json.load(f)
        opts.pls_cfg = pls_configurations
        print('PLS loaded')

    if args.unet_config !=None:
        with open(args.unet_config, 'r') as f:
            unet_configurations = json.load(f)
        opts.unet_cfg = unet_configurations
        print('Unet loaded')


    if args.input_image == None:
        args.massive_images = args.massive + '*'
        input_images = glob.glob(args.massive_images)
        input_images = sorted(input_images)


        labels = []
        if args.massive_labels != None:
            args.massive_labels = args.massive + '*'
            labels = glob.glob(args.massive_labels)
            labels = sorted(labels)

        elif args.massive_labels == None:
            labels = ['none']


        print(len(input_images), len(labels))

        if args.fcn_config != None:
            fcn_main = FCN_Main(opts.fcn_cfg)
        if args.pls_config != None:
            pls_main = PLS_Main(opts.pls_cfg)
        if args.unet_config != None:
            unet_main = Unet_Main(opts.unet_cfg)

        count = 0
        #for input_image in input_images:
        for i in range(len(input_images)):
            count += 1
            print(count)

            if len(labels) != 1:
                print(labels[i])
                ratio, np_label = ground_truth(labels[i])
            else:
                print(input_images[i])
                image = Image.open(input_images[0])
                np_image = np.asarray(image)
                ratio = 0
                np_label = np.zeros([np_image.shape[0], np_image.shape[1]])


            if args.fcn_config != None:
                fcn_ratio = fcn_main.run(input_images[i], np_label)
            if args.pls_config != None:
                pls_ratio = pls_main.run(input_images[i], np_label)
            if args.unet_config != None:
                unet_ratio = unet_main.run(input_images[i], np_label)

            #print('fcn', fcn_ratio, 'pls', pls_ratio, 'unet', unet_ratio)
            print('unet', unet_ratio)

        print(datetime.datetime.now())

    elif args.input_image != None:
        if args.input_label != None:
            ratio, np_label = ground_truth(args.input_label)
        else:
            image = Image.open(args.input_image)
            np_image = np.asarray(image)
            ratio = 0
            np_label = np.zeros([np_image.shape[0], np_image.shape[1]])

        print(type(np_label), np_label.shape)

        if args.fcn_config != None:
            fcn_main = FCN_Main(opts.fcn_cfg)
            fcn_ratio = fcn_main.run(args.input_image, np_label)

        if args.pls_config != None:
            pls_main = PLS_Main(opts.pls_cfg)
            pls_ratio = pls_main.run(args.input_image, np_label)

        if args.unet_config != None:
            unet_main = Unet_Main(opts.unet_cfg)
            unet_ratio = unet_main.run(args.input_image, np_label)


    else:
        raise Exception('input image path is not provided')


