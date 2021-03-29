import json
import argparse
from types import SimpleNamespace

from fcn.main import FCN_Main
from pls.main import PLS_Main
from unet.main import Unet_Main
from deeplab.main import ASPP_Main
from adaboost.main import AdaBoost_Main

import glob

import numpy as np
from PIL import Image
import pickle


def load_config(args):
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

    if args.deeplab_config != None:
        with open(args.deeplab_config, 'r') as f:
            deeplab_configurations = json.load(f)
        opts.deeplab_cfg = deeplab_configurations
        print('DeepLab loaded')

    if args.adaboost_config != None:
        with open(args.adaboost_config, 'r') as f:
            adaboost_configurations = json.load(f)
        opts.adaboost_cfg = adaboost_configurations
        print('AdaBoost loaded')

    return opts


def load_inputs(args):
    input_images = []
    labels = []

    if args.multiple_images != None:
        args.multiple_images = args.multiple_images + '*'
        input_images = glob.glob(args.multiple_images)
        input_images = sorted(input_images)

        if args.multiple_labels != None:
            args.multiple_labels = args.multiple_labels + '*'
            labels = glob.glob(args.multiple_labels)
            labels = sorted(labels)

    elif args.input_image != None:
        input_images.append(args.input_image)
        if args.input_label != None:
            labels.append(args.input_label)
    else:
        raise Exception('input image path is not provided')

#     print(len(input_images), len(labels))
    return input_images, labels



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fcn_config', type=str, help='path to fcn configuration list')
    parser.add_argument('--unet_config', type=str, help='path to unet configuration list')
    parser.add_argument('--pls_config', type=str, help='path to plsregression configuration list')
    parser.add_argument('--deeplab_config', type=str, help='path to deeplab configuration list')
    parser.add_argument('--adaboost_config', type=str, help='path to adaboost configuration list')

    parser.add_argument('--single_image', type=str, help='path to an input image')
    parser.add_argument('--single_label', type=str, help='path to an input label')
    parser.add_argument('--multiple_images', type=str, help='path to an input folder')
    parser.add_argument('--multiple_labels', type=str, help='y/n if there are labels for the massive images')
    
    args = parser.parse_args()

    if args.input_image != None and args.input_label == None:
        print('[warning] path to the label is not provided')

    if args.fcn_config == None and args.unet_config == None and args.pls_config == None and args.deeplab.config == None:
        print('[Error] None of configuration is provided')
        parser.print_help()
        exit(0)


    ## read configuration files
    opts = load_config(args)

    ## load classes
    if args.fcn_config != None:
        fcn_main = FCN_Main(opts.fcn_cfg)
    if args.pls_config != None:
        pls_main = PLS_Main(opts.pls_cfg)
    if args.unet_config != None:
        unet_main = Unet_Main(opts.unet_cfg)
    if args.deeplab_config != None:
        deeplab_main = ASPP_Main(opts.deeplab_cfg)
    if args.adaboost_config != None:
        adaboost_main = AdaBoost_Main(opts.adaboost_cfg)

    ## load images and labels
    input_images, labels = load_inputs(args)

    count = 0
    #for input_image in input_images:
    for i in range(len(input_images)):
        count += 1
        print(count)

        if args.fcn_config != None:
            fcn = fcn_main.run(input_images[i])
        if args.pls_config != None:
            pls = pls_main.run(input_images[i])
        if args.unet_config != None:
            unet = unet_main.run(input_images[i])
        if args.deeplab_config != None:
            deeplab = deeplab_main.run(input_images[i])
        if args.adaboost_config != None:
            adaboost = adaboost_main.evaluation(deeplab, fcn, unet, pls, input_images[i])



