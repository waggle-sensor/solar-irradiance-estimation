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

import waggle.plugin as plugin
import logging

logging.basicConfig(level=logging.DEBUG)

plugin.init()

TOPIC_INPUT_IMAGE = "sky_image"

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


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fcn_config', type=str, help='path to fcn configuration list')
    parser.add_argument('--unet_config', type=str, help='path to unet configuration list')
    parser.add_argument('--pls_config', type=str, help='path to plsregression configuration list')
    parser.add_argument('--deeplab_config', type=str, help='path to deeplab configuration list')
    parser.add_argument('--adaboost_config', type=str, help='path to adaboost configuration list')

    args = parser.parse_args()

    if args.fcn_config == None and args.unet_config == None and args.pls_config == None and args.deeplab_config == None:
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

    with open_data_source(id=TOPIC_INPUT_IMAGE) as cap:
        timestamp, image = cap.get()

        if args.fcn_config != None:
            fcn = fcn_main.run(image)
        if args.pls_config != None:
            pls = pls_main.run(image)
        if args.unet_config != None:
            unet = unet_main.run(image)
        if args.deeplab_config != None:
            deeplab = deeplab_main.run(image)
        if args.adaboost_config != None:
            adaboost = adaboost_main.evaluation(deeplab, fcn, unet, pls, image)

        def counting(inarray, thr):
            count = 0
            for i in inarray:
                if i > thr:
                    count += 1
            return round(count/(300*300), 2)

        if opts.fcn_cfg['result'] == 'send':
            plugin.publish('env.cloud_cover.fcn', counting(fcn, 0.3), timestamp=timestamp)
        if opts.unet_cfg['result'] == 'send':
            plugin.publish('env.cloud_cover.unet', counting(unet, 0.5), timestamp=timestamp)
        if opts.pls_cfg['result'] == 'send':
            plugin.publish('env.cloud_cover.pls', counting(pls, 0.5), timestamp=timestamp)
        if opts.deeplab_cfg['result'] == 'send':
            plugin.publish('env.cloud_cover.deeplab', counting(deeplab, 0.5), timestamp=timestamp)
        if opts.adaboost_cfg['result'] == 'send':
            plugin.publish('env.cloud_cover.adaboost', counting(adaboost), timestamp=timestamp)

