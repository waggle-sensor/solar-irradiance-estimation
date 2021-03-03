#!/usr/bin/python3
import os
import glob
import argparse

import PIL
from PIL import Image
import pickle

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk

from multiprocessing import Pool, Queue, Manager

import time

def preprocessing(deeplabpath, fcnpath, unetpath, plspath, imagepath, resizing):

    #print(deeplabpath.split('/')[-1].split('.')[0].split('_')[-1])
    #print(fcnpath.split('/')[-1].split('.')[0].split('_')[-1])
    #print(unetpath.split('/')[-1].split('.')[0].split('_')[-1])
    #print(imagepath.split('/')[-1].split('.')[0].split('_')[-1])
    #print(plspath.split('/')[-1].split('.')[0].split('_')[-1])

    if deeplabpath.split('/')[-1].split('.')[0].split('_')[-1] != fcnpath.split('/')[-1].split('.')[0].split('_')[-1] or \
       fcnpath.split('/')[-1].split('.')[0].split('_')[-1] != unetpath.split('/')[-1].split('.')[0].split('_')[-1] or \
       unetpath.split('/')[-1].split('.')[0].split('_')[-1] != imagepath.split('/')[-1].split('.')[0].split('_')[-1] or \
       imagepath.split('/')[-1].split('.')[0].split('_')[-1] != plspath.split('/')[-1].split('.')[0].split('_')[-1]:

       print(deeplabpath, fcnpath, unetpath, plspath, imagepath)
       print('inputs do not match')
       exit(1)

    print(imagepath)

    unet = []
    fcn = []
    deeplab = []
    pls = []
    with open(unetpath, 'r') as f:
        for line in f:
            unet.append(float(line.strip()))
    with open(fcnpath, 'r') as f:
        for line in f:
            fcn.append(int(float(line.strip())))
    with open(deeplabpath, 'r') as f:
        for line in f:
            deeplab.append(float(line.strip()))
    with open(plspath, 'r') as f:
        for line in f:
            test = float(line.strip().split('e')[0])
            if '-' in line.strip().split('e')[-1]:
                #print(line.strip().split('e')[0], line.strip().split('-')[-1])
                power = int(line.strip().split('-')[-1])
                test = test*10**(-power)
            elif '+' in line.strip().split('e')[-1]:
                #print(line.strip().split('e')[0], line.strip().split('+')[-1])
                power = int(line.strip().split('+')[-1])
                test = test * 10**(power)

            pls.append(test)

    # print(min(fcn), max(fcn))
    # print(min(unet), max(unet))
    # print(min(pls), max(pls))
    # print(min(deeplab), max(deeplab))

    # print('fcn', len(fcn), type(fcn))
    # print('unet', len(unet), type(unet))
    # print('pls', len(pls), type(pls))
    # print('deeplab', len(deeplab), type(deeplab))

    # grey = cv2.imread(imagepath, 0)
    # grey = cv2.resize(grey, (300,300))
    # E = entropy(grey, disk(10))
    # E = E.reshape(-1)

    return np.vstack((deeplab, fcn, unet, pls)).T



def evaluation(deeplab, fcn, unet, pls, image, args):
    adaboost = pickle.load(open(args.model, 'rb'))
    print('model loaded')

    count = 0
    for i in range(len(fcn)):
        count += 1
        print(count)
        X = preprocessing(deeplab[i], fcn[i], unet[i], pls[i], image[i], args.resize)
        score = adaboost.predict(X)
        #print('>>>>>> min :', np.min(Y_pred), 'max : ', np.max(Y_pred))
        # print("Y_pred : {}".format(Y_pred))


        if not os.path.exists(args.score):
            os.makedirs(args.score)
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        pathsplit = image[i].split('/')[-1].split('.')[0] + '.txt'
        output_path = ('{}/ADA_{}'.format(args.score,pathsplit))

        np.savetxt(output_path, score, delimiter=',')



        cloud = (np.asarray(score) > args.thr).sum()
        ratio = cloud / (args.resize * args.resize)
        #print(deeplab[i], ratio)



        #### mask
        for j in range(len(score)):
            if score[j] > args.thr:
                score[j] = True
            else:
                score[j] = False
        #### end

        cloud = 0
        for j in score:
            if j == 1:
                cloud += 1
        ratio = cloud/len(score)
        print('ada_wo_entropy', ratio)


        score [score <= args.thr] = 0
        score [score > args.thr] = 255


        test = score.reshape(-1, 300)
        # # print(test.shape)
        # test = np.array(test, dtype=np.uint8)
        # halo = {}
        # for j in score:
        #     if j not in halo:
        #         halo[j] = 1
        #     else:
        #         halo[j] += 1
        # print(halo)

        name = image[i].split('/')[-1]
        output_path = ('{}/ADA_{}'.format(args.output, name))
        test = PIL.Image.fromarray(test).convert('RGB')
        test.save(output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some input arguments')

    parser.add_argument('--deeplab', type=str, help='path to deeplab scores')
    parser.add_argument('--fcn', type=str, help='path to fcn scores')
    parser.add_argument('--unet', type=str, help='path to unet scores')
    parser.add_argument('--pls', type=str, help='path to pls scores')
    parser.add_argument('--label', type=str, help='path to labels')
    parser.add_argument('--image', type=str, help='path to images')

    parser.add_argument('--run', type=str, help='number of val data', choices=['single', 'multiple'])
    parser.add_argument('--model', type=str, help='path to a model')
    parser.add_argument('--thr', type=float, default=0.9, help='threshold')

    parser.add_argument('--output', type=str, default='output', help='output path for images')
    parser.add_argument('--score', type=str, default='score', help='output path for scores')

    parser.add_argument('--resize', type=int, default=300, help='resize image')
    parser.add_argument('--max_depth', type=int, default=4, help='max depth of tree')
    parser.add_argument('--n_estimators', type=int, default=200, help='number of estimators')

    parser.add_argument('--option', type=str)

    args = parser.parse_args()

    if args.image == None:
        parser.print_help()
        exit(0)


    deeplab = []
    fcn = []
    unet = []
    pls = []
    image = []

    if args.run == 'single':
        deeplab.append(args.deeplab)
        fcn.append(args.deeplab)
        unet.append(args.deeplab)
        pls.append(args.deeplab)
        image.append(args.image)
    elif args.run == 'multiple':
        name = args.deeplab + '*'
        deeplab = sorted(glob.glob(name))
        name = args.fcn + '*'
        fcn = sorted(glob.glob(name))
        name = args.unet + '*'
        unet = sorted(glob.glob(name))
        name = args.pls + '*'
        pls = sorted(glob.glob(name))
        name = args.image + '*'
        image = sorted(glob.glob(name))

    print('first:', len(deeplab), len(fcn), len(unet), len(pls), len(image))


    print('evaluation')
    if args.option == 'june':
        print('data from 484 node are in processing')
        nd = []
        nf = []
        nu = []
        npl = []

        line = image[0].split('/')[-1].split('T')[0]
        print(line)
        for i in range(len(deeplab)):
            if line in deeplab[i]:
                nd.append(deeplab[i])
                nf.append(fcn[i])
                nu.append(unet[i])
                npl.append(pls[i])

        print(len(nd), len(nf), len(nu), len(npl), len(image))
        evaluation(nd, nf, nu, npl, image, args)
    else:
        evaluation(deeplab, fcn, unet, pls, image, args)


