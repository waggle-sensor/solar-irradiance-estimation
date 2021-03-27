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

from multiprocessing import Pool, Queue, Manager

import datetime

def preprocessing(args):
    deeplabpath, fcnpath, unetpath, plspath, lblpath, imagepath, resizing, mode = args

    print(imagepath)
    if ':' in imagepath:
        # print(deeplabpath.split('/')[-1].split('T_')[-1].split('.')[0])
        # print(fcnpath.split('/')[-1].split('T_')[-1].split('.')[0])
        # print(unetpath.split('/')[-1].split('T_')[-1].split('.')[0])
        # print(plspath.split('/')[-1].split('T_')[-1].split('.')[0])
        # print(lblpath.split('/')[-1].split('t_')[-1].split('.')[0])
        # print(imagepath.split('/')[-1].split('.')[0])

        if deeplabpath.split('/')[-1].split('T_')[-1].split('.')[0] != fcnpath.split('/')[-1].split('T_')[-1].split('.')[0] or \
           fcnpath.split('/')[-1].split('T_')[-1].split('.')[0] != unetpath.split('/')[-1].split('T_')[-1].split('.')[0] or \
           unetpath.split('/')[-1].split('T_')[-1].split('.')[0] != plspath.split('/')[-1].split('T_')[-1].split('.')[0] or \
           plspath.split('/')[-1].split('T_')[-1].split('.')[0] != lblpath.split('/')[-1].split('t_')[-1].split('.')[0] or \
           lblpath.split('/')[-1].split('t_')[-1].split('.')[0] != imagepath.split('/')[-1].split('.')[0] or \
           imagepath.split('/')[-1].split('.')[0] != deeplabpath.split('/')[-1].split('T_')[-1].split('.')[0]:
           # print(deeplabpath, fcnpath, unetpath, plspath, lblpath, imagepath)
           print('inputs do not match')
           exit(1)

    else:
        # print(deeplabpath.split('/')[-1].split('_')[-1])
        # print(fcnpath.split('/')[-1].split('_')[-1])
        # print(unetpath.split('/')[-1].split('_')[-1])
        # print(plspath.split('/')[-1].split('_')[-1])
        # print(lblpath.split('/')[-1].split('_')[0] + '.txt')
        # print(imagepath.split('/')[-1].split('.')[0] + '.txt')

        if deeplabpath.split('/')[-1].split('_')[-1] != fcnpath.split('/')[-1].split('_')[-1] or \
           fcnpath.split('/')[-1].split('_')[-1] != unetpath.split('/')[-1].split('_')[-1] or \
           unetpath.split('/')[-1].split('_')[-1] != plspath.split('/')[-1].split('_')[-1] or \
           plspath.split('/')[-1].split('_')[-1] != lblpath.split('/')[-1].split('_')[0] + '.txt' or \
           lblpath.split('/')[-1].split('_')[0] + '.txt' != imagepath.split('/')[-1].split('.')[0] + '.txt' or \
           imagepath.split('/')[-1].split('.')[0] + '.txt' != plspath.split('/')[-1].split('_')[-1]:

           # print(deeplabpath, fcnpath, unetpath, plspath, lblpath, imagepath)
           print('inputs do not match')
           exit(1)

    # Create the dataset
    import time

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


    if mode == 'val':
        return np.vstack((deeplab, fcn, unet, pls)).T

    lbl = PIL.Image.open(lblpath).convert('L')
    lbl = np.asarray(lbl)
    lbl = np.resize(lbl, (resizing,resizing)).reshape(-1)

    lbl[lbl > 0] = 1

    X = np.vstack((deeplab, fcn, unet, pls)).T
    y = lbl

    return (X, y)



def adaboost_regression(X, y, maxdepth, nestimators):
    # Fit regression model
    rng = np.random.RandomState(1)
    # regr_1 = DecisionTreeRegressor(max_depth=maxdepth)

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=maxdepth),
                              n_estimators=nestimators, random_state=rng)

    # regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    # y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)

    model_file = 'AdaBoost_model_norm_wo_entropy.pkl'
    pickle.dump(regr_2, open(model_file, 'wb'))


def evaluation(deeplab, fcn, unet, pls, lbl, image, args):
    adaboost = pickle.load(open(args.model, 'rb'))
    print('model loaded')


    for i in range(len(fcn)):
        X = preprocessing(deeplab[i], fcn[i], unet[i], pls[i], lbl, image[i], args.resize, args.mode)
        score = adaboost.predict(X)
        #print('>>>>>> min :', np.min(Y_pred), 'max : ', np.max(Y_pred))
        # print("Y_pred : {}".format(Y_pred))

        # print(score.shape)

        cloud = (np.asarray(score) > args.thr).sum()
        ratio = cloud / (args.resize * args.resize)
        # print(deeplab[i], ratio)


        score [score <= args.thr] = 0
        score [score > args.thr] = 255

        # print(score.shape)


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

    parser.add_argument('--cpus', type=int, required=True)

    parser.add_argument('--mode', type=str, help='running mode', choices=['train', 'val'])

    parser.add_argument('--deeplab', type=str, help='path to deeplab scores')
    parser.add_argument('--fcn', type=str, help='path to fcn scores')
    parser.add_argument('--unet', type=str, help='path to unet scores')
    parser.add_argument('--pls', type=str, help='path to pls scores')
    parser.add_argument('--label', type=str, help='path to labels')
    parser.add_argument('--image', type=str, help='path to images')

    parser.add_argument('--val', type=str, help='number of val data', choices=['single', 'multiple'])
    parser.add_argument('--model', type=str, help='path to a model')
    parser.add_argument('--thr', type=str, default=0.5, help='threshold')
    parser.add_argument('--output', type=str, default='output', help='output path')

    parser.add_argument('--resize', type=int, default=300, help='resize image')
    parser.add_argument('--max_depth', type=int, default=4, help='max depth of tree')
    parser.add_argument('--n_estimators', type=int, default=200, help='number of estimators')

    args = parser.parse_args()

    if args.mode == None or args.image == None:
        parser.print_help()
        exit(0)

    if args.mode == 'train':
        deeplablist = sorted(glob.glob(os.path.join(args.deeplab, '*')), reverse=True)
        fcnlist = sorted(glob.glob(os.path.join(args.fcn, '*')), reverse=True)
        unetlist = sorted(glob.glob(os.path.join(args.unet, '*')), reverse=True)
        plslist = sorted(glob.glob(os.path.join(args.pls, '*')), reverse=True)
        lbllist = sorted(glob.glob(os.path.join(args.label, '*')), reverse=True)
        imagelist = sorted(glob.glob(os.path.join(args.image, '*')), reverse=True)


        print('d', len(deeplablist), ',f', len(fcnlist), ',u', len(unetlist), ',p', len(plslist),
             ',l', len(lbllist), ',i', len(imagelist))


        X_seq = np.array([])
        y_seq = np.array([])


        with Pool(processes=args.cpus) as pool:
            _args = [(deeplablist[i], fcnlist[i], unetlist[i], plslist[i], lbllist[i], imagelist[i], args.resize, args.mode) for i in range(len(fcnlist))]
            res = pool.map(preprocessing, _args)
            print('len(res)', len(res))
            #count = 0
            print(datetime.datetime.now())
            for i in res:
                #count += 1
                #print(count)
                X, y = i
                if( len(X_seq) == 0 ):
                    X_seq = X
                    y_seq = y
                else:
                    X_seq = np.concatenate((X_seq, X), axis=0)
                    y_seq = np.concatenate((y_seq, y), axis=0)

        print(datetime.datetime.now())
        # count = 0
        # for i in range(len(fcnlist)):
        #     count += 1
        #     print(count)
        #     X, y = preprocessing(deeplablist[i], fcnlist[i], unetlist[i], plslist[i], lbllist[i], imagelist[i], args.resize, args.mode)
        #     if( len(X_seq) == 0 ):
        #         X_seq = X
        #         y_seq = y
        #     else:
        #         X_seq = np.concatenate((X_seq, X), axis=0)
        #         y_seq = np.concatenate((y_seq, y), axis=0)

        print('start regression')
        print(datetime.datetime.now())
        adaboost_regression(X_seq, y_seq, args.max_depth, args.n_estimators)
        print(datetime.datetime.now())


    if args.mode == 'val':

        deeplab = []
        fcn = []
        unet = []
        pls = []
        image = []

        if args.val == 'single':
            deeplab.append(args.deeplab)
            fcn.append(args.deeplab)
            unet.append(args.deeplab)
            pls.append(args.deeplab)
            image.append(args.image)
        elif args.val == 'multiple':
            name = args.deeplab + '*'
            deeplab = sorted(glob.glob(name))
            name = args.fcn + '*'
            fcn = sorted(glob.glob(name))
            name = args.unet + '*'
            unet = sorted(glob.glob(name))
            name = args.pls + '*'
            pls = sorted(glob.glob(name))
            name = args.image + '*.jpg'
            image = sorted(glob.glob(name))

        print(len(deeplab), len(fcn), len(unet), len(pls), len(image))


        print('evaluation')
        evaluation(deeplab, fcn, unet, pls, label, image, args)


