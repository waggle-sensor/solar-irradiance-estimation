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


class AdaBoost_Main:
    def __init__(self, args):
        self.args = args
        self.adaboost = pickle.load(open(args['model'], 'rb'))

        if args['save'] == 'True':
            if not os.path.exists(args['score']):
                os.makedirs(args['score'])
            if not os.path.exists(args['output']):
                os.makedirs(args['output'])

    def evaluation(self, deeplab, fcn, unet, pls, image):

        X = np.vstack((deeplab, fcn, unet, pls)).T
        score = self.adaboost.predict(X)
        returning_score = score

        test = score.reshape(-1, 300)

        if self.args['save'] == 'True':
            pathsplit = image[i].split('/')[-1].split('.')[0] + '.txt'
            output_path = ('{}/ADA_{}'.format(self.args['score'], pathsplit))
            np.savetxt(output_path, score, delimiter=',')

            
            score [score <= args['thr']] = 0
            score [score > args['thr']] = 255

            name = image[i].split('/')[-1]
            output_path = ('{}/ADA_{}'.format(args['output'], name))
            test = PIL.Image.fromarray(test).convert('RGB')
            test.save(output_path)


        return returning_score
