#!/usr/bin/python3
from PIL import Image
import numpy as np

import cv2
import pickle

import os
import os.path as osp

class PLS_Main:
    def __init__(self, args):

        self.args = args
        self.net = pickle.load(open(args['model'], 'rb'))

        if not osp.exists(args['output']):
            os.makedirs(args['output'])
        if not osp.exists(args['score']):
            os.makedirs(args['score'])


    def generate_features(self, path, resizing):

        image = cv2.imread(path)
        image = cv2.resize(image, (resizing, resizing))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        r = image[:,:,2].astype('int16')
        b = image[:,:,0].astype('int16')
        g = image[:,:,1].astype('int16')

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        s = hsv[:,:,1].astype('int16')


        for i in range(len(b)):
            for j in range(len(b[0])):
                if b[i][j] == 0:
                    b[i][j] = 1

        rob = r/b
        rsb = r-b
        rbratio = (b-r)/(b+r)

        mix = np.hstack((np.transpose(np.matrix(s.flatten())), np.transpose(np.matrix(rob.flatten())), np.transpose(np.matrix(rbratio.flatten()))))

        return mix


    def run(self, image_path):

        test_feature = self.generate_features(image_path, self.args['resize'])
        score = self.net.predict(test_feature)[:,0]

        #### mask
        for i in range(len(score)):
            if score[i] > self.args['threshold']:
                score[i] = True
            else:
                score[i] = False
        returning_score = score

        ######### save output image and scores
        if self.args['save'] == "True":
            ## save score
            pathsplit = image_path.split('/')
            output_path = ('{}/OUT_{}'.format(self.args['score'], pathsplit[-1].split('.')[0] + '.txt'))
            np.savetxt(output_path, score, delimiter=',')

            ## save scores as an image
            score [score == False] = 0
            score [score == True] = 255

            predicted = score.T
            score = np.reshape(score, (300,300))

            pathsplit = image_path.split('/')
            output_path = ('{}OUT_{}'.format(self.args['output'], pathsplit[-1]))

            result = Image.fromarray(score.astype(np.uint8))
            result.save(output_path)

        return returning_score

