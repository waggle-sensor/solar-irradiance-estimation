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

        if not osp.exists(args['output']):
            os.makedirs(args['output'])

        self.net = pickle.load(open(args['model'], 'rb'))

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

        '''
        stand_rob = (rob-np.mean(rob))/np.std(rob)
        stand_rob = (stand_rob-np.mean(stand_rob))

        stand_rsb = (rsb-np.mean(rsb))/np.std(rsb)
        stand_rsb = (stand_rsb-np.mean(stand_rsb))

        stand_rbratio = (rbratio-np.mean(rbratio))/np.std(rbratio)
        stand_rbratio = (stand_rbratio-np.mean(stand_rbratio))

        stand_s = (s-np.mean(s))/np.std(s)
        stand_s = (stand_s-np.mean(stand_s))

        mix = np.hstack((np.transpose(np.matrix(stand_s.flatten())), np.transpose(np.matrix(stand_rob.flatten())), np.transpose(np.matrix(stand_rbratio.flatten()))))
        '''

        mix = np.hstack((np.transpose(np.matrix(s.flatten())), np.transpose(np.matrix(rob.flatten())), np.transpose(np.matrix(rbratio.flatten()))))
        return mix


    def run(self, image_path, np_label):

        test_feature = self.generate_features(image_path, self.args['resize'])
        score = self.net.predict(test_feature)[:,0]

        cloud = (score > self.args['threshold']).sum()
        cloud_perc = (cloud / (score.shape)) * 100


        maxs = max(score)
        mins = min(score)
        for i in range(len(score)):
            score[i] = (score[i]-mins)/(maxs-mins)

        ratio = 'nan'

        #print('pls', score.shape)

        '''
        #### mask
        for i in range(len(score)):
            if score[i] > self.args['threshold']:
                score[i] = True
            else:
                score[i] = False
        #### end
        '''
        ################## new!
        pathsplit = image_path.split('/')
        output_path = ('{}/OUT_{}'.
                        format(self.args['score'],
                        pathsplit[-1].split('.')[0] + '.txt'))
        np.savetxt(output_path, score, delimiter=',')
        ################### end
        '''
        cloud = 0
        for i in score:
            if i == 1:
                cloud += 1
        ratio = cloud/len(score)
        '''


        # pathsplit = image_path.split('/')
        # output_path = ('{}OUT_{}'.format(self.args['output'], pathsplit[-1]))

        #print('hi, this it output', output_path)

        # Imaging from predicted result
        #score [score <= self.args['threshold']] = 0
        #score [score > self.args['threshold']] = 1


        predicted = score.T

        #print(self.args['threshold'], type(self.args['threshold']))
        #score [score <= self.args['threshold']] = 0
        #score [score > self.args['threshold']] = 255

        notok = []
        what = {0:0, 255:0}
        hello = score
        #print(np.min(hello), np.max(hello))
        for i in hello:
           if i <= self.args['threshold']:
               notok.append(0)
               what[0] += 1
           else:
               notok.append(255)
               what[255] += 1
        #print(what)

        score = np.reshape(notok, (300,300))


        pathsplit = image_path.split('/')
        output_path = ('{}OUT_{}'.format(self.args['output'], pathsplit[-1]))

        result = Image.fromarray(score.astype(np.uint8))
        result.save(output_path)



        '''
        a = predicted[0:1]
        b = predicted[1:2]
        c = predicted[2:3]
        a = np.reshape(a, (self.args['resize'], self.args['resize']))
        b = np.reshape(b, (self.args['resize'], self.args['resize']))
        c = np.reshape(c, (self.args['resize'], self.args['resize']))

        a = np.array(a * 255, dtype = np.uint8)
        b = np.array(b * 255, dtype = np.uint8)
        c = np.array(c * 255, dtype = np.uint8)

        a_path = output_path.replace('.png', '_a.png')
        b_path = output_path.replace('.png', '_b.png')
        c_path = output_path.replace('.png', '_c.png')

        result = Image.fromarray(a)
        result.save(a_path)

        result = Image.fromarray(b.astype(np.uint8))
        result.save(b_path)

        result = Image.fromarray(c.astype(np.uint8))
        result.save(c_path)

        lista = [0,0,0,0,0,0,0,0,0,0,0,0]
        ## calculate ratio
        count = 0
        cloud = 0
        for i in range(len(a)):
            for j in range(len(a[0])):

                count += 1

                if a[i][j] < 10:
                    lista[0] += 1
                elif a[i][j] < 20:
                    lista[1] += 1
                elif a[i][j] < 30:
                    lista[2] += 1
                elif a[i][j] < 40:
                    lista[3] += 1
                elif a[i][j] < 50:
                    lista[4] += 1
                elif a[i][j] < 60:
                    lista[5] += 1
                elif a[i][j] < 70:
                    lista[6] += 1
                elif a[i][j] < 80:
                    lista[7] += 1
                elif a[i][j] < 90:
                    lista[8] += 1
                elif a[i][j] < 100:
                    lista[9] += 1
                elif a[i][j] < 250:
                    lista[10] += 1
                else:
                    lista[11] += 1


                if a[i][j] < 250:  # cloud  ## 255 = white = sky
                    cloud += 1

        #print(cloud, count)
        #print(lista)

        total = a.shape[0] * a.shape[1]
        ratio = round((cloud / total), 5)

        ## Calculate Intersection over Union (IoU)
        np_label = np.resize(np_label, (a.shape[0], a.shape[1]))


        iou = 0
        for i in range(len(np_label)):
            for j in range(len(np_label[0])):
               if np_label[i][j] > 0 and a[i][j] < 250:
                   iou += 1

        #print(iou, cloud, a.shape[0], a.shape[1])
        if cloud == 0:
            cloud = 1
        iou_cal = round((iou / cloud), 5)


        return ratio, iou_cal
        '''


        return ratio
