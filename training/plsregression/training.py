#!/usr/bin/python3

import cv2
import numpy as np

from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
import pickle

import argparse
import os
import os.path as osp

import json

from multiprocessing import Pool, Queue, Manager



def generate_features(path, resizing):
    print(path)
    image = cv2.imread(path)
    image = cv2.resize(image, (resizing, resizing))
    # print(image.shape)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    r = image[:,:,2].astype('int16')
    b = image[:,:,0].astype('int16')
    g = image[:,:,1].astype('int16')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    s = hsv[:,:,1].astype('int16')

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

def load_ground_truth_label(path, resizing):
    image = cv2.imread(path, 0).astype('int16')
    image = cv2.resize(image, (resizing, resizing))

    value = [0, 0, 0]
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] == 0:
                value[0] += 1
                image[i][j] = np.random.randint(1, 15, 1)
            elif image[i][j] == 255:
                value[1] +=1
            else:
                value[2] += 1

    if value[0] == (image.shape[0]*image.shape[1]):
        # print('>>>>>> all clear sky')
        image = np.random.randint(0, 15, (image.shape[0], image.shape[1]))
    elif value[1] == (image.shape[0]*image.shape[1]):
        # print('>>>>>> overcasted sky')
        image = np.random.randint(240, 255, (image.shape[0], image.shape[1]))
    else:
        pass
        # print('>>>>>> partial cloudy sky')

    cloud = 0
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > 127:    ## 255/2 = 127.5
                cloud += 1

    #print(path)
    #print('>>>>>> ground truth: ', (cloud/(image.shape[0]*image.shape[1]))*100)

    column_image = (np.transpose(np.matrix(image.flatten()))-np.mean(image))/np.std(image)
    column_image = column_image - np.mean(column_image)
    return_image = np.concatenate((column_image, column_image, column_image), axis=1)

    return return_image, (cloud/(image.shape[0]*image.shape[1]))*100

def pls_regression(features, labels, argscomp):#, test_feature):
    pls2 = PLSRegression(copy=True, n_components=argscomp, max_iter=500, scale=True)
    pls2.fit(features, labels)

    model_file = '/storage/pls_models/model.pkl'
    # joblib.dump(pls2, joblib_file) #, compress=4)

    # pls_loaded = joblib.load('model.pkl')

    pickle.dump(pls2, open(model_file, 'wb'))


def evaluation(val, resizing, thr):
    pls_loaded = pickle.load(open('model.pkl', 'rb'))
    print('model loaded')


    def threshold(result, gt_percentage, thr):
        cloud = (np.asarray(result) > thr).sum()
        percentage = (cloud / (result.shape[0] * result.shape[1]) ) * 100

        # print('>>>>>> segmentation result: ', percentage, '\n')
        print('>>>>>> segmentation result: ', percentage)
        print('>>>>>> absolute diff: ', np.abs(gt_percentage - percentage))


    for test_path in val:
        test_feature = generate_features(test_path, resizing)
        Y_pred = pls_loaded.predict(test_feature)
        #print('>>>>>> min :', np.min(Y_pred), 'max : ', np.max(Y_pred))
        # print("Y_pred : {}".format(Y_pred))

        key = test_path.strip().split('/')[-1].split('.')[0]
        if '_' in key:
            key = key.split('_')[-1]
        #print('>>>>>> image: ', key)

        gt_path = find_match(key)
        #print(gt_path)
        label, gt_percentage = load_ground_truth_label(gt_path, resizing)

        threshold(Y_pred, gt_percentage, thr)


def find_match(key):
    for file in label_list:
        if key in file:
            return file


def worker(args):
    path, resizing = args
    feature = Generate_Features(path, resizing)

    key = path.strip().split('/')[-1].split('.')[0]
    if '_' in key:
        key = key.split('_')[-1]


    gt_path = find_match(key)
    label, gt_percentage = Load_Ground_Truth_Label(gt_path, resizing)

    return (feature, label)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some input arguments')

    parser.add_argument('--root', default='./', help='a folder path where training images and labels are')
    parser.add_argument('--image_list', default='pls_image.list', help='a file path that contains paths of images')
    parser.add_argument('--label_list', default='pls_label.list', help='a file path that contains paths of labels')
    parser.add_argument('--resize', default=600, help='image size when training -- the same width and height')
    parser.add_argument('--threshold', default=1, help='threshold when evalutate a model')
    parser.add_argument('--n_cpu', default=4, help='number of used cpus when training')
    parser.add_argument('--n_pls', default=3, help='number of plsregression depth')
    parser.add_argument('--train_rate', default=0.85, help='separation ratio of trainig, default is 85% of images are used for training and rest for validation')
    parser.add_argument('--n_image', default=400, help='in case of short of memory, limit the number of images used for training')

    args = parser.parse_args()

    with open(osp.join(args.root, 'model.config'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, sort_keys=True))

    image_list = []
    label_list = []


    inputlist = args.root + args.image_list
    with open(inputlist, 'r') as f:
        for line in f:
            path = args.root + 'images/' + line.strip()
            image_list.append(path)

    inputlist = args.root + args.label_list
    with open(inputlist, 'r') as f:
        for line in f:
            path = args.root + 'labels/' + line.strip()
            label_list.append(path)


    tnum = [i for i in range(len(image_list))]
    tdata = int(len(image_list)*args.train_rate)
    tsample = sample(tnum, tdata)


    train = []
    val = []
    for i in tnum:
        if i in tsample:
            train.append(image_list[i])
        else:
            val.append(image_list[i])


    feature_seq = np.array([])
    label_seq = np.array([])

    t = train[:args.n_image]
    with Pool(processes=args.n_cpu) as pool:
        _args = tuple([(i, args.resize) for i in t])
        res = pool.map(worker, _args)
        for i in res:
            feature, label = i
            if( len(feature_seq) == 0 ):
                feature_seq = feature
                label_seq = label
            else:
                feature_seq = np.concatenate((feature_seq, feature), axis=0)
                label_seq = np.concatenate((label_seq, label), axis=0)

    print('start regression')
    # try:
    pls_regression(feature_seq, label_seq, args.n_pls)


    print('evaluation')
    evaluation(val, args.resize, args.threshold)

