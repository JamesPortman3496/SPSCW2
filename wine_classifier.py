#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utilities import load_data, print_features, print_predictions
from sklearn.decomposition import PCA

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

# Normalises an array around its mean
def normalise(arr):
    return np.divide(arr,np.mean(arr))

def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of
    # the function
    return [6,9]


def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    if (k<1) or (k>7):
        return None

    features=feature_selection(train_set,train_labels)

    reduced_train_set=train_set[:,features]
    normalised_reduced_train_set=np.array([normalise(reduced_train_set[:,0]),normalise(reduced_train_set[:,1])]).T

    reduced_test_set=test_set[:,features]
    normalised_reduced_test_set=np.array([normalise(reduced_test_set[:,0]),normalise(reduced_test_set[:,1])]).T

    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    obs_dist = lambda x : [dist(x, obs) for obs in normalised_reduced_train_set]

    test_dist=[]
    for t in normalised_reduced_test_set:
        test_dist.append(obs_dist(t))

    k_nearest=[np.argsort(test_dist[i])[:k].astype(np.int) for i in range (0,len(test_set))]

    classified=[]
    for i in range (0,len(test_set)):
        neighbours=k_nearest[i] # neighbours
        labels=train_labels[neighbours]
        classified.append(most_common_class(labels, train_set, train_labels, np.array([test_set[i]]), k))

    return classified

def most_common_class(labels, train_set, train_labels, test_set, k):
    class_occurences=[0,0,0]
    for l in labels:
        class_occurences[l-1]+=1

    if (class_occurences[0]>class_occurences[1]) and (class_occurences[0]>class_occurences[2]):
        return 1
    if (class_occurences[1]>class_occurences[0]) and (class_occurences[1]>class_occurences[2]):
        return 2
    if (class_occurences[2]>class_occurences[0]) and (class_occurences[2]>class_occurences[1]):
        return 3
    else:
        return knn(train_set, train_labels, test_set, k-1)[0]

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    features=feature_selection(train_set,train_labels)

    train_set_1 = []
    train_set_2 = []
    train_set_3 = []
    numOf = [0,0,0]

    means=train_set[:,features].mean(axis=0)

    reduced_train_set=train_set[:,features]
    normalised_reduced_train_set=np.array([normalise(reduced_train_set[:,0]),normalise(reduced_train_set[:,1])]).T

    reduced_test_set=test_set[:,features]
    normalised_reduced_test_set=np.array([normalise(reduced_test_set[:,0]),normalise(reduced_test_set[:,1])]).T

    for x in range (0, len(train_set)):
        if(train_labels[x] == 1):
            train_set_1.append(normalised_reduced_train_set[x,:])
            numOf[0] += 1
        if(train_labels[x] == 2):
            train_set_2.append(normalised_reduced_train_set[x,:])
            numOf[1] += 1
        if(train_labels[x] == 3):
            train_set_3.append(normalised_reduced_train_set[x,:])
            numOf[2] += 1

    matrix1 = np.matrix(train_set_1)
    matrix2 = np.matrix(train_set_2)
    matrix3 = np.matrix(train_set_3)
    matrixMean = np.zeros([2, 3])
    matrixSD = np.zeros([2, 3])

    for x in range(0,2):
        matrixMean[x, 0] = np.mean(matrix1[:,x])
        matrixMean[x, 1] = np.mean(matrix2[:,x])
        matrixMean[x, 2] = np.mean(matrix3[:,x])
        matrixSD[x, 0] = np.std(matrix1[:,x])
        matrixSD[x, 1] = np.std(matrix2[:,x])
        matrixSD[x, 2] = np.std(matrix3[:,x])

    classify = []
    for x in test_set:
        x=[x[features[0]]/means[0],x[features[1]]/means[1]]
        values = [0,0,0]
        for i in range (0, 3):
            omega = numOf[i]/len(train_set)
            for j in range (0,2):
                omega *= norm.pdf(x[j],matrixMean[j,i], matrixSD[j,i])
            values[i] = omega
        classify.append(np.argsort(values)[2]+1)

    return classify

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    if (k<1) or (k>7):
        return None

    reduced_train_set=train_set[:,[6,9,5]]
    normalised_reduced_train_set=np.array([normalise(reduced_train_set[:,0]),normalise(reduced_train_set[:,1]),normalise(reduced_train_set[:,2])]).T

    reduced_test_set=test_set[:,[6,9,5]]
    normalised_reduced_test_set=np.array([normalise(reduced_test_set[:,0]),normalise(reduced_test_set[:,1]),normalise(reduced_test_set[:,2])]).T

    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    obs_dist = lambda x : [dist(x, obs) for obs in normalised_reduced_train_set]

    test_dist=[]
    for t in normalised_reduced_test_set:
        test_dist.append(obs_dist(t))

    k_nearest=[np.argsort(test_dist[i])[:k].astype(np.int) for i in range (0,len(test_set))]

    classified=[]
    for i in range (0,len(test_set)):
        neighbours=k_nearest[i] # neighbours
        labels=train_labels[neighbours]
        classified.append(most_common_class(labels, train_set, train_labels, np.array([test_set[i]]), k))

    return classified

def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    pca=PCA(n_components=n_components)
    pca=pca.fit(train_set)
    transformed_train_set=pca.transform(train_set)
    transformed_test_set=pca.transform(test_set)

    dist = lambda x, y: np.sqrt(np.sum((x-y)**2)) # This is already generalised for n dimensions
    obs_dist = lambda x : [dist(x, obs) for obs in transformed_train_set]

    test_dist=[]
    for t in transformed_test_set:
        test_dist.append(obs_dist(t))

    k_nearest=[np.argsort(test_dist[i])[:k].astype(np.int) for i in range (0,len(test_set))]

    classified=[]
    for i in range (0,len(test_set)):
        neighbours=k_nearest[i] # neighbours
        labels=train_labels[neighbours]
        classified.append(most_common_class(labels, train_set, train_labels, np.array([test_set[i]]), k))

    return classified

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
