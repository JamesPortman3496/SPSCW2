from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np
import matplotlib.pyplot as plt
import scipy
from pprint import pprint
from utilities import *
from wine_classifier import *

def calculate_confusion_matrix(gt_labels, pred_labels):
    cm = np.zeros([3, 3])
    for i in range (0, 3):
        for j in range (0, 3):
            samples1 = 0
            for z in range (0, len(gt_labels)):
                if ((gt_labels[z] == i+1) & (pred_labels[z] == j+1)):
                    samples1 = samples1 + 1
            samples2 = 0
            for x in range (0, len(gt_labels)):
                if ((gt_labels[x] == i+1)):
                    samples2 = samples2 + 1
            cm[i, j] = samples1/samples2
    return cm

def confusion_matrix_alt_classifier():
    train_set, train_labels, test_set, test_labels = load_data()
    predicted=alternative_classifier(train_set, train_labels, test_set)
    confusion=calculate_confusion_matrix(test_labels, predicted)
    print(confusion)

def confusion_matrix_knn():
    train_set, train_labels, test_set, test_labels = load_data()
    for k in range (1,8):
        predicted=knn(train_set, train_labels, test_set,k)
        confusion=calculate_confusion_matrix(test_labels, predicted)
        print(confusion)

def confusion_matrix_knn_3d():
    train_set, train_labels, test_set, test_labels = load_data()
    for k in range (1,8):
        predicted=knn_three_features(train_set, train_labels, test_set, k)
        confusion=calculate_confusion_matrix(test_labels, predicted)
        print("k={}".format(k))
        print(confusion)

def confusion_matrix_knn_pca():
    train_set, train_labels, test_set, test_labels = load_data()
    for k in range (1,8):
        predicted=knn_pca(train_set, train_labels, test_set, k)
        confusion=calculate_confusion_matrix(test_labels, predicted)
        print("k={}".format(k))
        print(confusion)

def pca_plot():
    train_set, train_labels, test_set, test_labels = load_data()
    pca=PCA(n_components=2)
    pca=pca.fit(train_set)
    transformed_train_set=pca.transform(train_set)
    transformed_test_set=pca.transform(test_set)

    CLASS_1_C = r'#3366ff'
    CLASS_2_C = r'#cc3300'
    CLASS_3_C = r'#ffc34d'
    colours=[CLASS_1_C,CLASS_2_C,CLASS_3_C]

    trainColours=[]; testColours=[]
    for label in train_labels:
        trainColours.append(colours[label-1])
    for label in test_labels:
        testColours.append(colours[label-1])

    plt.scatter(transformed_train_set[:,0],transformed_train_set[:,1],c=trainColours)
    plt.title("PCA Transformed Training Set")
    plt.savefig("pca",bbox_inches='tight',dpi=100)

pca_plot()
