from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np
import matplotlib.pyplot as plt
import scipy
from pprint import pprint
from utilities import *
from wine_classifier import *

from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

class_1_colour = r'#3366ff'
class_2_colour = r'#cc3300'
class_3_colour = r'#ffc34d'

class_colours = [class_1_colour, class_2_colour, class_3_colour]

def calculate_confusion_matrix(gt_labels, pred_labels):
    cm = np.zeros([3, 3])
    for i in range (0, 3):
        for j in range (0, 3):
            samples1 = 0
            for z in range (0, len(gt_labels)):
                if ((gt_labels[z] == i+1) and (pred_labels[z] == j+1)):
                    samples1 +=1
            samples2 = 0
            for x in gt_labels:
                if (x == i+1):
                    samples2 += 1
            cm[i, j] = samples1/samples2
    return cm

def standardise(arr):
    mean=np.mean(arr)
    sd=np.std(arr)
    return np.divide((arr-mean),sd)

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
        print("k={}".format(k))
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

# standardises data & plots each pair of features against each other
def choose_two():
    train_set, train_labels, test_set, test_labels = load_data()

    # write your code here
    colours=[]
    for k in range (0,len(train_set)):
        colours.append(class_colours[train_labels[k]-1])

    for i in range (0,13):
        for j in range (i+1,13):
            plt.clf()

            xs=train_set[:,i]
            ys=train_set[:,j]

            xsstandardised=standardise(xs)
            ysstandardised=standardise(ys)

            plt.scatter(xsstandardised,ysstandardised,c=colours)
            plt.title("Features {} vs {}".format(i+1,j+1))
            plt.savefig("img/{}x{}".format(i+1,j+1),bbox_inches='tight',dpi=100)

def third_feature_correlation():
    train_set, train_labels, test_set, test_labels = load_data()

    n6=standardise(train_set[:,6])
    n9=standardise(train_set[:,9])

    for i in range (0,13):
        if (i!=6 and i!=9):
            cc=np.corrcoef([n6,n9,standardise(train_set[:,i])])

            corr=cc[0,2]**2+cc[1,2]**2-2*cc[0,1]*cc[0,2]*cc[1,2]
            corr=corr/(1-cc[0,1]**2)
            print("i={}, corr={}".format(i,np.sqrt(corr)))

print("KNN")
choose_two()
third_feature_correlation()
