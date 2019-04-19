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


train_set, train_labels, test_set, test_labels = load_data()
predicted=alternative_classifier(train_set, train_labels, test_set)
confusion=calculate_confusion_matrix(test_labels, predicted)
print(confusion)
