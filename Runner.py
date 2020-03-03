#!/usr/bin/env python
# -*- coding: utf-8 -*-

import DataLoader
import numpy as np
import random
import PopInformations
import test

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    ####################################################################
    # 第一阶段：读取数据，为训练和分类做准备
    ####################################################################

    datafile = []
#    datafile = ["zoo","mfeatmor","vehicle","yeast"]
#    datafile = ["mfeatmor","vehicle"]
    '''
    "abalone","cmc","dermatology","glass","iris","wine","thyroid","vertebral","ecoli","yeast","zoo","mfeatmor","vehicle",
                "waveforms","sat","mfeatzer","mfeatpix",
    '''
    datafile = ["Cancers","DLBCL","GCM","Leukemia1","Leukemia2",
               "Breast","Lung2","SRBCT","Lung1"]
#    datafile = ["vertebral","thyroid","yeast","zoo","mfeatmor","vehicle"]
#    datafile = ['cmc']
#     datafile = ['mfeatpix']
#    datafile = ["mfeatpix","mfeatzer","sat","waveforms"]

#    datafile = ["dermatology","glass","iris","wine","thyroid","vertebral","ecoli","yeast","cmc"]
#    datafile = ["zoo","mfeatmor"]
#    datafile = ["cmc","abalone","vertebral","glass"]
#    ["iris","glass","wine","yeast","zoo","thyroid","vertebral","dermatology","cmc","abalone","mfeatmor","vehicle","ecoli"]
    for index in range(len(datafile)):
        trainfile = "./data_uci/"+datafile[index]+"_train.data"
        testfile = "./data_uci/"+datafile[index]+"_test.data"
        validatefile = "./data_uci/"+datafile[index]+"_validation.data"
        # 其中x为特征空间，y为样本的标签
        train_x, train_y, validate_x, validate_y,instance_size = DataLoader.loadDataset(trainfile,validatefile)
        train_x, train_y, test_x, test_y, instance_size = DataLoader.loadDataset(trainfile, testfile)
    
        ####################################################################
        # 遗传算法实现ECOC，定参数
        ####################################################################
        maximum_iteration = 100
        pm = 0.1
        pop_size = 50
        num_parent = 5
        class_size = len(np.unique(np.array(train_y)))
        feature_size = int(len(train_x[0]))
        runindex = 1
        # base_learners = [tree.DecisionTreeClassifier(),GaussianNB(),KNeighborsClassifier(),SVC(),LogisticRegression(),MLPClassifier()]
        base_learners = [GaussianNB()]
#  """ Get classifiers from scikit-learn.
#            'KNN' - K Nearest Neighbors (sklearn.neighbors.KNeighborsClassifier)
#            'DTree' - Decision Tree (sklearn.tree.DecisionTreeClassifier)
#            'SVM' - Support Vector Machine (sklearn.svm.SVC)
#            'Bayes' - Naive Bayes (sklearn.naive_bayes.GaussianNB)
#            'Logi' - Logistic Regression (sklearn.linear_model.LogisticRegression)
#            'NN' - Neural Network (sklearn.neural_network.MLPClassifier)
#        adaboost: bool, default False.
#            Whether to use adaboost to promote the classifier.
#    """
        for i in range(runindex):
            print("running:"+str(i)+"次")
            init_num_classifier = random.randint(1*class_size, round(1.5*class_size))
            chrom_length = (class_size + feature_size) * init_num_classifier

            PopInfors = PopInformations.PopInfors(pop_size, class_size, feature_size, init_num_classifier, maximum_iteration, train_x, train_y, test_x, test_y,validate_x,validate_y,base_learners)
            test.GA(PopInfors,datafile[index],i,)
            




