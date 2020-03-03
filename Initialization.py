# coding:utf-8

import ThreeMatrix
import random

'''随机生成初代'''
def intialization(PopInfors):
    pop_size = PopInfors.pop_size
    class_size = PopInfors.class_size
    feature_size = PopInfors.feature_size
    num_classifier = PopInfors.num_classifier
    base_learners = PopInfors.base_learners
    chromes = []
    for i in range(pop_size):
        chrome = []
        for j in range(num_classifier):
            #编码矩阵
            for k in range(class_size):
                rand_num = random.randint(-1, 1)
                chrome.append(rand_num)
            #特征矩阵
            for k in range(feature_size):
                rand_num = random.randint(0, 1)
                chrome.append(rand_num)
            #基分类器
            rand_num = random.randint(0, len(base_learners)-1)
            chrome.append(rand_num)
        chromes.append(chrome)
    return chromes

#染色体解码
def decoding(chromes,PopInfors):
    class_size = PopInfors.class_size
    feature_size = PopInfors.feature_size
    segment = class_size+feature_size+1
    matrixs = []
    for i in range(len(chromes)):       
        num_classifier = int(len(chromes[i])/(class_size+feature_size+1))
        coding_matrix,feature_matrix,base_learner_matrix = [],[],[]
        for j in range(num_classifier):
            coding_matrix.append(chromes[i][segment*j:segment*j+class_size])
            feature_matrix.append(chromes[i][segment*j+class_size:segment*j+class_size+feature_size])
            base_learner_matrix.extend(chromes[i][segment*j+class_size+feature_size:segment*j+class_size+feature_size+1])
        threeMatrix = ThreeMatrix.ThreeMatrix(coding_matrix,feature_matrix,base_learner_matrix,-1,-1,-1,-1)
        matrixs.append(threeMatrix)
    return matrixs
        
#染色体编码
def coding(coding_ms,feature_ms,learner_ms):
    chrome = []
    for i in range(len(coding_ms)):
        chrome.extend(coding_ms[i])
        chrome.extend(feature_ms[i])
        chrome.append(learner_ms[i])
    return chrome