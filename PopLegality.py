#encoding:utf-8

import Initialization
import random
import copy
import numpy as np

#特征矩阵一行突变
def featureMatrixMutation(row):
    index = random.randint(0, len(row) - 1)
    while row[index] != 0:
        index = random.randint(0, len(row) - 1)
    row[index] = 1 
    return row
    
#编码矩阵一行突变
def codeMatrixMutation(row):
    index = random.randint(0, len(row) - 1)
    temp = [-1,1]
    if row[index] == 0:
        temp_index = random.randint(0, 1)
        row[index] = temp[temp_index]
    else:
        row[index] = -row[index]
    return row
    
#编码矩阵一行随机突变
def codeMatrixMutationR(row):
    index = random.randint(0, len(row) - 1)
    m_num = random.randint(-1, 1)
    while row[index] == m_num:
        m_num = random.randint(-1, 1)
    row[index] = m_num
    return row

def fitFeatureMatrix(feature_matrix):
    num_classifier = np.array(feature_matrix).shape[0]
    feature_size = np.array(feature_matrix).shape[1]
    #特征矩阵合法性判断
    #每行至少选择1/5个特征
    for i in range(num_classifier):
        row = feature_matrix[i]
        if row.count(1) < feature_size / 5:
            feature_matrix[i] = featureMatrixMutation(row)
    return feature_matrix
    

def fitCodingMatrix(coding_matrix):
    num_classifier = np.array(coding_matrix).shape[0]
    class_size = np.array(coding_matrix).shape[1]
    #编码矩阵合法性判断
    #行不相同，不相反，同时存在+-1
    for i in range(num_classifier):
        row = coding_matrix[i]
        if not 1 in row or not -1 in row:#判断是否同时存在+-1
            coding_matrix[i] = codeMatrixMutationR(row)
        #该行是否和其他行相同或相反
        temp_coding_matrix = copy.copy(coding_matrix)
        del temp_coding_matrix[i]
        if len(temp_coding_matrix) != 0:
            for temp_row in temp_coding_matrix:
                if (np.array(row)-np.array(temp_row)).tolist().count(0)==class_size or (np.array(row)+np.array(temp_row)).tolist().count(0)==class_size:
                    coding_matrix[i] = codeMatrixMutationR(row)

    #转置后，行不全0，2行不相同
    coding_matrix = np.transpose(coding_matrix)
    coding_matrix = coding_matrix.tolist()
    for i in range(class_size):
        row = coding_matrix[i]
        if row.count(0) == class_size:#判断是否全为零
            coding_matrix[i] = codeMatrixMutationR(row)
        #该行是否和其他行相同
        temp_coding_matrix = copy.copy(coding_matrix)
        del temp_coding_matrix[i]
        if len(temp_coding_matrix) != 0:
            for temp_row in temp_coding_matrix:
                if (np.array(row)-np.array(temp_row)).tolist().count(0) == num_classifier:
                    coding_matrix[i] = codeMatrixMutationR(row)
    coding_matrix = np.transpose(coding_matrix).tolist()  
    return coding_matrix
    
def checkCodingMatrixLegality(coding_matrix):  
    num_classifier = np.array(coding_matrix).shape[0]
    class_size = np.array(coding_matrix).shape[1]            
    #编码矩阵合法性判断
    #行不相同，不相反，同时存在+-1
    for i in range(num_classifier):
        row = coding_matrix[i]
        if not 1 in row or not -1 in row:##判断是否同时存在+-1
            return False
        #该行是否和其他行相同或相反
        temp_coding_matrix = copy.copy(coding_matrix)
        del temp_coding_matrix[i]
        if len(temp_coding_matrix) != 0:
            for temp_row in temp_coding_matrix:
                if (np.array(row)-np.array(temp_row)).tolist().count(0)==class_size or (np.array(row)+np.array(temp_row)).tolist().count(0)==class_size:
                    return False
    #转置后，行不全0,2行不相同
    coding_matrix = np.transpose(coding_matrix)
    coding_matrix = coding_matrix.tolist()
    for i in range(class_size):
        row = coding_matrix[i]
        if row.count(0) == class_size:#判断是否全为零
            return False
        #该行是否和其他行相同
        temp_coding_matrix = copy.copy(coding_matrix)
        del temp_coding_matrix[i]
        if len(temp_coding_matrix) != 0:
            for temp_row in temp_coding_matrix:
                if (np.array(row)-np.array(temp_row)).tolist().count(0) == num_classifier:
                    return False  
    return True

def checkFeatureMatrixLegality(feature_matrix):  
#    print('checkFeatureMatrixLegality')
    num_classifier = np.array(feature_matrix).shape[0]
    feature_size = np.array(feature_matrix).shape[1]
    #特征矩阵合法性判断
    #每行至少选择1/5个特征
    for i in range(num_classifier):
        row = feature_matrix[i]
        if row.count(1) < feature_size / 5:
            return False   
    return True
    
def checkLegality(chromes,PopInfors):
    matrixs = Initialization.decoding(chromes,PopInfors)
    for i in range(len(matrixs)):
        threeMatrix = matrixs[i]
        coding_matrix = threeMatrix.coding_matrix
        feature_matrix = threeMatrix.feature_matrix
        transpose_code_matrix = np.transpose(coding_matrix)
        legalityExamination(transpose_code_matrix)
        while not checkFeatureMatrixLegality(feature_matrix):
            feature_matrix = fitFeatureMatrix(feature_matrix)  
        matrixs[i].coding_matrix = (np.transpose(transpose_code_matrix)).tolist()
        matrixs[i].feature_matrix = feature_matrix
    return matrixs

#def checkMatrixsLegality(matrixs):
#    for i in range(len(matrixs)):
#        threeMatrix = matrixs[i]
#        coding_matrix = threeMatrix.coding_matrix
#        feature_matrix = threeMatrix.feature_matrix
#        while not checkCodingMatrixLegality(coding_matrix):
#            coding_matrix = fitCodingMatrix(coding_matrix)
#        while not checkFeatureMatrixLegality(feature_matrix):
#            feature_matrix = fitFeatureMatrix(feature_matrix)  
#        matrixs[i].coding_matrix = coding_matrix
#        matrixs[i].feature_matrix = feature_matrix
#    return matrixs

def checkMatrixsLegality(matrixs):
    for i in range(len(matrixs)):
        threeMatrix = matrixs[i]
        coding_matrix = threeMatrix.coding_matrix
        feature_matrix = threeMatrix.feature_matrix
        transpose_code_matrix = np.transpose(coding_matrix)
        legalityExamination(transpose_code_matrix)
        while not checkFeatureMatrixLegality(feature_matrix):
            feature_matrix = fitFeatureMatrix(feature_matrix)  
        matrixs[i].coding_matrix = (np.transpose(transpose_code_matrix)).tolist()
        matrixs[i].feature_matrix = feature_matrix
    return matrixs
         
            
'''examination of legality突变合法性检查'''
def legalityExamination(coding_matrix):
        code_matrix= coding_matrix
        i=0
        flag = False
        temparray = np.zeros(code_matrix.shape[1])
        for line in code_matrix:
            #不能含有全为0的行
            while (line==temparray).all():
                index = random.randint(0,code_matrix.shape[1]-1)
                line[index] = random.randint(-1,1)
                flag=True
            #不能含有相同的行
            temp_code_matrix = np.delete(code_matrix,i,axis=0)
            for j in range(temp_code_matrix.shape[0]):
                while ((line-temp_code_matrix[j])==temparray).all():
                    index = random.randint(0,code_matrix.shape[1]-1)
                    line[index] = random.randint(-1,1)
                    flag=True
            i=i+1
            
        class_size = code_matrix.shape[0]
        temparray = np.zeros(class_size)
        for i in range(code_matrix.shape[1]):
            #每一列必须包含1和-1
            while (code_matrix[:,i]==1).any()==False or (code_matrix[:,i]==-1).any()==False:
                index = random.randint(0,class_size-1)
                code_matrix[index][i] = random.randint(-1,1)
                flag=True
            #不能含有相同或相反的列
            transpose_code_matrix = np.transpose(code_matrix)
            temp_code_matrix = np.delete(transpose_code_matrix,i,axis=0)
            for j in range(temp_code_matrix.shape[0]):
                while ((transpose_code_matrix[i]-temp_code_matrix[j])==temparray).all() or ((transpose_code_matrix[i]+temp_code_matrix[j])==temparray).all():
                    index = random.randint(0,class_size-1)
                    transpose_code_matrix[i][index] = random.randint(-1,1)
                    code_matrix[index][i] = transpose_code_matrix[i][index]
                    flag=True

        if flag==True:
            legalityExamination(code_matrix)
