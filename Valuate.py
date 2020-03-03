#encoding:utf-8
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ECOCClassfier import ECOCClassifier2
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import TalentsPool
import sklearn.metrics as ms
import DeapSelect as ds
from sklearn.svm import SVC

'''每列预测准确情况'''
def newMatrix(PopInfors,coding_matrix,feature_matrix,base_learner_matrix):
    train_x = PopInfors.train_x
    train_y = PopInfors.train_y
    validate_x = PopInfors.validate_x
    validate_y = PopInfors.validate_y

    predict_matrix = []
    for i in range(len(coding_matrix)):
        temp_predict_matrix = []
            
        temp_coding_matrix = []
        temp_base_learner_matrix = []
        temp_feature_matrix = []
        
        temp_base_learner_matrix.append(base_learner_matrix[i])
        temp_coding_matrix.append(coding_matrix[i])
        temp_feature_matrix.append(feature_matrix[i])
        temp_coding_matrix = np.transpose(temp_coding_matrix)
        
        ecoc_classifier = ECOCClassifier2(temp_base_learner_matrix, temp_coding_matrix, temp_feature_matrix)
        predict = ecoc_classifier.fit_predict(train_x, train_y, validate_x)
        #预测准确为1，否则为0
        for j in range(len(predict)):
            if predict[j] == validate_y[j]:
                temp_predict_matrix.append(1)
            else:
                temp_predict_matrix.append(0)
        predict_matrix.append(temp_predict_matrix)
        
    return predict_matrix
    
'''汉明距离 diversity'''
def hammingDistance(a1,a2):
    n = np.array(a1) - np.array(a2)
    dis = 0
    for i in range(len(a1)):
        if n[i]!=0:
            dis+=1
    return dis


'''个体评价，增加f1score和distance两个属性值'''
def calObjValue(PopInfors, matrixs):
    fix_estimators = PopInfors.base_learners
    train_x = PopInfors.train_x
    train_y = PopInfors.train_y
    validate_x = PopInfors.validate_x
    validate_y = PopInfors.validate_y
    for j in range(len(matrixs)):
        threeMatrix = matrixs[j]
        
        coding_matrix = threeMatrix.coding_matrix
        feature_matrix = threeMatrix.feature_matrix
        base_learner_matrix = threeMatrix.base_learner_matrix
        # 将fs_matrix从[0,1]模式转换为[1,2...]表示第几个特征被选上的模式
        num_classifier = np.array(feature_matrix).shape[0]
        feature_size = np.array(feature_matrix).shape[1]
        temp_feature_matrix = []
        for i in range(num_classifier):
            temp = []
            for k in range(feature_size):
                if feature_matrix[i][k] == 1:
                    temp.append(k)
            temp_feature_matrix.append(temp)
        feature_matrix = temp_feature_matrix
              
        #将base_learner_matrix转化为具体的基分类器
        temp_base_learner_matrix = []
        for i in range(num_classifier):
            base_learner = fix_estimators[base_learner_matrix[i]]
            temp_base_learner_matrix.append(base_learner)
        base_learner_matrix = temp_base_learner_matrix
        
        temp_coding_matrix = copy.copy(coding_matrix)
        temp_coding_matrix = np.transpose(temp_coding_matrix)
        #编码矩阵f-score
        ecoc_classifier = ECOCClassifier2(base_learner_matrix, temp_coding_matrix, feature_matrix)
        predict = ecoc_classifier.fit_predict(train_x, train_y, validate_x)
        f1score = ms.f1_score(validate_y,predict,average='micro')
        
        #编码矩阵列汉明距离,此处已经修改为新的矩阵间距离
        predict_matrix = newMatrix(PopInfors,coding_matrix,feature_matrix,base_learner_matrix)
        distance = 0
        for i in range(num_classifier):
            for k in range(i+1,num_classifier):
                distance += hammingDistance(predict_matrix[i],predict_matrix[k])
        distance = np.float(distance/(num_classifier*(num_classifier-1))/len(predict_matrix[0])*2)
        
        matrixs[j].distance = distance
        matrixs[j].f1score = f1score
    return matrixs

'''分析编码矩阵的每行，得到混淆矩阵,增加编码矩阵每行准确率属性值'''
def addToTalentsPool(PopInfors, matrixs, talents_pool):
    fix_estimators = PopInfors.base_learners
    train_x = PopInfors.train_x
    train_y = PopInfors.train_y
    validate_x = PopInfors.validate_x
    validate_y = PopInfors.validate_y

    for j in range(len(matrixs)):
        threeMatrix = matrixs[j]
        
        coding_matrix = threeMatrix.coding_matrix
        feature_matrix = threeMatrix.feature_matrix
        base_learner_matrix = threeMatrix.base_learner_matrix
        num_classifier = np.array(coding_matrix).shape[0]
        
        # 将fs_matrix从[0,1]模式转换为[1,2...]表示第几个特征被选上的模式
        num_classifier = np.array(feature_matrix).shape[0]
        feature_size = np.array(feature_matrix).shape[1]
        temp_feature_matrix = []
        for i in range(num_classifier):
            temp = []
            for k in range(feature_size):
                if feature_matrix[i][k] == 1:
                    temp.append(k)
            temp_feature_matrix.append(temp)
        feature_matrix = temp_feature_matrix
              
        #将base_learner_matrix转化为具体的基分类器
        temp_base_learner_matrix = []
        for i in range(num_classifier):
            base_learner = fix_estimators[base_learner_matrix[i]]
            temp_base_learner_matrix.append(base_learner)
        base_learner_matrix = temp_base_learner_matrix
        
        segment_accuracies = []
        segment_confusion_matrixs = []
        #编码矩阵每行混淆矩阵
        for i in range(num_classifier):
            base_learner_ = [base_learner_matrix[i]]
            temp_code_matrix_ = [coding_matrix[i]]
            code_matrix_ = np.transpose(temp_code_matrix_)
            feature_matrix_ = [feature_matrix[i]]
            
            classifier = ECOCClassifier2(base_learner_,code_matrix_,feature_matrix_)
            predict_y = classifier.fit_predict(train_x, train_y, validate_x)
            
            #混淆矩阵
            C = confusion_matrix(validate_y,predict_y)
            #混淆矩阵准确率总和及每一类准确率
            c_accuracies = []
            c_accuracy = 0
            
            for k in range(C.shape[1]):
                c_accuracy += np.float(C[k][k])
                c_accuracies.append(np.float(C[k][k]))
                
            segment_accuracies.append(c_accuracy/len(predict_y))
            segment_confusion_matrixs.append(c_accuracies)

            #添加到精英池
            if c_accuracy > len(predict_y)*0.5:
                pool_talent = TalentsPool.Talents(threeMatrix.coding_matrix[i],threeMatrix.feature_matrix[i],threeMatrix.base_learner_matrix[i], c_accuracy/len(predict_y), c_accuracies)
                talents_pool.append(pool_talent)
                
        matrixs[j].segment_accuracies = segment_accuracies
        matrixs[j].segment_confusion_matrixs = segment_confusion_matrixs
    return talents_pool, matrixs
    