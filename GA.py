# coding: utf-8
import PopInformations
import Initialization
import numpy as np
import PopLegality
import Valuate
import Operate
import copy
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ECOCClassfier import ECOCClassifier2
import sklearn.metrics as ms
import Record

def GA(PopInfors,datafile,runindex):
    talents_pool = []
    maximum_iteration = PopInfors.maximum_iteration
    print('产生初代')
    chromes = Initialization.intialization(PopInfors)
    print('初代合法性判断')
    matrixs = PopLegality.checkLegality(chromes,PopInfors)
    print('个体评价')
    matrixs = Valuate.calObjValue(PopInfors, matrixs)
    talents_pool, matrixs = Valuate.addToTalentsPool(PopInfors, matrixs, talents_pool)
#    for threeMatrix in matrixs:
#        print(threeMatrix.f1score,threeMatrix.distance)
#    for threeMatrix in matrixs:
#        print('coding_matrix',threeMatrix.coding_matrix)
#        print('feature_matrix',threeMatrix.feature_matrix)
#        print('base_learner_matrix',threeMatrix.base_learner_matrix)
#        print('f1score',threeMatrix.f1score)
#        print('distance',threeMatrix.distance)
#        print('segment_accuracy',threeMatrix.segment_accuracies)
#        print('segment_confusion_matrixs',threeMatrix.segment_confusion_matrixs)
    parent_matrixs = copy.copy(matrixs)
    test_obj_values = []
    validate_obj_values = []
    columns = []
    distances = []
    for iteration in range(maximum_iteration):        
        print('交叉' + str(iteration))
        matrixs = Operate.crossover(PopInfors, matrixs, talents_pool)
        print('突变'+ str(iteration))
        matrixs = Operate.mutation(PopInfors, matrixs)
        print('合法性判断'+ str(iteration))
        matrixs = PopLegality.checkMatrixsLegality(matrixs)
        print('个体评价'+ str(iteration))
        matrixs = Valuate.calObjValue(PopInfors, matrixs)
        talents_pool, matrixs = Valuate.addToTalentsPool(PopInfors, matrixs, talents_pool)
        print('局部优化'+ str(iteration))
        matrixs = Operate.localOptimization(PopInfors,matrixs,talents_pool)
        print('精英保留'+ str(iteration))
        matrixs = Operate.eliteRetention(parent_matrixs,matrixs)
        parent_matrixs = matrixs
        
        
        best_matrix = copy.copy(matrixs[0])
        base_learner_matrix = best_matrix.base_learner_matrix
        coding_matrix = best_matrix.coding_matrix
        feature_matrix = best_matrix.feature_matrix
        fix_estimators = [tree.DecisionTreeClassifier(),GaussianNB(),KNeighborsClassifier(n_neighbors=3)]
        
        coding_matrix = np.transpose(coding_matrix)
        
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
        
        #转化成选用的哪个分类器
        estimators=[]
        for j in range(len(base_learner_matrix)):
            choose_temp=fix_estimators[base_learner_matrix[j]]
            estimators.append(choose_temp)
        print(estimators)
        
        train_x = PopInfors.train_x
        train_y = PopInfors.train_y       
#        validate_x = PopInfors.validate_x
#        validate_y = PopInfors.validate_y
        test_x = PopInfors.test_x
        test_y = PopInfors.test_y
        
        ecoc_classifier = ECOCClassifier2(estimators, coding_matrix, feature_matrix)
        predict_y_test = ecoc_classifier.fit_predict(train_x, train_y, test_x)
        test_obj_values.append(ms.f1_score(test_y,predict_y_test,average='micro'))
        
#        predict_y_validate = ecoc_classifier.fit_predict(train_x, train_y, validate_x)
#        validate_obj_values.append(ms.f1_score(validate_y,predict_y_validate,average='micro'))
        
        validate_obj_values.append(best_matrix.f1score)
        columns.append(num_classifier)
        distances.append(best_matrix.distance)
        
#    Record.logExcel(str(datafile)+str(runindex),test_obj_values,'test_values')
#    Record.logExcel(str(datafile)+str(runindex),validate_obj_values,'validate_values')
#    Record.logExcel(str(datafile)+str(runindex),test_obj_values,'column')
#    Record.logExcel(str(datafile)+str(runindex),test_obj_values,'distance')
    print(test_obj_values)
    print(validate_obj_values)
        
        