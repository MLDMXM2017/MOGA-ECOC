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
import ThreeMatrix
from sklearn.metrics import confusion_matrix

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
    parent_matrixs = copy.deepcopy(matrixs)
    elite = ThreeMatrix.ThreeMatrix(0,0,0,0,0,0,0)
    
    test_obj_values = []
    validate_obj_values = []
    columns = []
    distances = []

    average_validate_values = []
    average_diversities = []
    
    middle_pop_diversity = []
    middle_pop_f1score = []
    last_pop_diversity = []
    last_pop_f1score = []
    quarter_pop_diversity = []
    quarter_pop_f1score = []
    quarters_pop_diversity = []
    quarters_pop_f1score = []
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
        # print('局部优化'+ str(iteration))
        # matrixs = Operate.localOptimization1(PopInfors,matrixs,talents_pool)
        print('精英保留'+ str(iteration))
        matrixs,elite = Operate.eliteRetention(parent_matrixs,matrixs,elite)
        parent_matrixs = copy.deepcopy(matrixs)
        print('记录'+ str(iteration))
        
        best_matrix = copy.deepcopy(elite)
        base_learner_matrix = best_matrix.base_learner_matrix
        coding_matrix = best_matrix.coding_matrix
        feature_matrix = best_matrix.feature_matrix
        fix_estimators = PopInfors.base_learners
        
        coding_matrix = np.transpose(coding_matrix)
        
        # 将fs_matrix从[0,1]模式转换为[1,2...]表示第几个特征被选上的模式
        num_classifier = np.array(feature_matrix).shape[0]
        feature_size = np.array(feature_matrix).shape[1]
        temp_feature_matrix = []
        for i in range(num_classifier):
            temp = []
            for k in range(feature_size):
                # if feature_matrix[i][k] == 1:
                #     temp.append(k)
                temp.append(k)
            temp_feature_matrix.append(temp)
        feature_matrix = temp_feature_matrix
        
        #转化成选用的哪个分类器
        estimators=[]
        for j in range(len(base_learner_matrix)):
            choose_temp=fix_estimators[base_learner_matrix[j]]
            estimators.append(choose_temp)
        
        train_x = PopInfors.train_x
        train_y = PopInfors.train_y       
        test_x = PopInfors.test_x
        test_y = PopInfors.test_y
        
        ecoc_classifier = ECOCClassifier2(estimators, coding_matrix, feature_matrix)
        predict_y_test = ecoc_classifier.fit_predict(train_x, train_y, test_x)
        test_obj_values.append(ms.f1_score(test_y,predict_y_test,average='micro'))
        
        
        validate_obj_values.append(best_matrix.f1score)
        columns.append(num_classifier)
        distances.append(best_matrix.distance)
        print('精英',best_matrix.f1score,best_matrix.distance)
        print('精英test',ms.f1_score(test_y,predict_y_test,average='micro'))
        #该种群平均情况
        sum_validate_values = 0
        sum_diversity = 0
        for threeMatrix in matrixs:
            sum_validate_values += threeMatrix.f1score
            sum_diversity += threeMatrix.distance
        average_validate_values.append(sum_validate_values / PopInfors.pop_size)
        average_diversities.append(sum_diversity / PopInfors.pop_size)
        

        if iteration == maximum_iteration - 1:
            for threeMatrix in matrixs:
                last_pop_diversity.append(threeMatrix.distance)
                last_pop_f1score.append(threeMatrix.f1score)
        elif iteration == maximum_iteration/2 - 1:
            for threeMatrix in matrixs:
                middle_pop_diversity.append(threeMatrix.distance)
                middle_pop_f1score.append(threeMatrix.f1score)
        elif iteration == maximum_iteration/4 - 1:
            for threeMatrix in matrixs:
                quarter_pop_diversity.append(threeMatrix.distance)
                quarter_pop_f1score.append(threeMatrix.f1score) 
        elif iteration == maximum_iteration/4*3 - 1:
            for threeMatrix in matrixs:
                quarters_pop_diversity.append(threeMatrix.distance)
                quarters_pop_f1score.append(threeMatrix.f1score)
        
    # 混淆矩阵
    C = confusion_matrix(test_y,predict_y_test)
    c_accuracies = []
    c_accuracies1 = []
    for kk in range(C.shape[1]):
        c_accuracies.append(np.float(C[kk][kk]))
        c_accuracies1.append(np.float(C[kk][kk])/len(test_y))
    print(datafile,c_accuracies,c_accuracies1)
    Record.logExcel(str(datafile)+str(runindex),test_obj_values,'test_values')
    Record.logExcel(str(datafile)+str(runindex),validate_obj_values,'validate_values')
    Record.logExcel(str(datafile)+str(runindex),columns,'column')
    Record.logExcel(str(datafile)+str(runindex),distances,'diversity')

    Record.logExcel(str(datafile)+str(runindex),average_validate_values,'average_validate_values')
    Record.logExcel(str(datafile)+str(runindex),average_diversities,'average_diversity')

    Record.logExcel(str(datafile)+str(runindex),last_pop_diversity,'last_pop_diversity')
    Record.logExcel(str(datafile)+str(runindex),last_pop_f1score,'last_pop_f1score')
    Record.logExcel(str(datafile)+str(runindex),middle_pop_diversity,'middle_pop_diversity')
    Record.logExcel(str(datafile)+str(runindex),middle_pop_f1score,'middle_pop_f1score')
    Record.logExcel(str(datafile)+str(runindex),quarter_pop_diversity,'quarter_pop_diversity')
    Record.logExcel(str(datafile)+str(runindex),quarter_pop_f1score,'quarter_pop_f1score')
    Record.logExcel(str(datafile)+str(runindex),quarters_pop_diversity,'3quarter_pop_diversity')
    Record.logExcel(str(datafile)+str(runindex),quarters_pop_f1score,'3quarter_pop_f1score')
        