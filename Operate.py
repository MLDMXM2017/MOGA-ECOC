#encodingï¼šutf-8
import PopLegality
import random
import Valuate
import copy
import Initialization
from deap import tools
import DeapSelect
import math
import numpy as np

'''突变操作，根据segment的准确度有不同的突变概率'''
def mutation(PopInfors, matrixs):
    for j in range(len(matrixs)):
        threeMatrix = matrixs[j]
        coding_matrix = threeMatrix.coding_matrix
        segment_accuracies = threeMatrix.segment_accuracies
        new_coding_matrix = []
        for i in range(len(coding_matrix)):   
            segment = coding_matrix[i]
            segment_accuracy = segment_accuracies[i]
            pm = random.random()
            if segment_accuracy > 0.5:                
                if pm < 0.1:
                    segment = PopLegality.codeMatrixMutation(segment)
            else:
                if pm < 0.2:
                    segment = PopLegality.codeMatrixMutation(segment)
            new_coding_matrix.append(segment)
        threeMatrix.coding_matrix = new_coding_matrix
    return matrixs


'''交叉操作'''
def crossover(PopInfors, matrixs, talents_pool):
    pop_size = PopInfors.pop_size
    
    children_matrixs = []
    for cross_time in range(int(pop_size/2)):
        #随机产生父辈候选染色体
        random_seed = []
        for i in range(pop_size):
            random_seed.append(i)
        random.shuffle(random_seed)
        parents_seed = random_seed[:6]

        parents = []
        for i in range(len(parents_seed)):
            parents.append(matrixs[parents_seed[i]])
        parents1 = parents[:3]
        parents2 = parents[3:6]
  
        #选出父母
#        parents1 = sortMatrix(parents1)
#        parents2 = sortMatrix(parents2)
#        parent1 = parents1[0]
#        parent2 = parents2[0]
#        '''此处修改为新的selection'''
        parent1 = DeapSelect.selectBest1(parents1)
        parent2 = DeapSelect.selectBest1(parents2)
        
        segment_accuray1 = parent1.segment_accuracies
        segment_accuray2 = parent2.segment_accuracies

#分别找到两个矩阵最差的segment，并交叉
#        index1 = 0
#        index2 = 0
#        min_accuray1 = segment_accuray1[0]
#        min_accuray2 = segment_accuray2[0]
#        for i in range(1,len(segment_accuray1)):
#            if segment_accuray1[i] < min_accuray1:
#                min_accuray1 = segment_accuray1[i]
#                index1 = i
#        for i in range(1,len(segment_accuray2)):
#            if segment_accuray2[i] < min_accuray2:
#                min_accuray2 = segment_accuray2[i]
#                index2 = i
        
        index1 = random.randint(0,len(segment_accuray1)-1)
        index2 = random.randint(0,len(segment_accuray2)-1)
        #交叉
        children = renewMatrix(parent1,parent2,index1,index2)
        children_matrixs.extend(children)

    children_matrixs = Valuate.calObjValue(PopInfors, children_matrixs)
    talents_pool, children_matrixs = Valuate.addToTalentsPool(PopInfors, children_matrixs, talents_pool) 
    
    return children_matrixs

    
'''精英保留'''
def eliteRetention(parent_matrixs,children_matrixs,elite):
    if elite.f1score == 0:
        parent_matrixs = sortMatrix1(parent_matrixs)
        children_matrixs = sortMatrix1(children_matrixs)
        best_matrix = DeapSelect.selectBest1([parent_matrixs[0],children_matrixs[0]])
        if best_matrix.__dict__ == parent_matrixs[0].__dict__:
            children_matrixs[len(children_matrixs)-1] = parent_matrixs[0]
            elite = parent_matrixs[0]
        else:
            elite = children_matrixs[0]            
    else:
        children_matrixs = sortMatrix1(children_matrixs)
        best_matrix = DeapSelect.selectBest1([elite,children_matrixs[0]])
        if best_matrix.__dict__ == elite.__dict__:
            children_matrixs[len(children_matrixs)-1] = elite      
        else:
            elite = children_matrixs[0]
    return children_matrixs, elite

'''局部优化 删、突变、增'''
def localOptimization(PopInfors,matrixs,talents_pool):
    class_size = PopInfors.class_size
    min_num_classifier = class_size - 1
    max_num_classifier = 2*class_size -1
    
    matrixs = sortMatrix1(matrixs) 
    record_matrixs = copy.deepcopy(matrixs)
    
    good = record_matrixs[0]
    good_coding_matrix = good.coding_matrix
    good_segment_accuracies = good.segment_accuracies
    good_segment_confusion_matrixs = good.segment_confusion_matrixs
    
    #找到最优个体分类最差的segment
    worst_segment_accuracy = float("inf") 
    worst_segment_accuracy_index = 0
    
    for i in range(len(good_segment_accuracies)):
        if good_segment_accuracies[i] < worst_segment_accuracy:
            worst_segment_accuracy = good_segment_accuracies[i]
            worst_segment_accuracy_index = i  
            
    #找到最优个体分类最差的segment最好的一类
    best_confusion_matrix = good_segment_confusion_matrixs[worst_segment_accuracy_index] 
    best_class_accuracy = 0
    best_class_accuracy_index = 0   
    
    for i in range(len(best_confusion_matrix)):
        if best_confusion_matrix[i] > best_class_accuracy:
            best_class_accuracy_index = i
            best_class_accuracy = best_confusion_matrix[i]
            
    #找到最优个体分类最差的segment最差的一类
    worst_confusion_matrix = good_segment_confusion_matrixs[worst_segment_accuracy_index] 
    worst_class_accuracy = float("inf") 
    worst_class_accuracy_index = 0   
    
    for i in range(len(worst_confusion_matrix)):
        if worst_confusion_matrix[i] < worst_class_accuracy:
            worst_class_accuracy_index = i
            worst_class_accuracy = worst_confusion_matrix[i]
            
#    '''判断该column是否有贡献（根据上面所得的最好的一类） 准确率是否超过平均值'''
    segments = len(good_coding_matrix)
    sum_class_accuracy = 0
    for i in range(segments):
        temp_confusion_matrix = good_segment_confusion_matrixs[i] 
        temp_class_accuracy = temp_confusion_matrix[best_class_accuracy_index]
        sum_class_accuracy += temp_class_accuracy
    
#    '''删(突变)、增操作'''
    temp_matrix = copy.deepcopy(good)
#    '''删除'''
    print('初始 ',temp_matrix.f1score)
    #若该列最好的类无贡献，则删除
    if not best_class_accuracy > sum_class_accuracy/segments/2 and len(temp_matrix.coding_matrix) -1 > min_num_classifier:
        del temp_matrix.coding_matrix[worst_segment_accuracy_index]
        del temp_matrix.feature_matrix[worst_segment_accuracy_index]
        del temp_matrix.base_learner_matrix[worst_segment_accuracy_index]
        temp_matrix = PopLegality.checkMatrixsLegality([temp_matrix])
        temp_matrix = Valuate.calObjValue(PopInfors, temp_matrix)[0]
        print('删除后 ',temp_matrix.f1score)
    #'''突变'''
    #该列有贡献，则定向突变最差的类：-1变为1,1变为-1,0通过knn决定具体突变为-1/1
#    else:
#        if good_coding_matrix[worst_segment_accuracy_index][worst_class_accuracy_index] != 0:
#            temp_matrix.coding_matrix[worst_segment_accuracy_index][worst_class_accuracy_index] = -good_coding_matrix[worst_segment_accuracy_index][worst_class_accuracy_index]
#            temp_matrix = PopLegality.checkMatrixsLegality([temp_matrix])
#            temp_matrix = Valuate.calObjValue(PopInfors, temp_matrix)[0]
#            print('突变1 ',temp_matrix.f1score)
#        else:
#            temp_matrix1 = copy.deepcopy(good)   
#            temp_matrix2 = copy.deepcopy(good)
#            temp_matrix1.coding_matrix[worst_segment_accuracy_index][worst_class_accuracy_index] = 1
#            temp_matrix2.coding_matrix[worst_segment_accuracy_index][worst_class_accuracy_index] = -1
#            temp_matrixs = []
#            temp_matrixs.append(temp_matrix1)
#            temp_matrixs.append(temp_matrix2)
#            temp_matrixs = PopLegality.checkMatrixsLegality(temp_matrixs)
#            temp_matrixs = Valuate.calObjValue(PopInfors, temp_matrixs)
#            if temp_matrixs[0].f1score > temp_matrixs[1].f1score:
#                temp_matrix = temp_matrixs[0]
#            else:
#                temp_matrix = temp_matrixs[1]
#            print('突变2 ',temp_matrix.f1score)
#    '''增加'''
    flag = 0
    change_f1score= 0
    p_matrix= copy.deepcopy(temp_matrix)
    temp_matrix4 =  copy.deepcopy(temp_matrix)
    distance2 = 0
    while change_f1score >= 0 and len(temp_matrix4.coding_matrix) < max_num_classifier:
        p_matrix = copy.deepcopy(temp_matrix4)
        flag = 0
        if len(talents_pool) != 0 :
            #精英池中选择                
            for talent in talents_pool:
                if not talent.code_segment in temp_matrix4.coding_matrix:
                    flag = 1
                    break
                    
            if flag == 0:
                break
            else:
                random_index = random.randint(0,len(talents_pool)-1)
                choose_pool = talents_pool[random_index]        
                p_distance = matrixDistance(temp_matrix4,class_size)  
                while choose_pool.code_segment in temp_matrix4.coding_matrix:
                    random_index = random.randint(0,len(talents_pool)-1)
                    choose_pool = talents_pool[random_index]        
                temp_matrix4.coding_matrix.append(choose_pool.code_segment)
                temp_matrix4.feature_matrix.append(choose_pool.feature_segment)
                temp_matrix4.base_learner_matrix.append(choose_pool.estimator)
                temp_matrix4 = PopLegality.checkMatrixsLegality([temp_matrix4])
                temp_matrix4 = Valuate.calObjValue(PopInfors, temp_matrix4)[0]
                distance2 = matrixDistance(temp_matrix4,class_size)
                change_f1score = distance2 - p_distance
                print(change_f1score)
        else:
            break
    print('增加 ',p_matrix.f1score)
    #若连续三个column添加进来distance均减少，则终止
    count = 0       
    talents_pool, temp_matrix4 = Valuate.addToTalentsPool(PopInfors, [p_matrix], talents_pool)
    temp_matrix4 = temp_matrix4[0]
    temp__ = copy.deepcopy(temp_matrix4)
    temp_matrix5 = copy.deepcopy(temp_matrix4)
    temp_matrix6 = copy.deepcopy(temp_matrix4)
    if flag == 1:
        temp_matrix4 = p_matrix
#        temp_matrix5 = copy.deepcopy(temp_matrix4)
#        temp_matrix6 = copy.deepcopy(temp_matrix4)
        while count < 3 and len(temp_matrix5.coding_matrix)+1 < max_num_classifier:
            if len(talents_pool) != 0 :
                #精英池中选择
                random_index = random.randint(0,len(talents_pool)-1)
                choose_pool = talents_pool[random_index]        
                if not choose_pool.code_segment in temp_matrix5.coding_matrix:
                    temp_matrix5.coding_matrix.append(choose_pool.code_segment)
                    temp_matrix5.feature_matrix.append(choose_pool.feature_segment)
                    temp_matrix5.base_learner_matrix.append(choose_pool.estimator)
                    temp_matrix5 = PopLegality.checkMatrixsLegality([temp_matrix5])
                    temp_matrix5 = Valuate.calObjValue(PopInfors, temp_matrix5)[0]
                    distance3 = matrixDistance(temp_matrix5,class_size)
                    if distance3 < distance2:
                        count += 1
                    else:
                        count = 0
                        temp_matrix6 = copy.deepcopy(temp_matrix5)
                #精英池里面的column全部包含于该coding_matrix中，停止循环
                f = 0
                for talent in talents_pool:
                    if not talent.code_segment in temp_matrix5.coding_matrix:
                        f = 1
                        break
                if f == 0:
                    break
            else:
                break

    #新矩阵个体评价
#    temp_matrix6 = PopLegality.checkMatrixsLegality([temp_matrix6])
    temp_matrix6 = Valuate.calObjValue(PopInfors, [temp_matrix6])
    talents_pool, temp_matrix6 = Valuate.addToTalentsPool(PopInfors, temp_matrix6, talents_pool) 
    print('增加 ',temp_matrix6[0].f1score)
    if temp_matrix6[0].f1score > temp__.f1score:
        record_matrixs[0] = temp_matrix6[0]
    else:
        record_matrixs[0] = temp__
#    return record_matrixs
#    chrome = Initialization.coding(record_matrixs[0].coding_matrix,record_matrixs[0].feature_matrix,record_matrixs[0].base_learner_matrix)
#    return chrome
    return record_matrixs[0]
    
'''局部优化（新）添加多列'''
def localOptimization2(PopInfors,matrixs,talents_pool):
    class_size = PopInfors.class_size
    min_num_classifier = class_size - 1
    max_num_classifier = 2*class_size -1
    
    matrixs = sortMatrix(matrixs)
    record_matrixs = copy.deepcopy(matrixs)
    
    good = record_matrixs[0]
    good_coding_matrix = good.coding_matrix
    good_segment_accuracies = good.segment_accuracies
    good_segment_confusion_matrixs = good.segment_confusion_matrixs
    
    #找到最优个体分类最差的segment
    worst_segment_accuracy = float("inf") 
    worst_segment_accuracy_index = 0
    
    for i in range(len(good_segment_accuracies)):
        if good_segment_accuracies[i] < worst_segment_accuracy:
            worst_segment_accuracy = good_segment_accuracies[i]
            worst_segment_accuracy_index = i        
        
    #找到最优个体分类最差的segment最差的一类
    worst_confusion_matrix = good_segment_confusion_matrixs[worst_segment_accuracy_index] 
    worst_class_accuracy = float("inf") 
    worst_class_accuracy_index = 0   
    
    for i in range(len(worst_confusion_matrix)):
        if worst_confusion_matrix[i] < worst_class_accuracy:
            worst_class_accuracy_index = i
            worst_class_accuracy = worst_confusion_matrix[i]
 
    #找到其他矩阵中该类分类准确率最好的segment
    best_segment_accuracy = -1
    best_matrix_index = 1
    best_matrix_segment_index = 0
    for i in range(1,len(matrixs)):
        segment_confusion_matrixs = matrixs[i].segment_confusion_matrixs
        for j in range(len(segment_confusion_matrixs)):
            if segment_confusion_matrixs[j][worst_class_accuracy_index] > best_segment_accuracy:
                best_segment_accuracy = segment_confusion_matrixs[j][worst_class_accuracy_index]
                best_matrix_index = i
                best_matrix_segment_index = j
                
    find_matrix = matrixs[best_matrix_index]
    
    find_coding_segment = find_matrix.coding_matrix[best_matrix_segment_index]
    find_feature_segment = find_matrix.feature_matrix[best_matrix_segment_index]
    find_base_learner = find_matrix.base_learner_matrix[best_matrix_segment_index]
    '''局部优化操作'''
    chromes = []
    chrome = Initialization.coding(good.coding_matrix,good.feature_matrix,good.base_learner_matrix)
    chromes.append(chrome)
    #find列替换该列
    temp_matrix1 = copy.deepcopy(good)
    temp_matrix1.coding_matrix[worst_segment_accuracy_index] = find_coding_segment
    temp_matrix1.feature_matrix[worst_segment_accuracy_index] = find_feature_segment
    temp_matrix1.base_learner_matrix[worst_segment_accuracy_index] = find_base_learner
    chrome1 = Initialization.coding(temp_matrix1.coding_matrix,temp_matrix1.feature_matrix,temp_matrix1.base_learner_matrix)
    chromes.append(chrome1)

    #添加find列或者精英池中添加一segment
    if len(good_coding_matrix)+1 < max_num_classifier:
        distance0 = matrixDistance(good,class_size)
        #添加find列
        temp_matrix2 = copy.deepcopy(good)
        temp_matrix2.coding_matrix.append(find_coding_segment)
        temp_matrix2.feature_matrix.append(find_feature_segment)
        temp_matrix2.base_learner_matrix.append(find_base_learner)
        temp_matrix2 = Valuate.calObjValue(PopInfors, [temp_matrix2])[0]
        distance1 = matrixDistance(temp_matrix2,class_size)
        
        flag = 0
        change_f1score = 0
        p_matrix= []
        if distance1 > distance0:#继续添加column,直至distance开始减少
            temp_matrix4 =  copy.deepcopy(temp_matrix2)
            while change_f1score >= 0 and len(temp_matrix4.coding_matrix)+1 < max_num_classifier:
                flag = 0
                if len(talents_pool) != 0 :
                    #精英池中选择                
                    for talent in talents_pool:
                        if not talent.code_segment in temp_matrix4.coding_matrix:
                            flag = 1
                            break
                    
                    if flag == 0:
                        break
                    else:
                        random_index = random.randint(0,len(talents_pool)-1)
                        choose_pool = talents_pool[random_index]        
                        p_distance = matrixDistance(temp_matrix4,class_size)
                        p_matrix = copy.deepcopy(temp_matrix4)
                        while choose_pool.code_segment in temp_matrix4.coding_matrix:
                            random_index = random.randint(0,len(talents_pool)-1)
                            choose_pool = talents_pool[random_index]        
                        temp_matrix4.coding_matrix.append(choose_pool.code_segment)
                        temp_matrix4.feature_matrix.append(choose_pool.feature_segment)
                        temp_matrix4.base_learner_matrix.append(choose_pool.estimator)
                        chrome_ = Initialization.coding(temp_matrix4.coding_matrix,temp_matrix4.feature_matrix,temp_matrix4.base_learner_matrix)
                        temp_matrix4 = PopLegality.checkLegality([chrome_],PopInfors)
                        temp_matrix4 = Valuate.calObjValue(PopInfors, temp_matrix4)[0]
                        distance2 = matrixDistance(temp_matrix4,class_size)
                        print(distance2)
                        change_f1score = distance2 - p_distance
                else:
                    break
            #若连续三个column添加进来distance均减少，则终止
            count = 0       
            temp_matrix5 = copy.deepcopy(temp_matrix4)
            temp_matrix6 = copy.deepcopy(temp_matrix4)
            if not flag == 0:
                temp_matrix4 = p_matrix
                temp_matrix5 = copy.deepcopy(temp_matrix4)
                temp_matrix6 = copy.deepcopy(temp_matrix4)
                while count < 3 and len(temp_matrix5.coding_matrix)+1 < max_num_classifier:
                    if len(talents_pool) != 0 :
                        #精英池中选择
                        random_index = random.randint(0,len(talents_pool)-1)
                        choose_pool = talents_pool[random_index]        
                        if not choose_pool.code_segment in temp_matrix5.coding_matrix:
                            temp_matrix5.coding_matrix.append(choose_pool.code_segment)
                            temp_matrix5.feature_matrix.append(choose_pool.feature_segment)
                            temp_matrix5.base_learner_matrix.append(choose_pool.estimator)
                            chrome_ = Initialization.coding(temp_matrix5.coding_matrix,temp_matrix5.feature_matrix,temp_matrix5.base_learner_matrix)
                            temp_matrix5 = PopLegality.checkLegality([chrome_],PopInfors)
                            temp_matrix5 = Valuate.calObjValue(PopInfors, temp_matrix5)[0]
                        distance3 = matrixDistance(temp_matrix5,class_size)
                        if distance3 < distance2:
                            count += 1
                            print('局部优化:::',count)
                        else:
                            count = 0
                            temp_matrix6 = copy.deepcopy(temp_matrix5)
                        #精英池里面的column全部包含于该coding_matrix中，停止循环
                        f = 0
                        for talent in talents_pool:
                            if not talent.code_segment in temp_matrix5.coding_matrix:
                                f = 1
                                break
                        if f == 0:
                            break
                    else:
                        break
            
            chrome6 = Initialization.coding(temp_matrix6.coding_matrix,temp_matrix6.feature_matrix,temp_matrix6.base_learner_matrix)
            chromes.append(chrome6)
        else:
            chrome2 = Initialization.coding(temp_matrix2.coding_matrix,temp_matrix2.feature_matrix,temp_matrix2.base_learner_matrix)
            chromes.append(chrome2)
    
    #删除该列
    if len(good_coding_matrix)-1 > min_num_classifier and len(good_coding_matrix)-1 > 1:
        temp_matrix3 = copy.deepcopy(good)
        del temp_matrix3.coding_matrix[worst_segment_accuracy_index]
        del temp_matrix3.feature_matrix[worst_segment_accuracy_index]
        del temp_matrix3.base_learner_matrix[worst_segment_accuracy_index]
        chrome3 = Initialization.coding(temp_matrix3.coding_matrix,temp_matrix3.feature_matrix,temp_matrix3.base_learner_matrix)
        chromes.append(chrome3)
    

    
    new_good = chooseGood(PopInfors, chromes, talents_pool)
    new_good = [new_good]
    #新矩阵个体评价
    new_good = Valuate.calObjValue(PopInfors, new_good)
    talents_pool, new_good = Valuate.addToTalentsPool(PopInfors, new_good, talents_pool)
    
    record_matrixs[0] = new_good[0]
    return record_matrixs
    
'''（计算矩阵间行距离）判断操作之后的矩阵是否优化'''
def matrixDistance(matrix,class_size):
#    transpose_coding_matrix = np.transpose(coding_matrix)
#    distance = 0
#    for i in range(class_size):
#        for k in range(class_size):
#            distance += Valuate.hammingDistance(transpose_coding_matrix[i],transpose_coding_matrix[k])
#    distance = np.float(distance/(class_size*(class_size-1)))
    distance = matrix.f1score
    return distance
    
'''局部优化'''
def localOptimization1(PopInfors,matrixs,talents_pool):
    class_size = PopInfors.class_size
    min_num_classifier = class_size - 1
    max_num_classifier = class_size * 2
    
    matrixs = sortMatrix1(matrixs)
    record_matrixs = copy.deepcopy(matrixs)
    record_matrixs1 = copy.deepcopy(matrixs)
    
    good = record_matrixs[0]
    good_coding_matrix = good.coding_matrix
    good_segment_accuracies = good.segment_accuracies
    good_segment_confusion_matrixs = good.segment_confusion_matrixs
    
    #找到最优个体分类最差的segment
    worst_segment_accuracy = float("inf") 
    worst_segment_accuracy_index = 0
    
    for i in range(len(good_segment_accuracies)):
        if good_segment_accuracies[i] < worst_segment_accuracy:
            worst_segment_accuracy = good_segment_accuracies[i]
            worst_segment_accuracy_index = i        
        
    #找到最优个体分类最差的segment最差的一类
    worst_confusion_matrix = good_segment_confusion_matrixs[worst_segment_accuracy_index] 
    worst_class_accuracy = float("inf") 
    worst_class_accuracy_index = 0   
    
    for i in range(len(worst_confusion_matrix)):
        if worst_confusion_matrix[i] < worst_class_accuracy:
            worst_class_accuracy_index = i
            worst_class_accuracy = worst_confusion_matrix[i]
 
    #找到其他矩阵中该类分类准确率最好的segment
    best_segment_accuracy = -1
    best_matrix_index = 1
    best_matrix_segment_index = 0
    for i in range(1,len(matrixs)):
        segment_confusion_matrixs = matrixs[i].segment_confusion_matrixs
        for j in range(len(segment_confusion_matrixs)):
            if segment_confusion_matrixs[j][worst_class_accuracy_index] > best_segment_accuracy:
                best_segment_accuracy = segment_confusion_matrixs[j][worst_class_accuracy_index]
                best_matrix_index = i
                best_matrix_segment_index = j
                
    find_matrix = matrixs[best_matrix_index]
    
    find_coding_segment = find_matrix.coding_matrix[best_matrix_segment_index]
    find_feature_segment = find_matrix.feature_matrix[best_matrix_segment_index]
    find_base_learner = find_matrix.base_learner_matrix[best_matrix_segment_index]
#    '''局部优化操作'''
    chromes = []
    chrome = Initialization.coding(good.coding_matrix,good.feature_matrix,good.base_learner_matrix)
    chromes.append(chrome)
    #find列替换该列
    temp_matrix1 = copy.deepcopy(good)
    temp_matrix1.coding_matrix[worst_segment_accuracy_index] = find_coding_segment
    temp_matrix1.feature_matrix[worst_segment_accuracy_index] = find_feature_segment
    temp_matrix1.base_learner_matrix[worst_segment_accuracy_index] = find_base_learner
    chrome1 = Initialization.coding(temp_matrix1.coding_matrix,temp_matrix1.feature_matrix,temp_matrix1.base_learner_matrix)
    chromes.append(chrome1)
    
    #突变基分类器,后来添加
    temp_matrix_add = copy.deepcopy(good)
    num_base_learners = len(PopInfors.base_learners)
    random_index = random.randint(0,num_base_learners-1)
    learner0 = temp_matrix_add.base_learner_matrix[worst_segment_accuracy_index] 
    while learner0 == random_index:
        random_index = random.randint(0,num_base_learners-1)
    temp_matrix_add.base_learner_matrix[worst_segment_accuracy_index] = random_index
    chrome_add = Initialization.coding(temp_matrix_add.coding_matrix,temp_matrix_add.feature_matrix,temp_matrix_add.base_learner_matrix)
    chromes.append(chrome_add)
    
    #添加find列或者精英池中添加一segment
    if len(good_coding_matrix)+1 < max_num_classifier:
        #添加find列
        temp_matrix2 = copy.deepcopy(good)
        temp_matrix2.coding_matrix.append(find_coding_segment)
        temp_matrix2.feature_matrix.append(find_feature_segment)
        temp_matrix2.base_learner_matrix.append(find_base_learner)
        chrome2 = Initialization.coding(temp_matrix2.coding_matrix,temp_matrix2.feature_matrix,temp_matrix2.base_learner_matrix)
        chromes.append(chrome2)

        if len(talents_pool) != 0 :
            #精英池中选择
            random_index = random.randint(0,len(talents_pool)-1)
            choose_pool = talents_pool[random_index]        
            temp_matrix4 = copy.deepcopy(good)
            temp_matrix4.coding_matrix.append(choose_pool.code_segment)
            temp_matrix4.feature_matrix.append(choose_pool.feature_segment)
            temp_matrix4.base_learner_matrix.append(choose_pool.estimator)
            chrome4 = Initialization.coding(temp_matrix4.coding_matrix,temp_matrix4.feature_matrix,temp_matrix4.base_learner_matrix)
            chromes.append(chrome4)
 
    #删除该列
    if len(good_coding_matrix)-1 > min_num_classifier and len(good_coding_matrix)-1 > 1:
        temp_matrix3 = copy.deepcopy(good)
        del temp_matrix3.coding_matrix[worst_segment_accuracy_index]
        del temp_matrix3.feature_matrix[worst_segment_accuracy_index]
        del temp_matrix3.base_learner_matrix[worst_segment_accuracy_index]
        chrome3 = Initialization.coding(temp_matrix3.coding_matrix,temp_matrix3.feature_matrix,temp_matrix3.base_learner_matrix)
        chromes.append(chrome3)
    
    
    best_temp = localOptimization(PopInfors,record_matrixs1,talents_pool)
        
    new_good = chooseGood(PopInfors, chromes, talents_pool)
    new_good = [new_good]
    #新矩阵个体评价
    new_good = Valuate.calObjValue(PopInfors, new_good)
    talents_pool, new_good = Valuate.addToTalentsPool(PopInfors, new_good, talents_pool)
    
    record_matrixs[0] = new_good[0]
    print(record_matrixs[0].f1score)
    if record_matrixs[0].f1score < best_temp.f1score:
        record_matrixs[0] = copy.deepcopy(best_temp)
    return record_matrixs
       
#选出局部优化之后最好的个体
def chooseGood(PopInfors, chromes, talents_pool):
    matrixs = PopLegality.checkLegality(chromes,PopInfors)
    matrixs = Valuate.calObjValue(PopInfors, matrixs)
    talents_pool, matrixs = Valuate.addToTalentsPool(PopInfors, matrixs, talents_pool)
    matrixs = sortMatrix1(matrixs)
    return matrixs[0]

'''将矩阵按照fitness从高到低排序'''
def sortMatrix1(matrixs):
    for i in range(len(matrixs) - 1):
        for j in range(len(matrixs)-1-i):
            if matrixs[j].f1score < matrixs[j+1].f1score:
                matrixs[j],matrixs[j+1] = matrixs[j+1],matrixs[j]
    return matrixs
                
    
'''将矩阵按照新的fitness,diversity从高到低排序'''
def sortMatrix(matrixs):
    new_matrixs = DeapSelect.newSort(matrixs)
    return new_matrixs

def renewMatrix(parent1,parent2,index1,index2):
    coding_matrix1 = parent1.coding_matrix
    coding_matrix2 = parent2.coding_matrix
    feature_matrix1 = parent1.feature_matrix
    feature_matrix2 = parent2.feature_matrix
    base_learner_matrix1 = parent1.base_learner_matrix
    base_learner_matrix2 = parent2.base_learner_matrix
    #交换该segment
    coding_matrix1[index1],coding_matrix2[index2] = coding_matrix2[index2],coding_matrix1[index1]
    
    feature_matrix1[index1],feature_matrix2[index2] = feature_matrix2[index2],feature_matrix1[index1]
    
    base_learner_matrix1[index1],base_learner_matrix2[index2] = base_learner_matrix2[index2],base_learner_matrix1[index1]
    
    child1 = parent1
    child2 = parent2
    
    child1.coding_matrix = coding_matrix1
    child2.coding_matrix = coding_matrix2
    child1.feature_matrix = feature_matrix1
    child2.feature_matrix = feature_matrix2
    child1.base_learner_matrix = base_learner_matrix1
    child2.base_learner_matrix = base_learner_matrix2
    children = []
    children.append(child1)
    children.append(child2)

    return children
        
        

