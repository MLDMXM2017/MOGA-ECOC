import array
import random
import json
import ThreeMatrix
import numpy
import copy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools


def evaluate(individual):
    return -individual.f1score,-individual.distance

def returnThreeMatrix():
    threeMatrix = ThreeMatrix.ThreeMatrix(-1,-1,-1,-1,-1,-1,-1)
    return threeMatrix

# threeMatrix1 = ThreeMatrix.ThreeMatrix(1,1,1,2,2,1,1)
# threeMatrix2 = ThreeMatrix.ThreeMatrix(2,2,2,1,1,2,2)
# threeMatrix3 = ThreeMatrix.ThreeMatrix(3,3,3,3,3,3,3)
threeMatrix1 = ThreeMatrix.ThreeMatrix(1,1,1,1,1,1,1)
threeMatrix2 = ThreeMatrix.ThreeMatrix(1,1,1,1,1,1,1)
threeMatrix3 = ThreeMatrix.ThreeMatrix(1,1,1,1,1,1,1)
threeMatrixs = []
threeMatrixs.append(threeMatrix1)
threeMatrixs.append(threeMatrix2)
threeMatrixs.append(threeMatrix3)

IND_SIZE = 1
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.register("matrix", returnThreeMatrix)
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.matrix, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#toolbox.register("select", tools.selNSGA2)
toolbox.register("select", tools.selSPEA2)
#toolbox.register("select", tools.selBest)
#toolbox.register("select", tools.selTournamentDCD)??
#toolbox.register("select", tools.selRoulette)??
#toolbox.register("select", tools.sortNondominated)??


def transformMatrixs(threeMatrixs):
    pop = toolbox.population(len(threeMatrixs))
    
    for i in range(len(threeMatrixs)):
        pop[i][0].coding_matrix = threeMatrixs[i].coding_matrix
        pop[i][0].feature_matrix = threeMatrixs[i].feature_matrix
        pop[i][0].base_learner_matrix = threeMatrixs[i].base_learner_matrix
        pop[i][0].f1score = threeMatrixs[i].f1score
        pop[i][0].distance = threeMatrixs[i].distance
        pop[i][0].segment_accuracies = threeMatrixs[i].segment_accuracies
        pop[i][0].segment_confusion_matrixs = threeMatrixs[i].segment_confusion_matrixs
    
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]  
    
    for i in range(len(invalid_ind)):
        ind = invalid_ind[i]
        ind.fitness.values = evaluate(ind[0])
    
    return pop

'''重新定义最优个体的选择'''
def selectBest1(threeMatrixs):
#    pop = transformMatrixs(threeMatrixs)
#    best = toolbox.select(pop, 1)[0]
#    return best[0]
    best = copy.deepcopy(threeMatrixs[0])
    for i in range(len(threeMatrixs)):
        if threeMatrixs[i].f1score == best.f1score:
            best = copy.deepcopy(selectBest(threeMatrixs))
        elif threeMatrixs[i].f1score > best.f1score:
            best = copy.deepcopy(threeMatrixs[i])
    return best


def selectBest(threeMatrixs):
    pop = transformMatrixs(threeMatrixs)
    best = toolbox.select(pop, 1)[0]
    return best[0]


def newSort(threeMatrixs):
    new_matrixs = []
    pop = transformMatrixs(threeMatrixs)
    pop = toolbox.select(pop, len(pop))
    for i in range(len(pop)):
        new_matrixs.append(pop[i][0])
    return new_matrixs

#best = selectBest(threeMatrixs)
#newSort(threeMatrixs)
