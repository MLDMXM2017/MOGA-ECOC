# MOGA-ECOC

Multi-Objective Genetic Algorithm Based Error Correcting Output Codes  
This is the implementation for paper: [A Novel Multi-Objective Genetic Algorithm Based Error Correcting Output Codes]

## Acknowledgement
- Our Genetic Algorithm is modified from [DEAP 1.3.0](https://deap.readthedocs.io/en/master/)
- The classifier is modified from [scikit-learn 0.22](https://scikit-learn.org/stable/)

## Environment
- **Windows 10 64 bit**
- **Python 3**
- **DEAP 1.3.0**  
DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas. One can install it by this way:  `easy_install deap` or `pip install deap`  
- **scikit-learn 0.22**  
[Anaconda](https://www.anaconda.com/) is strongly recommended, all necessary python packages for this project are included in the Anaconda.

## Function
By using the DEAP framwork, each step of genetic algorithm is implemented as follows:
- Initialization.intialization: create the first generation
- Operate.crossover: cross matrix by column
- Operate.mutation: mutate individual by some bits on column
- PopLegality.checkMatrixsLegality: legalize matrix according to rules
- Valuate.calObjValue: calculate multi-objective f1-score/diversity/instance
- Operate.eliteRetention: keep elite

## Dataset
- **Data format**  
Data sets are included in the folder `($root_path)/data_uci`. All the data sets used in experiments are splited into three parts: 
`xx_train.data`、`xx_validation.data` and `xx_test.data`. And the proportion of these three parts is set to 2:1:1. 
These datasets are formed by setting each row as a feature and each column as an instance.  
The dataset included in the folder must have no null/nan/? values.
- **Data processing**
Feature Selection and Scaling will be done automatically.

## Runner Setup
The `Runner.py`  calls the MOGA. As the paper mentioned, there are some running modes can be used.  

The variable `base_learners` in the `Runner.py` can control the base classifiers in algorithm. 
If one uses **Homogeneous Ensembles**, only one type of classifier can be used.
By using **Heterogeneous Ensembles**, some different types of classifiers provided in scikit-learn package can be used to form a heterogeneous ensemble.  

The three Objectives used to evaluate the matrix are: F-score、distance and pairwise diversity between columns in the ECOC Matrix. 
One can adjust the proportion between these three objectives by setting `DeapSelect.py`.  

The results are collected and written in the filefolder `records_NSGA`.
