
#Simulations for real datasets, \ell_2 classification.
# The datasets are standard and can be readily found online. They are not included in the repo because of space limitations. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
parser=argparse.ArgumentParser(description="Job launcher")
from scipy.special import erf
from sklearn.kernel_ridge import KernelRidge


import sys
parser.add_argument("-g", type=float) #gamma parameter
parser.add_argument("-k", type=str)  #kernel
parser.add_argument("-v", type=int) #index of sample
parser.add_argument("-d", type=str) #dataset
args=parser.parse_args()


sys.path.insert(1, 'g3m_utils/')
from state_evolution.data_models.custom import CustomSpectra
from state_evolution.experiments.learning_curve import CustomExperiment
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import make_scorer



gamma=args.g #gamma parameter for the RBF/poly kernel

kernel=args.k

dataset=args.d
#Loading the dataset
X=np.load("datasets/{}_X.npy".format(dataset))
y=np.load("datasets/{}_y.npy".format(dataset))

n, p = X.shape


samples_range=np.logspace(1,np.log10(n)-.4,11)
samples=int(np.round(samples_range[args.v]))

def score(yh,y):
    return np.sum(np.sign(yh)*np.sign(y))

score=make_scorer(score, greater_is_better=True)

# Simulate the problem for a given sample complexity and average over given number of seeds.
def simulate(samples,  seed, estimator="logistic"):
    verbose=False
    n = samples
   
    
    np.random.seed(seed)
    inds = np.random.choice(range(X.shape[0]), size=samples, replace=False)
   
    X_train = X[inds, :] # training data
    y_train = y[inds] # training labels

    X_test = X # test data, using the whole dataset. See also Canatar et al., Nature 2020 and Loureiro et al., NIPS 2021
    y_test = y # test labels
    
    if kernel=="polynomial":
        reg= GridSearchCV(KernelRidge(kernel=kernel,degree=5,gamma=gamma),
                param_grid={"alpha": Grid_}
                        ,scoring=score)
    else:
        reg= GridSearchCV(KernelRidge(kernel=kernel,gamma=gamma),
                    param_grid={"alpha": Grid_}
                            ,scoring=score)
    
    reg.fit(X_train,y_train)
    yhat_train = np.sign(reg.predict(X_train))
    yhat_test =np.sign( reg.predict(X_test))
    
    lamb=reg.best_params_["alpha"]   
    
    train_error = 1-np.mean(yhat_train == y_train)
    test_error  = 1- np.mean(yhat_test == y_test)

    
    return [train_error,test_error,lamb,samples,seed]

   
Grid_=np.array([0]+list(np.logspace(-8,3,500)))

pool=mp.Pool(mp.cpu_count())
results=pool.starmap_async(simulate,[(samples,seed) for seed in range(80)]).get()
pool.close()
results=np.array(results)

keys=["train_error","test_error","lambda","samples","seed"]

simus=pd.DataFrame(results, columns=keys)
simus["sample_complexity"]=simus["samples"]/p

simus.to_csv("data/Real_l2/Simusl2_{}_{}_gamma{}_samples{}.csv".format(dataset,kernel,gamma,samples))
