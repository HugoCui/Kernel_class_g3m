#simulations curves for artificial Gaussian dataset, \ell_2 classification.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
parser=argparse.ArgumentParser(description="Job launcher")
from scipy.special import erf
from sklearn.metrics import make_scorer

import sys
parser.add_argument("-a",type=float)#alpha
parser.add_argument("-r",type=float)#r
parser.add_argument("-p",type=int)#p
parser.add_argument("-s",type=float)#sigma
parser.add_argument("-v",type=int)#sample complexity
args=parser.parse_args()
samples_range=np.logspace(1,np.log10(args.p),11)
samples=int(np.round(samples_range[args.v]))

sys.path.insert(1, 'g3m_utils/')
from state_evolution.data_models.custom import CustomSpectra
from state_evolution.experiments.learning_curve import CustomExperiment
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier, LogisticRegression


p = args.p
alph=args.a
r=args.r 
k = p
d=p

gamma = k/p
sig=args.s





spec_Omega = np.array([p/(k+1)**alph for k in range(p)])
Omega=np.diag(spec_Omega)
Phi = spec_Omega
Psi = spec_Omega

teacher = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(p)]))


rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU = Phi**2 * teacher**2


Phi=np.diag(Phi)




def get_instance(*, samples,seed):  #creates a dataset
    np.random.seed(seed)
    X = np.sqrt(spec_Omega) * np.random.normal(0,1,(samples, p))
    np.random.seed(seed+76545)
    y = np.sign(X @ teacher / np.sqrt(p)+sig*np.random.randn(samples))
    
    return X/np.sqrt(p), y
    
def score(yh,y):
    return np.sum(np.sign(yh)*np.sign(y))

score=make_scorer(score, greater_is_better=True)

def ridge_estimator(X, y, lamb):
    '''
    Implements the pseudo-inverse ridge estimator.
    '''
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y


# Simulate the problem for a given sample complexity and average over given number of seeds.
def simulate(samples,  seed, estimator="logistic"):
    verbose=False
    n = samples
    #lamb=n*n**(-ell)
   
    V_train, y_train = get_instance(samples = n,seed=seed*756)
    
    V_test, y_test = get_instance(samples = n,seed=seed*123)
    
    
    reg = linear_model.RidgeCV(alphas=np.hstack((np.logspace(-5,1.8,100),np.array([1e-15]))),scoring=score)
    reg.fit(V_train,y_train)
    w=np.array(reg.coef_)
    #print(w)
    #print(reg.coef_)
    yhat_train = np.sign(V_train @ w )
    yhat_test =np.sign( V_test @ w )
    
    lamb=reg.alpha_    
    
    train_error = 1-np.mean(yhat_train == y_train)
    test_error  = 1- np.mean(yhat_test == y_test)-np.arccos(np.sqrt((rho/(rho+sig**2))))/np.pi

    q=np.dot(w, Omega @ w) / p
    m=np.dot(w, Phi @ teacher)/ np.sqrt(p*d)
    
    return [train_error,test_error,q,m,alph,r,p,lamb,rho,samples,seed,sig]

   


pool=mp.Pool(mp.cpu_count())
results=pool.starmap_async(simulate,[(samples,seed) for seed in range(80)]).get()
pool.close()
results=np.array(results)

keys=["train_error","test_error","q","m","alpha","r","p","lambda","rho","samples","seed","sigma"]

simus=pd.DataFrame(results, columns=keys)
simus["sample_complexity"]=simus["samples"]/p

simus.to_csv("data/CV_l2_simus/simus_l2_alpha{}_r{}_p{}_samples{}_sigma{}.csv".format(alph,r,p,samples,sig))
