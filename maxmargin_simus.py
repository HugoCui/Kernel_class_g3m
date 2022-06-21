# Simulations for artificial Gaussian datasets, max-margin classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
parser=argparse.ArgumentParser(description="Job launcher")
from scipy.special import erf


import sys
parser.add_argument("-l",type=float)#lambda, regularization
parser.add_argument("-a",type=float)#alpha, capacity
parser.add_argument("-r",type=float)#r, source
parser.add_argument("-p",type=int)#p
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

lamb = 1e-4  #Vanishing regularization for max-margin.





spec_Omega = np.array([p/(k+1)**alph for k in range(p)])
Omega=np.diag(spec_Omega)
Phi = spec_Omega
Psi = spec_Omega

teacher = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(p)]))


rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU = Phi**2 * teacher**2


Phi=np.diag(Phi)




def get_instance(*, samples,seed):
    np.random.seed(seed)  
    X = np.sqrt(spec_Omega) * np.random.normal(0,1,(samples, p))
    y = np.sign(X @ teacher / np.sqrt(p))
    
    return X/np.sqrt(p), y
    


def logistic_estimator(*, X, y, lamb):
    w = LogisticRegression(penalty='l2',C=1./lamb, fit_intercept=False,solver='lbfgs',random_state=0,max_iter=10000, tol=0.00001).fit(X,y).coef_[0]
    return w


def simulate(samples, lamb, seed, estimator="logistic"):
    verbose=False
    n = samples
    
   
    V_train, y_train = get_instance(samples = n,seed=seed)
    
    V_test, y_test = get_instance(samples = n,seed=3*seed)
    
    if estimator=="logistic":
        w = logistic_estimator(X = V_train, y = y_train, lamb = lamb)
    elif estimator=="SVM":
        w = SVM_estimator(X = V_train, y = y_train, lamb = lamb)
    
            
    yhat_train = np.sign(V_train @ w)
    yhat_test = np.sign(V_test @ w)
    
    
    train_error = 1-np.mean(yhat_train == y_train)
    test_error  = 1- np.mean(yhat_test == y_test)

    q=np.dot(w, Omega @ w) / p
    m=np.dot(w, Phi @ teacher)/ np.sqrt(p*d)
    
    return [train_error,test_error,q,m,alph,r,p,lamb,rho,samples,seed]

   


pool=mp.Pool(mp.cpu_count())
results=pool.starmap_async(simulate,[(samples,lamb,seed) for seed in range(40)]).get()
pool.close()
results=np.array(results)

keys=["train_error","test_error","q","m","alpha","r","p","lambda","rho","samples","seed"]

simus=pd.DataFrame(results, columns=keys)
simus["sample_complexity"]=simus["samples"]/p

simus.to_csv("data/simus/simus_maxmargin_alpha{}_r{}_p{}_samples{}.csv".format(alph,r,p,samples))
