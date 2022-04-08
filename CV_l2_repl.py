#!/usr/bin/env python
# coding: utf-8

# # Logistic regression on the G$^3$M model

# In[55]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
parser=argparse.ArgumentParser(description="Job launcher")
from scipy.special import erf
from scipy.integrate import quad,nquad
from scipy.optimize import minimize_scalar
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.model_selection import train_test_split

#from sklearn.kernel_ridge import KernelRidge
import sys

epsabs = 1e-10
epsrel = 1e-10
limit = 100
#parser.add_argument("-l",type=float)#lamb
parser.add_argument("-a",type=float)#alpha
parser.add_argument("-r",type=float)#r
parser.add_argument("-s",type=float)#sigma
parser.add_argument("-p",type=int)#p
args=parser.parse_args()


sys.path.insert(1, 'g3m_utils/')
from state_evolution.data_models.custom import CustomSpectra
from state_evolution.experiments.learning_curve import CustomExperiment
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier, LogisticRegression

# %load_ext autoreload
# %autoreload 2


# ## Global Variables

# In[56]:


# Dimensions
p = args.p
alph=args.a
r=args.r 
sig=args.s

k = p
d=p
 #Noise variance
gamma = k/p

# Regularisation
#ell = args.l


# ## Replicas

# In[3]:

alphas=np.logspace(1,np.log10(p),10)/p
l=len(alphas)



spec_Omega = np.array([p/(k+1)**alph for k in range(p)])
#Omega=np.diag(spec_Omega0)
Phi = spec_Omega
Psi = spec_Omega

teacher = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(p)]))


# In[5]:


rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU = Phi**2 * teacher**2

def error_lam(log_lam,alpha):
    lam=np.exp(-np.log(10)*log_lam)
    data_model = CustomSpectra(gamma = 1,
                           rho = rho+sig**2, 
                           spec_Omega = spec_Omega, 
                           spec_UPhiPhitUT = spec_Omega**2 * teacher**2)
    experiment = CustomExperiment(task = 'l2_classification', 
                              regularisation = lam, 
                              data_model = data_model, 
                              tolerance = 1e-16, 
                              damping = 0.3, 
                              verbose = False, 
                              max_steps = 1000)
    experiment.learning_curve(alphas = np.array([alpha]))
    error=experiment.get_curve()["test_error"][0]
    return error
    



def get_all(alpha, iter):   
    replicas={"lamb":[],"samples":[],"test_error":[], "a":[],"b":[],"c":[],"lamb0":[],"sigma":[],"p":[]}
    #samples=np.logspace(1,np.log10(p),20)
    #alphas = samples/p#np.linspace(0.01, .8, 5)

    minimization=minimize_scalar(error_lam, (-3,0),bounds=(-3,0), method="bounded",args=(alpha),tol=1e-10)
    log_lamb=minimization.x
    lamb=10**(-log_lamb)
    data_model = CustomSpectra(gamma = 1,
                           rho = rho+sig**2, 
                           spec_Omega = spec_Omega, 
                           spec_UPhiPhitUT = spec_Omega**2 * teacher**2) #rho<- rho+sigma**2 is a conveninent artificial way to include noise

    experiment = CustomExperiment(task = 'l2_classification', 
                                  regularisation = lamb, 
                                  data_model = data_model, 
                                  tolerance = 1e-6, 
                                  damping = 0.3, 
                                  verbose = True, 
                                  max_steps = 1000)
    experiment.learning_curve(alphas =[alpha])
    replicas=experiment.get_curve()
    replicas["eta"]=replicas["m"]**2/(rho*replicas["q"])
    replicas["test_error"]=np.arccos(np.sqrt((rho/(rho+sig**2))*replicas["eta"]))/np.pi-np.arccos(np.sqrt((rho/(rho+sig**2))))/np.pi
    print(replicas)
    return list(replicas.values[0])+[sig]



pool=mp.Pool(mp.cpu_count())
results=pool.starmap_async(get_all,[(alphas[i],i) for i in range(l)]).get()
pool.close()
#results=np.array(results).astype(np.float)

keys=['task',
 'gamma',
 'lambda',
 'rho',
 'sample_complexity',
 'V',
 'm',
 'q',
 'Vhat',
 'mhat',
 'qhat',
 'test_error',
 'train_loss',"eta",
 "sigma"
]



replicas=pd.DataFrame(data=results, columns=keys)


#print(replicas)
#print(results)




replicas["eta"]=replicas["m"].values/np.sqrt((rho)*replicas["q"].values)
replicas["r1"]=replicas["mhat"].values/replicas["Vhat"].values
replicas["r2"]=replicas["sample_complexity"].values*replicas["qhat"].values/replicas["Vhat"].values**2
replicas["z"]=replicas["sample_complexity"].values**2*replicas["lambda"].values*p/replicas["Vhat"].values
replicas["alpha"]=alph
replicas["r"]=r
replicas["p"]=p




replicas.to_csv("data/CV_l2_repl/l2_p{}_alpha{}_r{}_sigma{}.csv".format(p,alph,r,sig))




