# g3m (theory) curves for max-margin classification.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
parser=argparse.ArgumentParser(description="Job launcher")
from scipy.special import erf
from scipy.integrate import quad,nquad
import sys

epsabs = 1e-10
epsrel = 1e-10
limit = 100
parser.add_argument("-l",type=float)#lamb
parser.add_argument("-a",type=float)#alpha
parser.add_argument("-r",type=float)#r
parser.add_argument("-p",type=int)#p
args=parser.parse_args()


sys.path.insert(1, 'g3m_utils/')  #g3m package, see Loureiro et al., NIPS 2021
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
 #Noise variance
gamma = k/p

# Regularisation
lamb = args.l
print("lambda=",lamb)


alphas=np.logspace(1,np.log10(p),10)/p
l=len(alphas)



spec_Omega0 = np.array([p/(k+1)**alph for k in range(p)])
Omega0=np.diag(spec_Omega0)
Phi = spec_Omega0
Psi = spec_Omega0

teacher = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(p)]))
rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU0 = Phi**2 * teacher**2



def gaussian(x, mean=0, var=1):
    return np.exp(-.5 * (x-mean)**2/var) / np.sqrt(2*np.pi*var)

def loss(z):
    return max(0,1-z)

def Getf(ω,y,V):
    return (1-V > ω*y)*y+(1-V < ω*y)*(ω*y <1)*(y-ω)/V

# Mhat_x #
def integrate_for_mhat(M, sqQ, V, sigma, lamb=1, int_lim=np.inf):
    if(lamb==0):
        print("entered if lambda==0")
        I1 = 0
        I2 = quad(lambda ξ: 2*gaussian(ξ)*gaussian(M*ξ/sqQ,var=sigma)*(1-sqQ*ξ)/V, -int_lim, 1./sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    else:
        I1 = quad(lambda ξ: 2*gaussian(ξ)*gaussian(M*ξ/sqQ,var=sigma),-int_lim, (1-V)/sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        I2 = quad(lambda ξ: 2*gaussian(ξ)*gaussian(M*ξ/sqQ,var=sigma)*(1-sqQ*ξ)/V,(1-V)/sqQ,1./sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    return I1+I2

# Vhat_x #
def integrate_for_Vhat(M, sqQ, V, sigma, lamb=1, int_lim=np.inf):
    if(lamb==0):
        I = quad(lambda ξ: (1+erf(M*ξ/(sqQ*np.sqrt(2*sigma))))*gaussian(ξ), -int_lim, 1./sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    else:
        I = quad(lambda ξ: (1+erf(M*ξ/(sqQ*np.sqrt(2*sigma))))*gaussian(ξ), (1-V)/sqQ, 1./sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    return -I/V

# Qhat_x#
def integrate_for_Qhat(M, sqQ, V, sigma, lamb=1, int_lim=np.inf):
    if(lamb==0):
        I1 = 0
        I2 = quad(lambda ξ: (1+erf(M*ξ/(sqQ*np.sqrt(2*sigma)))) * gaussian(ξ) *((1-sqQ*ξ)/V)**2,-np.inf, 1./sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    else:
        I1 = quad(lambda ξ: (1+erf(M*ξ/(sqQ*np.sqrt(2*sigma)))) * gaussian(ξ), -int_lim, (1-V)/sqQ)[0]
        I2 = quad(lambda ξ: (1+erf(M*ξ/(sqQ*np.sqrt(2*sigma)))) * gaussian(ξ) *((1-sqQ*ξ)/V)**2,(1-V)/sqQ, 1./sqQ, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    return (I1 + I2)

def Integrand_training_loss(ξ, M, Q, V, Vstar, y):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar = V*Getλ(ω,y,V)+ω
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * loss(λstar)

def traning_loss(M, Q, V, Vstar):
    I1 = quad(lambda ξ: Integrand_training_error_logistic(ξ, M, Q, V, Vstar, 1) * gaussian(ξ), -10, 10)[0]
    I2 = quad(lambda ξ: Integrand_training_error_logistic(ξ, M, Q, V, Vstar,-1) * gaussian(ξ), -10, 10)[0]
    return (1/2)*(I1 + I2)

# In[161]:


def integrate_for_qvm(qhat,mhat,Vhat,lamb):
    if lamb!=0:

        V = np.mean(spec_Omega0/(lamb + Vhat * spec_Omega0))


        q = np.mean((spec_Omega0**2 * qhat +
                                       mhat**2 * spec_Omega0 * diagUtPhiPhitU0) /
                                      (lamb + Vhat*spec_Omega0)**2)

        m = mhat/np.sqrt(gamma) * np.mean(diagUtPhiPhitU0/(lamb + Vhat*spec_Omega0))

    else:
        V= np.mean(spec_Omega0/(1+ Vhat * spec_Omega0))


        q = np.mean((spec_Omega0**2 * qhat +
                                       mhat**2 * spec_Omega0 * diagUtPhiPhitU0) /
                                      (1 + Vhat*spec_Omega0)**2)

        m = mhat/np.sqrt(gamma) * np.mean(diagUtPhiPhitU0/(1 + Vhat*spec_Omega0))
        

    return V, q, m
 


# In[162]:


def update_hat(q0, m, v, alpha, gamma, rho,lamb):
    sigma=rho-m**2/q0
    sq0=np.sqrt(q0)
    
    Im = integrate_for_mhat(m, sq0, v, sigma,lamb=lamb)
    Iv = integrate_for_Vhat(m, sq0, v, sigma,lamb=lamb)
    Iq = integrate_for_Qhat(m, sq0, v, sigma,lamb=lamb)
        
    mhat = alpha / np.sqrt(gamma) * Im
    vhat = -alpha * Iv
    qhat0 = alpha * Iq
    
    return qhat0, mhat, vhat

def update_overlaps(qhat0, mhat, vhat, gamma, lamb):
    IV, IQ, IM = integrate_for_qvm(qhat0,mhat,vhat,lamb)
    v = IV
    m =IM
    q0= IQ
    
    return q0, m, v


# In[163]:


def update_sp(q0, m, v, alpha, gamma, lamb, rho): 
    qhat0, mhat, vhat = update_hat(q0, m, v, alpha, gamma, rho,lamb)     
    q0new, mnew, vnew = update_overlaps(qhat0, mhat, vhat, gamma,lamb)

    return q0new, mnew, vnew, qhat0, mhat, vhat


# In[164]:


def damping(q_new, q_old, coef_damping=0.7):
    return (1 - coef_damping) * q_new + coef_damping * q_old

def iterate_sp(alpha, gamma, lamb, rho, max_iter=int(1000), init=(1, 0.02, 0.01), eps=1e-7, verbose=True):
    # Initialise qu and qv
    q0 = np.zeros(max_iter)

    m = np.zeros(max_iter)
    v = np.zeros(max_iter)
    qhat = np.zeros(max_iter)    
    mhat = np.zeros(max_iter)
    vhat = np.zeros(max_iter)
    
    v[0], q0[0], m[0] = init
    vhat[0], qhat[0], mhat[0] = 0,0,0
        
    for t in range(max_iter - 1):
        q0tmp, mtmp, vtmp, qhat[t+1],mhat[t+1],vhat[t+1] = update_sp(q0[t], m[t], v[t], alpha, gamma, lamb, rho)
        q0[t + 1], m[t+1], v[t+1] = damping(q0tmp, q0[t]), damping(mtmp, m[t]), damping(vtmp, v[t])
        
        if verbose:
            print('t: {}, q0: {}, m: {}, v: {}'.format(t, q0[t+1], m[t+1], v[t+1]))
        
        diff = np.linalg.norm(q0[t + 1] - q0[t])  + np.linalg.norm(m[t + 1] - m[t]) + np.linalg.norm(v[t + 1] - v[t])
                
        if diff < eps:
            break
            
    return q0[:t + 1], m[:t + 1], v[:t+1], qhat[:t + 1], mhat[:t + 1], vhat[:t+1],t



def get_all(alpha, iter):
    init=(1,0.2,0.01)
    # Initialise qu and qv
    q, m, v, qhat, mhat, vhat , t = iterate_sp(alpha, gamma, lamb, rho, verbose=True, max_iter=1000, init=init) 
    
    qf = q[-1]
    mf = m[-1]
    vf = v[-1]
    
    qhatf=qhat[-1]
    mhatf=mhat[-1]
    vhatf=vhat[-1]
    
    test_error=np.arccos(mf / np.sqrt(rho*qf))/np.pi
    #init = (vf, qf, mf)

    return ["hinge",gamma,lamb,rho,alpha,vf,mf,qf,vhatf,mhatf,qhatf,test_error]



pool=mp.Pool(mp.cpu_count())
results=pool.starmap_async(get_all,[(alphas[i],i) for i in range(l)]).get()
pool.close()


keys=['task', 'gamma',
       'lambda', 'rho', 'sample_complexity', 'V', 'm', 'q', 'Vhat', 'mhat',
       'qhat', 'test_error']

replicas=pd.DataFrame(data=results, columns=keys)


replicas["eta"]=replicas["m"].values/np.sqrt(rho*replicas["q"].values)
replicas["r1"]=replicas["mhat"].values/replicas["Vhat"].values
replicas["r2"]=replicas["sample_complexity"].values*replicas["qhat"].values/replicas["Vhat"].values**2
replicas["z"]=replicas["sample_complexity"].values/replicas["Vhat"].values
replicas["alpha"]=alph
replicas["r"]=r
replicas["p"]=p




replicas.to_csv("data/replica_hinge/hinge_p{}_alpha{}_r{}_lamb{}.csv".format(p,alph,r,lamb))






