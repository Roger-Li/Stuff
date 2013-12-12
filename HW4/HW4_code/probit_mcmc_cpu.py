# -*- coding: utf-8 -*-
"""
STA 250 - HW4
Problem 2: Probit MCMC - CPU

@author: roger
"""

from __future__ import division
from truncnorm import truncNorm
from scipy.stats import truncnorm
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import time



''' probit mcmc function '''
def probit_mcmc_cpu(Y, X, beta_0, Sigma_0_inv, niter, burnin, d):
    
    
    # y:                vector of length n
    # x:                (n x p)design matrix
    # beta_0            (p x 1)prior mean
    # Sigma_0_inv       (p x p)prior precision
    # niter             number of post burnin iterations
    # burnin            number of burnin iterations
    
    print "Now try dataset_",d
    start_time = time.time()
    n = len(X[:,0])      
    p = len(beta_0)
    beta = np.matrix(np.zeros(shape = (p, burnin + niter + 1)))
    beta[:,0] = beta_0
       
    V = np.linalg.pinv(Sigma_0_inv + X.T * X)
    
    Z = np.zeros(shape=(n, 1))
    y0 = np.ones(shape=(n,1))-Y
    y1 = Y
    

    # The following vectors are bounds for truncated normal sampling    
    a = np.zeros(n)
    b = np.zeros(n)
    a_inf = -np.inf*np.ones(n)
    b_inf = np.inf*np.ones(n)  
 
    
    for i in range(1,burnin + niter + 1):
              
        mu = X*beta[:,i-1]
        mu = np.asarray(mu)
        b0 = b-np.squeeze(mu)
        a1 = a-np.squeeze(mu)

        z0 = np.matrix(truncnorm.rvs(a_inf, b0, size=n)).T
        z1 = np.matrix(truncnorm.rvs(a1, b_inf, size=n)).T

        
        Z = np.multiply(y0,z0)+np.multiply(y1, z1)        
        Z = X*beta[:,i-1]+Z
       
        mu_vec = np.asarray(V*(Sigma_0_inv * beta_0 + X.T *Z)).reshape(-1)
        beta[:,i] = np.random.multivariate_normal(mean = mu_vec, cov=V, size=1).T

    # finalize the sample get rid of the ones from the burning period 
    beta_output = np.asarray(beta[:,burnin+1:niter+burnin+1])
    end_time=time.time()    
    
    '''Estimates output'''
    beta_stats = []
    for i in range(p):
        beta_stats.append([np.mean(beta_output[i,:]), np.median(beta_output[i,:])])
    beta_stats.append([000000,end_time-start_time])
    print "data set_",d,"is completed"
    
    np.savetxt("result/res_data_CPU_0"+str(d)+".txt", beta_stats, delimiter=",") 
    
    '''Traceplot and Posterior density plot'''
    for i in range(p):        
        data0 = np.squeeze(np.asarray(beta_output[i,:]))
    # this create the kernel, given an array it will estimate the probability over that values
        kde0 = gaussian_kde( data0 )   
        kde0.covariance_factor = lambda : .25
        kde0._compute_covariance()
   
    # these are the values over wich your kernel will be evaluated\
        dist_space0 = np.linspace( min(data0), max(data0), 100 )
   
    # Now plot    
        f, axarr = plt.subplots(2, sharex=False)
        axarr[0].plot(data0)
        tt_1='CPU: Dataset_'+str(d)+' :Traceplot of beta_'+str(i+1)
        axarr[0].set_title(tt_1)
        axarr[1].plot(dist_space0, kde0(dist_space0) )
        tt_2='CPU: Dataset_'+str(d)+' :Posterior density of beta__'+str(i+1)
        axarr[1].set_title(tt_2)
        filename = 'CPU: Dataset'+str(d)+'_Beta_'+str(i+1)+'.png'
        f.savefig(filename)
        
    #return beta_output
      


'''Main program starts here'''            
for d in range(1,5):
    
    ''' data import and pre-processing '''
    data = []
    with open("data/data_0"+str(d)+".txt", "r") as f:
        for columns in ( raw.strip().split() for raw in f ):
            data.append(columns)
    
    headers = data[0][:]
    reader = data[1:][:]
    column = {}
    for h in headers:
        column[h] = []
    
    for row in reader:
           for h, v in zip(headers, row):
             column[h].append(v)
    
    n = len(column[headers[0]])
    p = 8
    
    y = np.zeros(shape = (n,1))
    x = np.zeros(shape = (n,p))
    
    y = map(float, column[headers[0]])
    for j in range(p):
        #print column[headers[j-1]]    
        x[:,j] = map(float, column[headers[j+1]])
    
    
    x = np.matrix(x)    # x[:,0] is the intercept column
    y = np.matrix(y).T
    
    '''Initialization of other parameters'''
    beta_0 = np.zeros(shape = (p,1))
    sigma_0_inv = np.matrix(np.diag(np.ones(p)))
    
    niter=2000
    burnin=500
    
    '''Gibbs Sampler''' # this function takes care of everything
    beta = probit_mcmc_cpu(y, x, beta_0, sigma_0_inv, niter, burnin, d)
    
 