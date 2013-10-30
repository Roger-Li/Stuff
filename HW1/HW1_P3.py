from __future__ import division
import sys
import math
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import csv
import random as rd


''' define useful functions'''
def logit_inverse (t):
    x = np.exp(t)/(1+np.exp(t))
    return x
    
def log_targ(y,x,beta,beta_0,sigma_0_inv):     # posterior target distribution
    
    n = len(y)
    #m = np.matrix(np.ones(shape = (n,1)))
    s = 0
    for i in range(n):
        s=s+y[i]*(x[i,:]*beta) - 1*np.log(1+np.exp(x[i,:]*beta))
    log_pi=s-(beta-beta_0).T*sigma_0_inv*(beta-beta_0)/2
    return log_pi

def post_credible_interval(data, confidence=0.95):    # compute posterior credible interval
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.tstd(a)
    
    
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
    
def sacf(Y, k=1): #Computes the sample autocorrelation function coeffficient rho for given lag k
    flen = float(len(Y))
    ybar = float(sum([y for y in Y])) / flen
    D = sum([(y-ybar)**2 for y in Y])
    N = sum([ (y-ybar)* (ytpk -ybar) for (y, ytpk) in zip(Y[:-k],Y[k:])])
    return N/D
    
''' data import and pre-processing '''
data = []
with open("breast_cancer.txt", "r") as f:
    for columns in ( raw.strip().split() for raw in f ):
        data.append(columns)

# print len(data)
# print data[0][:]


headers = data[0][:]
reader = data[1:][:]
column = {}
for h in headers:
    column[h] = []

for row in reader:
       for h, v in zip(headers, row):
         column[h].append(v)

# print column['diagnosis']
# print float(column[headers[2]][4])
n = len(column[headers[0]])
p = 11

y = np.zeros(shape = (n,1))
x = np.ones(shape = (n,p))
for i in range(n):
    if column['diagnosis'][i] == '"M"':
        y[i] = 1

for j in range(1,p):
    #print column[headers[j-1]]
    x[:,j] = map(float, column[headers[j-1]])


x = np.matrix(x)    # x[:,0] is the intercept column
y = np.matrix(y)



beta_0 = np.zeros(shape = (p,1))
sigma_0_inv = np.matrix(1E-3 * np.diag(np.ones(p)))



'''Now define the main function'''

def bayes_logreg(y,x,beta_0,sigma_0_inv,niter=10000,burnin=1000, \
                 print_every=1000,retune=100,verbose=True):
    ## tuning process of proposal distribution beta_prop ~ N(beta_curt,V)
    # initial v
    '''    
    Now we import data for initial guess of beta's covariance matrix v.
    
    In order to get a convergence MCMC, we'll have to take a peek of
    the parameter's information by using the covariance matrix of GLM 
    model fit. 
    
    This is done in R and imported as v in python, for code-swap reviewer, 
    please download the V_init.csv file to test the code 
    '''
    p = 11    
    v = np.zeros(shape=(p,p))
    with open('V_init.csv', 'rb') as csvfile:
        v_reader = csv.reader(csvfile, delimiter=';')
        i = 0    
        for row in v_reader:    
            v[i,:] = np.array(row).astype('float')
            i = i+1
    v = np.matrix(v) 
    
    beta_curt = np.zeros(shape = (p,1))
    accept_rate = 0

    # tuning
    while accept_rate < 0.3 or accept_rate > 0.6:
        num_accept = 0
        # beta_curt = np.matrix('0;0')
        for i in range(retune):              
            beta_prop = np.random.multivariate_normal(np.reshape(beta_curt,p), v, 1).T
            log_alpha = log_targ(y,x,beta_prop,beta_0,sigma_0_inv) - log_targ(y,x,beta_curt,beta_0,sigma_0_inv)
            log_u = np.log(np.random.uniform(low=0.0, high=1.0, size=1))
            if log_u < log_alpha:
                beta_curt = beta_prop
                num_accept = num_accept+1         
            else:
                beta_curt = beta_curt
#       print num_accept, log_alpha[0,0], beta_curt
        
        accept_rate = float(num_accept) / retune
        if accept_rate <= 0.3:
            v = (1/np.exp(1))*v
        elif accept_rate >= 0.6:
            v = np.exp(1)*v
#        print accept_rate
#        print v
    
    print "Acceptance rate after tuning:", 100*accept_rate
    print "Covariance matrix after tuning:"
    print v
    
    
    ## MH procedure
    
    beta_curt = np.zeros(shape = (p,1))
    # store beta:    
    beta_store_b = np.matrix(np.zeros(shape = ((burnin+niter),p)))
    # store acceptance rate:
    accept_store_b = np.zeros(shape = ((burnin+niter),1))
    
    
    for i in range(burnin+niter):
        beta_prop = np.random.multivariate_normal(np.reshape(beta_curt,p), v, 1).T
        log_alpha = log_targ(y,x,beta_prop,beta_0,sigma_0_inv) - log_targ(y,x,beta_curt,beta_0,sigma_0_inv) 
        log_u = np.log(np.random.uniform(low=0.0, high=1.0, size=1))
#        print "num of iteration of MH", i
        if log_u < log_alpha:
            beta_curt = beta_prop
            accept_store_b[i] = 1         
        else:
            beta_curt = beta_curt
            
        
        beta_store_b[i,:] = np.transpose(beta_curt)
        
    # get rid of the burning period data
    beta_store = beta_store_b[burnin:(burnin+niter),:]
    accept_store = accept_store_b[burnin:(burnin+niter)]
    #print len(accept_store)
    #print np.sum(accept_store_b[0:burnin])/burnin
    print "Acceptance rate of the MCMC %:", \
          100*round(np.sum(accept_store)/niter, 2)
    
#    beta_0 = beta_store[:,0]
#    beta_1 = beta_store[:,1]
    for i in range(p):
        print "mean_beta_",i, ":", np.mean(beta_store[:,i])

          
    ## extract posterior quantiles
    percentile = np.zeros(shape=(99,p))
    
    pp = []   #list of quantiles to compute
    for i in range(1,100):
        pp.append(float(i))
    
    for i in range(p):
        percentile[:,i] = np.percentile(a = beta_store[:,i], q = pp)
    # np.savetxt("q3_quantile.csv", percentile, delimiter=",")
    
#    for i in range(p):
#        plt.plot(beta_store[:,p])    
#    plt.show()
     
    ''' Calculate posterior credible intervals for beta'''
    for j in range(p):
        me, l,r = post_credible_interval(beta_store[:,j], confidence = 0.95)
        print "The 95% posterior credible interval for beta_",j,"is :", (l, r)
 
    '''lag-1 autocorrelation'''
    for j in range(p):
        rho = sacf(beta_store[:,j], k=1)
        print "The lag-1 autocorrelation for beta_",j,"is :", rho 
    
    ''' Posterior predictive checking'''
    n = 569   # number of data   
    num_new = 300    # size of the new sample 
    y_new = np.zeros(shape = (num_new,n))
    index = rd.sample(xrange(niter), num_new)
    for i in range(num_new):
        beta = beta_store[index[i],:]
        for j in range(n):
            y_new[i,j] = np.random.binomial(1, logit_inverse(x[j,:]*np.transpose(beta)))

    y_mean = np.zeros(shape = (num_new,1))
    for i in range(num_new):
        y_mean[i,:] = np.mean(y_new[i,:])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, rectangles = ax.hist(y_mean, 50, normed=True)   
    fig.canvas.draw()         
    p1 = ax.axvline(x=np.mean(y))          # plot the true mean
    ax.legend([p1], ["True Mean"])
    plt.show()


''' Now let's run the code and see what happens, note that it takes time:'''
bayes_logreg(y,x,beta_0,sigma_0_inv,niter=10000,burnin=3000, \
                 print_every=1000,retune=100,verbose=True) 
                 


 

