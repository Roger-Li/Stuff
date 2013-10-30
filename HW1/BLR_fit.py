
##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

from __future__ import division
from scipy import stats
import sys
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start = 1000
length_datasets = 200
#######################

# Note: this only sets the random seed for numpy, so if you intend
# on using other modules to generate random numbers, then be sure
# to set the appropriate RNG here

if (nargs <= 1):
    sim_num = sim_start + 1
    np.random.seed(1330931)
else:
    # Decide on the job number, usually start at 1000:
    sim_num = sim_start + int(sys.argv[2])
    # Set a different random seed for every job number!!!
    np.random.seed(762*sim_num + 1330931)

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################



with open("data/blr_data_" + str(sim_num) + '.csv', "rf") as f:
    reader = csv.reader(f)
    headers = reader.next()

    column = {}
    for h in headers:
        column[h] = []
    
    for row in reader:
       for h, v in zip(headers, row):
         column[h].append(v)

length = len(column['y'])
y = np.zeros(shape = (length,1))
m = np.zeros(shape = (length,1))
x = np.zeros(shape = (length,2))
for i in range(length):
    y[i] = float(column['y'][i])
    m[i] = int(float(column['n'][i]))
    x[i,:] = float(column['X1'][i]), float(column['X2'][i])

beta_0 = np.zeros(shape = (2,1))
sigma_0_inv = np.diag(np.ones(2))

# convert everything to matrix
y = np.matrix(y)
m = np.matrix(m)
x = np.matrix(x)
# beta_0 = np.matrix(beta_0)
sigma_0_inv = np.matrix(sigma_0_inv)



def log_targ(m,y,x,beta,beta_0,sigma_0_inv):
    n=len(m)
    s=0
    for i in range(n):
        s=s+y[i]*(x[i,:]*beta)-m[i]*np.log(1+np.exp(x[i,:]*beta))
    log_pi=s-(beta-beta_0).T*sigma_0_inv*(beta-beta_0)/2

    return log_pi

def bayes_logreg(m,y,x,beta_0,sigma_0_inv,niter=10000,burnin=1000, \
                 print_every=1000,retune=100,verbose=True):
    ## tuning process of proposal distribution beta_prop ~ N(beta_curt,V)
    # initial v
    v = np.matrix(np.diag(np.ones(2)))  
    beta_curt = np.zeros(shape = (2,1))
    accept_rate = 0

    # tuning
    while accept_rate < 0.3 or accept_rate > 0.6:
        num_accept = 0
        # beta_curt = np.matrix('0;0')
        for i in range(retune):              
            beta_prop = np.random.multivariate_normal(np.reshape(beta_curt,2), v, 1).T
            log_alpha = log_targ(m,y,x,beta_prop,beta_0,sigma_0_inv) - log_targ(m,y,x,beta_curt,beta_0,sigma_0_inv)
            log_u = np.log(np.random.uniform(low=0.0, high=1.0, size=1))
            if log_u < log_alpha:
                beta_curt = beta_prop
                num_accept = num_accept+1         
            else:
                beta_curt = beta_curt
#        print num_accept, log_alpha[0,0], beta_curt
        
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
    
    beta_curt = np.zeros(shape = (2,1))
    # store beta:    
    beta_store_b = np.matrix(np.zeros(shape = ((burnin+niter),2)))
    # store acceptance rate:
    accept_store_b = np.zeros(shape = ((burnin+niter),1))
    
    
    for i in range(burnin+niter):
        beta_prop = np.random.multivariate_normal(np.reshape(beta_curt,2), v, 1).T
        log_alpha = log_targ(m,y,x,beta_prop,beta_0,sigma_0_inv) - log_targ(m,y,x,beta_curt,beta_0,sigma_0_inv) 
        log_u = np.log(np.random.uniform(low=0.0, high=1.0, size=1))
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
    print "Acceptance rate %:", \
          100*round(np.sum(accept_store)/niter, 2)
    
    beta_0 = beta_store[:,0]
    beta_1 = beta_store[:,1]
    print "mean_beta_0", np.mean(beta_0)
    print "mean_beta_1", np.mean(beta_1)
          
    ## extract posterior quantiles
    percentile = np.zeros(shape=(99,2))
    
    pp = []   #list of quantiles to compute
    for i in range(1,100):
        pp.append(float(i))
    
    percentile[:,0] = np.percentile(a = beta_0, q = pp)
    percentile[:,1] = np.percentile(a = beta_1, q = pp)
    np.savetxt("results/blr_res_" + str(sim_num) + '.csv', percentile, delimiter=",")
    
#    plt.plot(beta_store[:,0])
#    plt.plot(beta_store[:,1])
#    plt.show()
    

    
        



bayes_logreg(m,y,x,beta_0,sigma_0_inv,niter=10000,burnin=1000, \
                 print_every=1000,retune=100,verbose=True)       