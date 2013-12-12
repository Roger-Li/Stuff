
# -*- coding: utf-8 -*-
import itertools, random

import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl
import math

epsilon = 10e-8
max_iter = 1000

class Gaussian:
    def __init__(self, X=np.zeros((0,1)), kappa_0=0, nu_0=1.0001, mu_0=None, 
            Psi_0=None): 
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]

        self._hash_covar = None
        self._inv_covar = None

        if mu_0 == None: # initial mean for the cluster
            self._mu_0 = np.zeros((1, self.n_var))
        else:
            self._mu_0 = mu_0
       

        self._kappa_0 = kappa_0 # mean fraction

        self._nu_0 = nu_0 # degrees of freedom
        if self._nu_0 < self.n_var:
            self._nu_0 = self.n_var

        if Psi_0 == None:
            self._Psi_0 = 5*np.eye(self.n_var) #  this 5  factor should be a prior, ~ dependent on the dataset
        else:
            self._Psi_0 = Psi_0
        assert(self._Psi_0.shape == (self.n_var, self.n_var))

        if X.shape[0] > 0:
            self.fit(X)
        else:
            self.default()


    def default(self):
        self.mean = np.matrix(np.zeros((1, self.n_var))) 
        self.covar =  np.matrix(np.eye(self.n_var))


    def recompute_ss(self):
        """ need to have actualized _X, _sum, and _square_sum """ 
        self.n_points = self._X.shape[0]
        self.n_var = self._X.shape[1]
        if self.n_points <= 0:
            self.default()
            return

        kappa_n = self._kappa_0 + self.n_points
        nu = self._nu_0 + self.n_points 
        mu = np.matrix(self._sum) / self.n_points
        mu_mu_0 = mu - self._mu_0

        C = self._square_sum - self.n_points * (mu.transpose() * mu)
        Psi = (self._Psi_0 + C + self._kappa_0 * self.n_points
             * mu_mu_0.transpose() * mu_mu_0 / (self._kappa_0 + self.n_points))

        self.mean = ((self._kappa_0 * self._mu_0 + self.n_points * mu) 
                    / (self._kappa_0 + self.n_points))
        self.covar = (Psi * (kappa_n + 1)) / (kappa_n * (nu - self.n_var + 1))


    def inv_covar(self):
        """ memoize the inverse of the covariance matrix """
        if self._hash_covar != hash(self.covar):
            self._hash_covar = hash(self.covar)
            self._inv_covar = np.linalg.inv(self.covar)
        return self._inv_covar


    def fit(self, X):
        """ to add several points at once without recomputing """
        self._X = X
        self._sum = X.sum(0)
        self._square_sum = np.matrix(X).transpose() * np.matrix(X)
        self.recompute_ss()

    
    def add_point(self, x):
        """ add a point to this Gaussian cluster """
        if self.n_points <= 0:
            self._X = np.array([x])
            self._sum = self._X.sum(0)
            self._square_sum = np.matrix(self._X).transpose() * np.matrix(self._X)
        else:
            self._X = np.append(self._X, [x], axis=0)
            self._sum += x
            self._square_sum += np.matrix(x).transpose() * np.matrix(x)
        self.recompute_ss()


    def rm_point(self, x):
        """ remove a point from this Gaussian cluster """
        # Find the indice of the point x in self._X, be careful with
        indices = (abs(self._X - x)).argmin(axis=0)
        indices = np.matrix(indices)
        ind = indices[0,0]
        for ii in indices:
            if (ii-ii[0] == np.zeros(len(ii))).all(): # ensure that all coordinates match (finding [1, 1] in [[1, 2], [1, 1]] would otherwise return indice 0)
                ind = ii[0,0]
                break
        tmp = np.matrix(self._X[ind])
        self._sum -= self._X[ind]
        self._X = np.delete(self._X, ind, axis=0)
        self._square_sum -= tmp.transpose() * tmp
        self.recompute_ss()


    def pdf(self, x):
        """ probability density function for a multivariate Gaussian """
        size = len(x)
        #assert(size == self.mean.shape[1])
        #assert((size, size) == self.covar.shape)
        det = np.linalg.det(self.covar)
        #assert(det != 0)
        norm_const = 1.0 / (math.pow((2*np.pi), float(size)/2) 
                * math.pow(det, 1.0/2))
        x_mu = x - self.mean
        inv = self.covar.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.transpose()))
        return norm_const * result



class DPMM:
    def _get_means(self):
        return np.array([g.mean for g in self.params.itervalues()])


    def _get_covars(self):
        return np.array([g.covar for g in self.params.itervalues()])


    def __init__(self, n_components=-1, alpha=1.0):
        self.params = {0: Gaussian()}
        self.n_components = n_components
        self.means_ = self._get_means()
        self.alpha = alpha


    def fit_collapsed_Gibbs(self, X):
        """ according to Neal(2000) and Xu(2009) """
        mean_data = np.matrix(X.mean(axis=0))
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]
        self._X = X
        if self.n_components == -1:
            # initialize with 1 cluster for each datapoint
            self.params = dict([(i, Gaussian(X=np.matrix(X[i]), mu_0=mean_data)) for i in xrange(X.shape[0])])
            self.z = dict([(i,i) for i in range(X.shape[0])])
            self.n_components = X.shape[0]
            previous_means = 2 * self._get_means()
            previous_components = self.n_components
        else:
            # init randomly (or with k-means)
            self.params = dict([(j, Gaussian(X=np.zeros((0, X.shape[1])), mu_0=mean_data)) for j in xrange(self.n_components)])
            self.z = dict([(i, random.randint(0, self.n_components - 1)) 
                      for i in range(X.shape[0])])
            previous_means = 2 * self._get_means()
            previous_components = self.n_components
            for i in xrange(X.shape[0]):
                self.params[self.z[i]].add_point(X[i])

        print "Initialized collapsed Gibbs sampling with %i cluster" % (self.n_components)

        n_iter = 0 
        while (n_iter < max_iter 
                and (previous_components != self.n_components
                or abs((previous_means - self._get_means()).sum()) > epsilon)):
            n_iter += 1
            previous_means = self._get_means()
            previous_components = self.n_components

            for i in xrange(X.shape[0]):
                # remove X[i]'s sufficient statistics from z[i]
                self.params[self.z[i]].rm_point(X[i])
                # if it empties the cluster, remove it and decrease K
                if self.params[self.z[i]].n_points <= 0:
                    self.params.pop(self.z[i])
                    self.n_components -= 1

                tmp = []
                for k, param in self.params.iteritems():
                    # compute P_k(X[i]) = P(X[i] | X[-i] = k)
                    marginal_likelihood_Xi = param.pdf(X[i])
                    # set N_{k,-i} = dim({X[-i] = k})
                    # compute P(z[i] = k | z[-i], Data) = N_{k,-i}/(α+N-1)
                    mixing_Xi = param.n_points / (self.alpha + self.n_points - 1)
                    tmp.append(marginal_likelihood_Xi * mixing_Xi)
                    
                # compute P*(X[i]) = P(X[i]|λ)
                base_distrib = Gaussian(X=np.zeros((0, X.shape[1])))
                prior_predictive = base_distrib.pdf(X[i])
                
                # compute P(z[i] = * | z[-i], Data) = α/(α+N-1)
                prob_new_cluster = self.alpha / (self.alpha + self.n_points - 1)
                tmp.append(prior_predictive * prob_new_cluster)

                # normalize P(z[i]) (tmp above)
                s = sum(tmp)
                tmp = map(lambda e: e/s, tmp)

                # sample z[i] ~ P(z[i])
                rdm = np.random.rand()
                total = tmp[0]
                k = 0
                while (rdm > total):
                    k += 1
                    total += tmp[k]
                # add X[i]'s sufficient statistics to cluster z[i]
                new_key = max(self.params.keys()) + 1
                if k == self.n_components: # create a new cluster
                    self.z[i] = new_key
                    self.n_components += 1
                    self.params[new_key] = Gaussian(X=np.matrix(X[i]))
                else:
                    self.z[i] = self.params.keys()[k]
                    self.params[self.params.keys()[k]].add_point(X[i])
         

            print "still sampling, %i clusters currently" % (self.n_components)

        self.means_ = self._get_means()


    def predict(self, X):
        """ produces and returns the clustering of the X data """
        if (X != self._X).any():
            self.fit_collapsed_Gibbs(X)
        mapper = list(set(self.z.values())) # to map our clusters id to
        # incremental natural numbers starting at 0
        Y = np.array([mapper.index(self.z[i]) for i in range(X.shape[0])])
        return Y



# Number of samples per component
n_samples = 50

# Generate random sample, two components
np.random.seed(0)

# 2, 2-dimensional Gaussians
C = np.array([[0., -0.1], [2, .5]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]



from sklearn import mixture

# Fit a mixture of gaussians with EM using five components
gmm = mixture.GMM(n_components=5, covariance_type='full')
gmm.fit(X)


dpmm = DPMM(n_components=5) # 
# n_components is the number of initial clusters 
dpmm.fit_collapsed_Gibbs(X)

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

X_repr = X
if X.shape[1] > 2:
    from sklearn import manifold
    X_repr = manifold.Isomap(n_samples/10, n_components=2).fit_transform(X)

for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpmm, 'Dirichlet Process GMM (Collapsed Gibbs)')]):
    splot = pl.subplot(2, 1, 1 + i)
    Y_ = clf.predict(X)
    print Y_
    for j, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
       
        if not np.any(Y_ == j):
            continue

        pl.scatter(X_repr[Y_ == j, 0], X_repr[Y_ == j, 1], .8, color=color)

        if clf.means_.shape[len(clf.means_.shape) - 1] == 2: 
            # Plot an ellipse to show the Gaussian component
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            if i == 1:
                mean = mean[0] 
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color='k')
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

    pl.xlim(-10, 10)
    pl.ylim(-3, 6)
    pl.xticks(())
    pl.yticks(())
    pl.title(title)


pl.savefig('dpmm.png')