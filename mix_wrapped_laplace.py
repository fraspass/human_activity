#!/usr/bin/env python
import sys
from scipy.stats import laplace,uniform
import numpy as np
from numpy import pi,sqrt
from sklearn.preprocessing import normalize

###########################################
## Uniform-wrapped Laplace mixture model ##
###########################################

# Obtain the weighted median
def weighted_median(x,w,sort=False):
  if len(x) != len(w):
    raise ValueError('The number of weights must be the same as the number of observations')
  q = np.argsort(x) if sort else np.arange(len(x))
  r = np.searchsorted(np.cumsum(w[q] if sort else w) / float(np.sum(w)),0.5)
  out = x[q][r] if sort else x[r]
  return out 

# Density of the wrapped normal
def dwrplap(x,mu,beta,wrap=2*pi):
  lim_inf = int(np.floor((mu-7*beta)/wrap)-1)
  lim_sup = int(np.ceil((mu+7*beta)/wrap)+1)
  k_vals = [q*wrap+x for q in range(lim_inf,lim_sup+1)]
  return sum([laplace.pdf(q,mu,beta)*(x >= 0 and x <= wrap) for q in k_vals])

# Clustering using a uniform-wrapped Laplace mixture model and the EM algorithm
def em_wrap_mix_laplace(x,mu=pi,beta=1.0,theta=0.9,eps=0.0001,max_iter=1000,L=20,wrap=2*pi,conv_lik=False,verbose=False):
  # Sort the vector of obserbations (sort only once in the weighted median function)
  qq = x.argsort()
  x = np.sort(x)
  # Theta
  if theta > 0 and theta < 1:
  	theta = [theta,1-theta]
  else:
  	theta = [0.9,0.1]
  # Mu 
  if mu < 0 or mu >= 2*pi:
  	mu %= (2*pi)
  # Initialize the difference between the parameter estimates
  diff = eps+1
  # Initialize number of iterations
  iter_num = 1
  while diff > eps and iter_num < max_iter:
    # Save the previous value of the parameters
    mu_old = mu
    beta_old = beta
    theta_old = theta
    # E-step with normalised rows
    z = normalize(np.insert(theta_old[0]*laplace.pdf(np.transpose(np.tile(x,(2*L+1,1)))+2*pi*np.tile(range(-L,L+1),(len(x),1)),\
    	mu_old,beta_old),2*L+1,theta_old[1]/float(wrap),axis=1),"l1")
    # M-step
    zt = z.sum(axis=0).tolist()
    z_sum = [sum(zt[:(len(zt)-1)]),zt[len(zt)-1]]
    # Update the mixing proportions
    theta = [zval/sum(z_sum) for zval in z_sum]
    # Update mu
    z1 = z[:,:(z.shape[1]-1)]
    norm_const = float(z1.sum())
    ## Important: assumes that the vectors were sorted
    mu = weighted_median((np.transpose(np.tile(x,(2*L+1,1)))+2*pi*np.tile(range(-L,L+1),(len(x),1))).flatten('F'),z1.flatten('F'),sort=False)
    # Update beta
    beta = np.multiply(abs(((np.transpose(np.tile(x,(2*L+1,1)))+2*pi*np.tile(range(-L,L+1),(len(x),1)))-mu)),z1).sum() / norm_const
    mu %= (2*pi)
    # Compute the convergence criterion
    if conv_lik:
      loglik = np.sum(np.log([(theta[0] * dwrplap(xi, mu, beta) + theta[1] / (2*np.pi)) for xi in x]))
      diff = 1.0 if iter_num == 1 else (loglik - loglik_old)
      loglik_old = loglik
    else:
      diff = max([np.abs((mu-mu_old) / mu_old),np.abs((beta-beta_old) / beta_old),np.abs((theta[0]-theta_old[0])) / theta_old[0]])
      ## diff = max([np.abs(mu-mu_old),np.abs(beta-beta_old),np.abs(theta[0]-theta_old[0])])
    # Update the number of iterations
    iter_num += 1
    # Print iterations
    if verbose:
      print 'Iteration ' + str(iter_num) + '\t' + 'mu: ' + str(mu) + '\t' + 'beta: ' + str(beta) + '\t' + 'theta: ' + str(theta[0])
  # Assign the wrapped times to clusters using the highest probability
  auto_prob = z[:,:(z.shape[1]-1)].sum(axis=1)
  human_prob = z[:,z.shape[1]-1]
  cluster = [np.argmax([human_prob[obs],auto_prob[obs]]) for obs in range(len(x))]
  # Obtain the highest probability for the kappas
  kappa = [np.argmax(z[obs,:z.shape[1]-1])-L for obs in range(len(x))]
  # Return parameters and cluster allocations
  return mu,beta,np.array(cluster)[qq],np.array(kappa)[qq],iter_num==max_iter