#!/usr/bin/env python
import sys
from scipy.stats import norm,uniform
import numpy as np
from numpy import pi,sqrt
from sklearn.preprocessing import normalize

# Density of the wrapped normal
def dwrpnorm(x,mu,stdev,wrap=2*pi):
  lim_inf = int(np.floor((mu-5*stdev)/wrap)-1)
  lim_sup = int(np.ceil((mu+5*stdev)/wrap)+1)
  k_vals = [q*wrap+x for q in range(lim_inf,lim_sup+1)]
  return sum([norm.pdf(q,mu,stdev)*(x >= 0 and x <= wrap) for q in k_vals])

# Clustering using a uniform-wrapped Gaussian mixture model and the EM algorithm
def em_wrap_mix(x,mu=3.14,sigma2=1,theta=0.9,eps=0.0001,max_iter=1000,L=20,wrap=2*pi):
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
  while diff>eps and iter_num<max_iter:
    # Save the previous value of the parameters
    mu_old = mu
    sigma2_old = sigma2
    theta_old = theta
    # E-step with normalised rows
    z = normalize(np.insert(theta_old[0]*norm.pdf(np.transpose(np.tile(x,(2*L+1,1)))+2*pi*np.tile(range(-L,L+1),(len(x),1)),\
    	mu_old,sqrt(sigma2_old)),2*L+1,theta_old[1]/float(wrap),axis=1),"l1")
    # M-step
    zt = z.sum(axis=0).tolist()
    z_sum = [sum(zt[:(len(zt)-1)]),zt[len(zt)-1]]
    # Update the mixing proportions
    theta = [zval/sum(z_sum) for zval in z_sum]
    # Update mu
    z1 = z[:,:(z.shape[1]-1)]
    norm_const = float(z1.sum())
    mu = np.multiply(np.transpose(np.tile(x,(2*L+1,1)))+2*pi*np.tile(range(-L,L+1),(len(x),1)),z1).sum()/norm_const
    # Update sigma
    sigma2 = np.multiply((((np.transpose(np.tile(x,(2*L+1,1)))+2*pi*np.tile(range(-L,L+1),(len(x),1)))-mu)**2),z1).sum()/norm_const
    mu %= (2*pi)
    # Compute the maximum difference between two consecutive parameter estimates
    diff = max([np.abs(mu-mu_old),np.abs(sigma2-sigma2_old),np.abs(theta[0]-theta_old[0])])
    # Update the number of iterations
    iter_num += 1
  # Assign the wrapped times to clusters using the highest probability
  auto_prob = z[:,:(z.shape[1]-1)].sum(axis=1)
  human_prob = z[:,z.shape[1]-1]
  cluster = [np.argmax([human_prob[obs],auto_prob[obs]]) for obs in range(len(x))]
  # Obtain the highest probability for the kappas
  kappa = [np.argmax(z[obs,:z.shape[1]-1])-L for obs in range(len(x))]
  # Return parameters and cluster allocations
  return mu,sigma2,cluster,kappa,iter_num==max_iter
