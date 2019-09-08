#!/usr/bin/env python
import sys
from scipy.special import gammaln
from scipy.stats import gamma,geom,norm
from scipy.stats import laplace as lpl
from collections import Counter
import numpy as np
from numpy import pi,sqrt
import cps_circle
import mix_wrapped, mix_wrapped_laplace
import matplotlib.pyplot as plt

#########################################################################################################################
## Collapsed Metropolis-within-Gibbs sampler with Reversible Jump steps for separation of human and automated activity ##
#########################################################################################################################

#### Collapsed Gibbs sampler
def collapsed_gibbs(t, p, fixed_duration=False, n_samp=10000, n_chains=3, L=10, laplace=False, shift=False,\
		mu0=pi, lambda0=1, alpha0=1, beta0=1, gamma0=1, delta0=1, nu=0.1, eta=1, burnin=1000):
	## Set the independence of the priors for the Laplace distribution here
	independent_prior = False
	## Define the matrices containing the parameter values
	mu = np.zeros([n_samp+burnin,n_chains])
	sigma2 = np.zeros([n_samp+burnin,n_chains])
	ell = np.zeros([n_samp+burnin,n_chains],"i")
	tau = {}
	## Matrix of cluster allocations
	N = len(t)
	cluster = np.zeros([n_chains,N])
	## Transformation using the periodicity
	if fixed_duration:
		x = np.diff(np.insert(t,0,t[0])) % p / p * 2*pi
	else: 
		x = np.array([obs % p / p * 2*pi for obs in t])
	if shift:
		x += np.pi
		x %= (2*np.pi)
	## Transformation to daily arrival time
	y = np.array([obs % 86400 / 86400 * 2*pi for obs in t])
	## Vector of evaluations for the human density
	evals = np.linspace(0,2*pi,1000)
	evals_array = np.zeros([n_samp,n_chains,1000])
	## Vector of allocations 
	mean_theta = np.zeros([n_samp,n_chains])
	## Initial configuration
	if laplace:
		mu_start,sigma_start,cluster_init,kappa_init,itmax = mix_wrapped_laplace.em_wrap_mix_laplace(x,mu=pi,beta=0.5,theta=0.9,\
			eps=0.0001,max_iter=500,L=10)
	else:
		mu_start,sigma_start,cluster_init,kappa_init,itmax = mix_wrapped.em_wrap_mix(x,mu=pi,sigma2=0.5,theta=0.9,eps=0.0001,\
			max_iter=500,L=10)
	if itmax:
		sigma_start = 1.0
		probs = lpl.pdf(x,mu_start,sigma_start) if laplace else norm.pdf(x,mu_start,np.sqrt(sigma_start))
		threshold = np.percentile(probs,10)
		cluster_init = np.array(probs > threshold)
		kappa_init = np.zeros(N,"i")
	z_init = [kappa_init[obs] if cluster_init[obs] == 1 else L+1 for obs in range(N)]
  	## Loop over different chains
	for c in range(n_chains):
		## Sample ell from the prior or start from a given starting point
		ell[0,c] = 2
		## Obtain the optimised starting values of the parameters from the EM algorithm
		sigma2[0,c] = sigma_start
		mu[0,c] = mu_start
		if laplace: 
			mu_kappa = 0
		z = np.array(z_init)
		y_daily = np.sort(y[z == (L+1)])
		Nz = len(y_daily)
		## Initial choice of the changepoints -> run MCMC on the subset obtained using the EM
		## Last sample is the initial value of tau
		tau[0,c] = np.sort(np.random.uniform(0,2*pi,2))
		ell[0,c] = 2
		y_positions = np.searchsorted(y_daily,tau[0,c])
		likelihoods = np.zeros(2)
		for _ in range(100000):
			move_types=["insert","shift"]
			if ell[0,c] > 1:
				move_types += ["delete"]
				move_type = np.random.choice(move_types)
			else:
				move_type = np.random.choice(move_types)
			## According to the move type, propose a new configuraton for the changepoints
			if move_type == "insert":
				accept, tau[0,c], ell[0,c], y_positions, likelihoods = cps_circle.propose_insert_changepoint(tau[0,c],\
					y_positions,y_daily,likelihoods,nu,eta)
			elif move_type == "delete":
				accept, tau[0,c], ell[0,c], y_positions, likelihoods = cps_circle.propose_delete_changepoint(tau[0,c],\
					y_positions,y_daily,likelihoods,nu,eta)
			elif move_type == "shift":
				accept, tau[0,c], ell[0,c], y_positions, likelihoods = cps_circle.propose_shift_changepoint(tau[0,c],\
					y_positions,y_daily,likelihoods,nu,eta)
		## Re-compute likelihoods and y_positions
		bin_pos = np.searchsorted(tau[0,c],y)
		bin_pos[bin_pos==len(tau[0,c])] = 0
		## Bin counts
		if Nz == 0:
			Nj = np.repeat(0,ell[0,c])
		else:
			y_positions = np.searchsorted(y_daily,tau[0,c])
			Nj = np.insert(np.diff(y_positions),0,y_positions[0]+Nz-y_positions[ell[0,c]-1])
		likelihoods = cps_circle.tau_likelihood(tau[0,c],y_daily,y_positions,eta)[:]
		likelihood = sum(likelihoods)
		## Allocations
		counts_kappa = Counter(z)
		Na = [counts_kappa[kappa] for kappa in range(-L,L+2)]
		## Indices (continuously updated by randomly shuffling)
		indices = range(N)
		## Loop over number of samples
		for i in range(1,burnin+n_samp):
			## Print status of MCMC
    			if i < burnin:
        			sys.stdout.write("\r+++ Burnin +++ %d / %d " % (i+1,burnin))
        			sys.stdout.flush()
    			elif i == burnin:
        			sys.stdout.write("\n")
    			elif i < burnin + n_samp - 1:
        			sys.stdout.write("\r+++ Samples +++ %d / %d " % (i+1-burnin,n_samp))
        			sys.stdout.flush()
    			else:
        			sys.stdout.write("\r+++ Samples +++ %d / %d\n " % (i+1-burnin,n_samp))
			## Update the latent allocations in randomised order
			np.random.shuffle(indices)
			## Loop for the latent variables
			for k in range(N)[: (N / 20)]:
				j = indices[k]
				zold = z[j]
				## Probability that the event is human for different values of zold and bj
				bj = bin_pos[j]
				# Compute the probabilities for the human according to the position of the event
				if zold == (L+1) and bj != 0:
					prob_human = 1.0/(tau[i-1,c][bj]-tau[i-1,c][bj-1]) * (eta*(tau[i-1,c][bj]-tau[i-1,c][bj-1]) +\
						Nj[bj] - 1) / (2*pi*eta + Nz - 1)
				elif zold != (L+1) and bj != 0:
					prob_human = 1.0/(tau[i-1,c][bj]-tau[i-1,c][bj-1]) * (eta*(tau[i-1,c][bj]-tau[i-1,c][bj-1]) +\
						Nj[bj]) / (2*pi*eta + Nz)
				elif zold == (L+1) and bj == 0:
					prob_human = 1.0/(2*pi-tau[i-1,c][len(tau[i-1,c])-1]+tau[i-1,c][bj]) *\
					(eta*(2*pi-tau[i-1,c][len(tau[i-1,c])-1]+tau[i-1,c][bj]) + Nj[bj] - 1) / (2*pi*eta + Nz - 1)
				elif zold != (L+1) and bj == 0:
					prob_human = 1.0/(2*pi-tau[i-1,c][len(tau[i-1,c])-1]+tau[i-1,c][bj]) *\
					(eta*(2*pi-tau[i-1,c][len(tau[i-1,c])-1]+tau[i-1,c][bj]) + Nj[bj]) / (2*pi*eta + Nz)
				## Probability that the event is automated
				if laplace:
					kappa_probs = lpl.pdf(2*pi*np.arange(-L,L+1)+x[j],mu[i-1,c],sigma2[i-1,c])
				else:
					kappa_probs = norm.pdf(2*pi*np.arange(-L,L+1)+x[j],mu[i-1,c],np.sqrt(sigma2[i-1,c]))
				prob_auto = sum(kappa_probs)
				## Proportions of allocations 
				Nah = [Na[2*(L+1)-1] - (zold == (L+1)),sum(Na[:2*(L+1)-1]) - (zold != (L+1))]
				## Resample the allocations
				probs = [(h1 + h2) * h3 for h1, h2, h3 in zip(Nah,[gamma0,delta0],[prob_human,prob_auto])]
				probs /= sum(probs)
				ha = np.random.choice(range(2),p=probs)
				kappa_probs /= sum(kappa_probs)
				z[j] = L+1 if ha==0 else np.random.choice(np.arange(-L,L+1),1,p=kappa_probs)
				## Update the counts
				zsamp = z[j]
				if zsamp != zold:
					Na[zold+L] -= 1
					Na[zsamp+L] += 1
					if zsamp == (L+1):
						Nj[bj] += 1
						Nz += 1
					if zold == (L+1):
						Nj[bj] -= 1
						Nz -= 1
				## Update taus and ells only for a multiple of 50
				if k % 50 == 0:
					## Re-calculate y_positions
					y_daily = np.sort(y[z == (L+1)])
					y_positions = np.searchsorted(y_daily,tau[i-1,c])
					## Resample the taus and move ell
					move_types=["insert","shift"]
					if ell[i-1,c] > 1:
						move_types += ["delete"]
						move_type = np.random.choice(move_types)
					else:
						move_type = np.random.choice(move_types)
					## According to the move type, propose a new configuraton for the changepoints
					if move_type == "insert":
						accept, tau[i-1,c], ell[i-1,c], y_positions, likelihoods = cps_circle.propose_insert_changepoint(tau[i-1,c],\
							y_positions,y_daily,likelihoods,nu,eta)
					elif move_type == "delete":
						accept, tau[i-1,c], ell[i-1,c], y_positions, likelihoods = cps_circle.propose_delete_changepoint(tau[i-1,c],\
							y_positions,y_daily,likelihoods,nu,eta)
					elif move_type == "shift":
						accept, tau[i-1,c], ell[i-1,c], y_positions, likelihoods = cps_circle.propose_shift_changepoint(tau[i-1,c],\
							y_positions,y_daily,likelihoods,nu,eta)
					## Allocation to the bins
					if Nz == 0:
						Nj = np.repeat(0,ell[i-1,c])
					else:
						Nj = np.insert(np.diff(y_positions),0,y_positions[0]+Nz-y_positions[ell[i-1,c]-1])
					bin_pos = np.searchsorted(tau[i-1,c],y)
					bin_pos[bin_pos==len(tau[i-1,c])] = 0
			## Update the cluster allocations after burnin
			if i > burnin:
				cluster[c,] += np.array(z != (L+1))
			## Update the parameters for the wrapped Gaussian\Laplace part
			x_wrapped = x[z != (L+1)]
			Nw = N - Nz
			if laplace:
				alpha_post = lambda0 + Nw + (1 if not independent_prior else 0)
				beta_post = beta0 + np.sum(abs(x_wrapped+2*pi*z[z!=(L+1)] - mu[i-1,c])) 
				beta_post += (lambda0 * np.abs(mu[i-1,c] - mu0) if independent_prior else 0)
				sigma2[i,c] = 1.0/np.random.gamma(alpha_post,1.0/beta_post)
				mu_prop = np.random.laplace(loc=mu[i-1,c],scale=0.1,size=1)
				num_ratio = - (np.sum(abs(x_wrapped+2*pi*z[z!=(L+1)] - mu_prop))) / sigma2[i,c]
				if not independent_prior:
					num_ratio -= lambda0 * np.abs(mu_prop - mu0) / sigma2[i,c]
				else:
					num_ratio -= np.abs(mu_prop - mu0) / lambda0
				den_ratio = - (np.sum(abs(x_wrapped+2*pi*z[z!=(L+1)] - mu[i-1,c]))) / sigma2[i,c]
				if not independent_prior:
					den_ratio -= lambda0 * np.abs(mu[i-1,c] + 2*pi*mu_kappa - mu0) / sigma2[i,c]
				else:
					den_ratio -= np.abs(mu[i-1,c] + 2*pi*mu_kappa - mu0) / lambda0
				if (- np.random.exponential()) < (num_ratio - den_ratio):
					mu[i,c] = mu_prop
					mu_kappa = int(mu_prop // (2*pi))
				else:
					mu[i,c] = mu[i-1,c]
			else:
				lambda_post = lambda0 + Nw
				xbar = 0.0 if Nw==0 else np.mean(x_wrapped+2*pi*z[z!=(L+1)])
				mu_post = (lambda0*mu0 + Nw*xbar)/lambda_post
				alpha_post = alpha0 + Nw/2.0
				vbar = 0.0 if Nw==0 else 1.0/2*Nw*np.var(x_wrapped+2*pi*z[z!=L+1])
				beta_post = beta0 + vbar + (Nw*lambda0) / float(lambda_post) * (xbar - mu0)**2 / 2
				sigma2[i,c] = 1.0/np.random.gamma(alpha_post,1.0/beta_post)
				mu[i,c] = np.random.normal(mu_post,sqrt(float(sigma2[i,c])/lambda_post)) % (2*pi)
			## Propose a move type for the changepoints and ell
			y_daily = np.sort(y[z == (L+1)])
			## Update the likelihoods
			likelihoods = cps_circle.tau_likelihood(tau[i-1,c],y_daily,y_positions,eta)[:]
			likelihood = sum(likelihoods) 
			y_positions = np.searchsorted(y_daily,tau[i-1,c])
			## Moves
			move_types=["insert","shift"]
			if ell[i-1,c] > 1:
				move_types += ["delete"]
				move_type = np.random.choice(move_types)
			else:
				move_type = np.random.choice(move_types)
			## According to the move type, propose a new configuraton for the changepoints
			if move_type == "insert":
				accept, tau[i,c], ell[i,c], y_positions, likelihoods = cps_circle.propose_insert_changepoint(tau[i-1,c],\
					y_positions,y_daily,likelihoods,nu,eta)
			elif move_type == "delete":
				accept, tau[i,c], ell[i,c], y_positions, likelihoods = cps_circle.propose_delete_changepoint(tau[i-1,c],\
					y_positions,y_daily,likelihoods,nu,eta)
			elif move_type == "shift":
				accept, tau[i,c], ell[i,c], y_positions, likelihoods = cps_circle.propose_shift_changepoint(tau[i-1,c],\
					y_positions,y_daily,likelihoods,nu,eta)
			## Allocation to the bins
			Nz = len(y_daily)
			if Nz == 0:
				Nj = np.repeat(0,ell[i,c])
			else:
				y_positions = np.searchsorted(y_daily,tau[i,c])
				Nj = np.insert(np.diff(y_positions),0,y_positions[0]+Nz-y_positions[ell[i,c]-1])
			bin_pos = np.searchsorted(tau[i,c],y)
			bin_pos[bin_pos==len(tau[i,c])] = 0
			## Evaluate the segment means
			if i > burnin:
				sm = np.zeros(len(tau[i,c]))
				sm[0] = cps_circle.segment_mean(tau[i,c][len(tau[i,c])-1],tau[i,c][0],y_positions[len(tau[i,c])-1],\
					y_positions[0],Nz,eta)
				for h in range(1,len(tau[i,c])):
					sm[h] = cps_circle.segment_mean(tau[i,c][h-1],tau[i,c][h],y_positions[h-1],y_positions[h],Nz,eta)
				bp = np.searchsorted(tau[i,c],evals)
				bp[bp==len(tau[i,c])] = 0
				evals_array[i-burnin-1,c] = sm[bp]
				mean_theta[i-burnin-1,c] = float(N-Nz+gamma0) / float(N+gamma0+delta0)
	return mu[burnin:], sigma2[burnin:], tau, ell[burnin:], cluster/n_samp, evals_array, mean_theta
