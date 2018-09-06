#!/usr/bin/env python
import sys
from scipy.special import gammaln
from scipy.stats import gamma,geom,norm
from collections import Counter
import numpy as np
from numpy import pi,sqrt
import cps_circle
import mix_wrapped
import matplotlib.pyplot as plt

#########################################################################################################################
## Collapsed Metropolis-within-Gibbs sampler with Reversible Jump steps for separation of human and automated activity ##
#########################################################################################################################

#### Collapsed Gibbs sampler
def collapsed_gibbs(t,p,n_samp=10000,n_chains=3,L=5,mu0=pi,tau0=1,alpha=1,beta=1,nu=0.1,eta=1,burnin=1000):
	## Define the matrices containing the parameter values
	mu = np.zeros([n_samp+burnin,n_chains])
	sigma2 = np.zeros([n_samp+burnin,n_chains])
	ell = np.zeros([n_samp+burnin,n_chains],"i")
	tau = {}
	## Matrix of cluster allocations
	N = len(t)
	cluster = np.zeros([n_chains,N])
	## Transformations
	x = np.array([obs % p / p * 2*pi for obs in t])
	y = np.array([obs % 86400 / 86400 * 2*pi for obs in t])
	## Vector of evaluations for the human density
	evals = np.linspace(0,2*pi,1000)
	evals_array = np.zeros([n_samp,n_chains,1000])
	## Vector of allocations 
	mean_theta = np.zeros([n_samp,n_chains])
	## Initial configuration
	mu_start,sigma_start,cluster_init,kappa_init, itmax = mix_wrapped.em_wrap_mix(x,mu=pi,sigma2=0.5,theta=0.9,eps=0.0001,\
		max_iter=500,L=L)
	if itmax:
		sigma_start = 2
		probs = norm.pdf(x,mu_start,np.sqrt(sigma_start))
		threshold = np.percentile(probs,10)
		cluster_init = np.array(probs > threshold)
		kappa_init = np.zeros(N,"i")
	z_init = [kappa_init[obs] if cluster_init[obs] == 1 else L+1 for obs in range(N)]
  	## Loop over different chains
  	for c in range(n_chains):
		## Sample ell from the prior
		ell[0,c] = 2
		## Obtain the optimised starting values of the parameters from the EM algorithm
		sigma2[0,c] = sigma_start
		mu[0,c] = mu_start
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
			print i
			## Update the latent allocations in randomised order
			np.random.shuffle(indices)
			## Loop for the latent variables
			for k in range(N):
				j = indices[k]
				zold = z[j]
				## Probability that the event is human for different values of zold and bj
				bj = bin_pos[j]
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
				kappa_probs = norm.pdf(2*pi*np.arange(-L,L+1)+x[j],mu[i-1,c],np.sqrt(sigma2[i-1,c]))
				prob_auto = sum(kappa_probs)
				## Proportions of allocations 
				Nah = [Na[2*(L+1)-1],sum(Na[:2*(L+1)-1])]
				## Resample the allocations
				probs = [(h1 + h2) * h3 for h1, h2, h3 in zip(Nah,[alpha,beta],[prob_human,prob_auto])]
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
			if i>burnin:
				cluster[c,] += np.array(z != (L+1))
			## Update the parameters for the wrapped normal part
			x_wrapped = x[z != (L+1)]
			Nw = N - Nz
			tau_post = tau0 + Nw
			xbar = 0.0 if Nw==0 else np.mean(x_wrapped+2*pi*z[z!=(L+1)])
			mu_post = (tau0*mu0 + Nw*xbar)/tau_post
			alpha_post = alpha + Nw/2.0
			vbar = 0.0 if Nw==0 else 1.0/2*Nw*np.var(x_wrapped+2*pi*z[z!=L+1])
			beta_post = beta + vbar + (Nw*tau0) / float(tau_post) * (xbar - mu0)**2 / 2
			sigma2[i,c] = 1.0/np.random.gamma(alpha_post,1.0/beta_post)
			mu[i,c] = np.random.normal(mu_post,sqrt(float(sigma2[i,c])/tau_post)) % (2*pi)
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
				mean_theta[i-burnin-1,c] = float(N-Nz+alpha) / float(N+alpha+beta)
	return mu, sigma2, tau, ell, cluster/n_samp, evals_array, mean_theta

### Example runs ('outlook' and 'candy') are the imported datasets
t = collapsed_gibbs(outlook,p=8.00095,n_samp=5000,n_chains=1,L=5,mu0=np.pi,tau0=1,alpha=1,beta=1,nu=0.1,eta=1,burnin=500)
t = collapsed_gibbs(candy,p=55.66,n_samp=2500,n_chains=1,L=5,mu0=np.pi,tau0=1,alpha=1,beta=1,nu=0.1,eta=1,burnin=500)

### Example plots
plt.plot(np.transpose(np.apply_along_axis(np.percentile,0,t[5],[1,5,10,90,95,99])[:,0,:]))
plt.plot(np.apply_along_axis(np.mean,0,t[5])[0])
plt.show()
