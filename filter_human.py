#!/usr/bin/env python
import sys, os
import argparse
import numpy as np
from mcmc_sampler import collapsed_gibbs
from fourier import detect_periodicity
import time

###############
## Main file ##
###############

# PARSER to give parameter values 
parser = argparse.ArgumentParser()

### Parameters

# - Parameters for the MCMC
parser.add_argument("-N","--nsamp", type=int, dest="nsamp", default=20000, help="Integer: number of samples after burnin for each chain, default 20000.")
parser.add_argument("-B","--nburn", type=int, dest="nburn", default=5000, help="Integer: number of samples after burnin for each chain, default 5000.")
parser.add_argument("-C","--nchain", type=int, dest="nchain", default=1, help="Integer: number of chains, default 1.")

# - Add period (if known)
parser.add_argument("-p","--period", type=float, dest="period", default=0.0, help="Float: periodicity (if known), default: calculated via Fourier test.")

# - Truncate the periodicity
truncate_periodicity = False
parser.add_argument("-r", "--trunc", action = 'store_true', dest = "truncate_periodicity", default = truncate_periodicity,
    help = 'Round the periodicity to 2 decimal digits, default False.')

# - Use the fixed duration model
fixed_duration = False
parser.add_argument("-o", "--fixed_duration", action = 'store_true', dest = "fixed_duration", default = fixed_duration,
    help = 'Add if the fixed duration model should be used, default False.')

# - Analysis using the wrapped Laplace / Gaussian distribution
laplace = False
parser.add_argument("-l", "--laplace", action = 'store_true', dest = "laplace", default = laplace,
    help = 'Add if the wrapped Laplace model should be used, default False.')

# - Analysis using the wrapped Laplace / Gaussian distribution
shift = False
parser.add_argument("-s", "--shift", action = 'store_true', dest = "shift", default = shift,
    help = 'Add if the wrapped events should be shifted by pi, default False.')

# - NIG prior
parser.add_argument("-m","--mu", type=float, dest="mu", default=np.pi, help="Float: first parameter of the NIG prior, default pi.")
parser.add_argument("-t","--tau", type=float, dest="tau", default=1.0, help="Float: second parameter of the NIG prior, default 1.0.")
parser.add_argument("-a","--alpha", type=float, dest="alpha", default=1.0, help="Float: third parameter of the NIG prior, default 1.0.")
parser.add_argument("-b","--beta", type=float, dest="beta", default=1.0, help="Float: fourth parameter of the NIG prior, default 1.0.")

# - Beta prior
parser.add_argument("-g","--gamma", type=float, dest="gamma", default=1.0, help="Float: first parameter of the Beta prior on theta, default 1.0.")
parser.add_argument("-d","--delta", type=float, dest="delta", default=1.0, help="Float: second parameter of the Beta prior on theta, default 1.0.")

# - Priors on the step function
parser.add_argument("-e","--eta", type=float, dest="eta", default=1.0, help="Float: concentration parameter of the Dirichlet prior, default 1.0.")
parser.add_argument("-v","--nu", type=float, dest="nu", default=0.1, help="Float: parameter of the Geometric prior, default 0.1.")

# - Maximum number of kappas to sample
parser.add_argument("-k","--lmax", type=int, dest="lmax", default=5, help="Integer: maximum (absolute) value for kappa, default 5")

# Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="Results", const=True, nargs="?",\
    help="String: name of the destination folder for the output files.")

## Parse arguments
args = parser.parse_args()
nburn = args.nburn
nsamp = args.nsamp
nchain = args.nchain
mu = args.mu
tau = args.tau
alpha = args.alpha
beta = args.beta
gamma = args.gamma
delta = args.delta
eta = args.eta
nu = args.nu
per = args.period
lmax = args.lmax
dest_folder = args.dest_folder
if args.fixed_duration:
    fixed_duration = True
if args.laplace:
    laplace = True
if args.truncate_periodicity:
    truncate_periodicity = True
if args.shift:
    shift = True

# Create output directory if doesn't exist
if dest_folder != '' and not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

## Import dataset
t = np.loadtxt(sys.stdin)

## Obtain periodicity
p = detect_periodicity(t)['p'] if per == 0.0 else per
if truncate_periodicity:
    p = round(p,2)

## Run MCMC
start = time.time()
mcmc_out = collapsed_gibbs(t, p=p, fixed_duration=fixed_duration, n_samp=nsamp, n_chains=nchain, L=lmax, laplace=laplace, shift=shift,\
    mu0=np.pi, lambda0=tau, alpha0=alpha, beta0=beta, gamma0=gamma, delta0=delta, nu=nu, eta=eta, burnin=nburn)
end = time.time()
print "Time elapsed:" + '\t' + str(end - start)

### Save output
save_x = True
if save_x:
    np.savetxt(dest_folder+'/x.txt' if dest_folder != '' else 'x.txt',t,fmt='%f',delimiter=',')
np.savetxt(dest_folder+'/mu.txt' if dest_folder != '' else 'mu.txt',mcmc_out[0],fmt='%f',delimiter=',')
np.savetxt(dest_folder+'/sigma.txt' if dest_folder != '' else 'sigma.txt',mcmc_out[1],fmt='%f',delimiter=',')
np.savetxt(dest_folder+'/ell.txt' if dest_folder != '' else 'ell.txt',mcmc_out[3],fmt='%f',delimiter=',')
np.savetxt(dest_folder+'/z.txt' if dest_folder != '' else 'z.txt',mcmc_out[4].T,fmt='%f',delimiter=',')
np.savetxt(dest_folder+'/theta.txt' if dest_folder != '' else 'theta.txt',mcmc_out[6][:-1,:],fmt='%f',delimiter=',')

## Output: quantiles of the estimated density of human events
density_out = mcmc_out[5].reshape((mcmc_out[5].shape[0] * mcmc_out[5].shape[1],mcmc_out[5].shape[2]))
density_quantiles = np.hstack((np.transpose(np.apply_along_axis(np.percentile,0,density_out,[1,5,10,90,95,99])),np.mean(density_out,axis=0).reshape(density_out.shape[1],1)))
np.savetxt(dest_folder+'/density_quantiles.txt' if dest_folder != '' else 'density_quantiles.txt',density_quantiles,fmt='%f',delimiter=',')

## Output: filtered events using 0.5 threshold
np.savetxt(dest_folder+'/auto_events.txt' if dest_folder != '' else 'auto_events.txt',t[np.mean(mcmc_out[4],axis=0) > 0.5] ,fmt='%f',delimiter=',')
np.savetxt(dest_folder+'/human_events.txt' if dest_folder != '' else 'human_events.txt',t[np.mean(mcmc_out[4],axis=0) <= 0.5] ,fmt='%f',delimiter=',')
