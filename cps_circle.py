#!/usr/bin/env python
import sys
from scipy.special import gammaln
from scipy.stats import gamma
from collections import Counter
import numpy as np
from numpy import pi

##########################
## Functions for RJMCMC ##
##########################

## Utility function used in the proposals
def get_left_right_cps_positions(tau,tau_position,y_positions,adding=False):
    # If the proposed tau is smaller than all the others, then set both to 0
    if tau_position == 0:
        left_tau_location = tau[len(tau)-1]
        left_y_position = y_positions[len(tau)-1]
    else:
        # Else set to the left tau and left x location
        left_tau_location = tau[tau_position-1]
        left_y_position = y_positions[tau_position-1]
    # If the proposed tau is larger than all the others, then set to 1 and n
    if tau_position == len(tau)-(0 if adding else 1):
        right_tau_location = tau[0]
        right_y_position = y_positions[0]
    else:
        # Else set to the right tau and right x location
        right_tau_location = tau[tau_position+(0 if adding else 1)]
        right_y_position = y_positions[tau_position+(0 if adding else 1)]
    return left_tau_location, right_tau_location, left_y_position, right_y_position

## Utility function used in the proposals
def segment_likelihood(tau1,tau2,y1,y2,n,eta):
    # Same left and right position is not admitted
    if y1==y2:
        return 0
    # Calculate the likelihood for the segment
    if tau1 < tau2:
        return gammaln((y2-y1)+eta*(tau2-tau1)) - gammaln(eta*(tau2-tau1)) - (y2-y1)*np.log(tau2-tau1)
    else:
        return gammaln((n-y1+y2)+eta*(2*pi-tau1+tau2)) - gammaln(eta*(2*pi-tau1+tau2)) - (n-y1+y2)*np.log(2*pi-tau1+tau2)

## Utility function used in the proposals
def tau_likelihood(tau,y,y_positions,eta,get_means=False):
    # Number of changepoints
    ell = len(tau)
    lhds = np.array([0]*ell,dtype=float)
    if get_means:
        probs = np.array([0]*ell,dtype=float)
        densities = np.array([0]*ell,dtype=float)
    # Calculate probabilities and densities
    for i in range(ell):
        y1 = y_positions[ell-1] if i == 0 else y_positions[i-1]
        y2 = y_positions[i]
        tau1 = tau[ell-1] if i == 0 else tau[i-1]
        tau2 = tau[i]
        lhds[i] = segment_likelihood(tau1,tau2,y1,y2,len(y),eta)
        if get_means:
            probs[i],densities[i] = segment_mean(tau1,tau2,y1,y2,len(y),eta)
    if get_means:
        return lhds,probs,densities
    return lhds

## Utility function used in the proposals
def segment_mean(tau1,tau2,y1,y2,n,eta):
    # Calculate the posterior mean of the segment
    if tau2 > tau1:
        segment_prob = (eta*(tau2-tau1)+y2-y1)/(2*pi*eta+n)
    else: 
        segment_prob = (eta*(2*pi-tau1+tau2)+n-y1+y2)/(2*pi*eta+n)
    # Return segment height
    segment_height = segment_prob/(tau2-tau1) if tau2 > tau1 else segment_prob/(2*pi-tau1+tau2)
    return segment_height

## Proposal 1: shift one of the changepoints
def propose_shift_changepoint(tau,y_positions,y,likelihoods,nu,eta):
    # Choose the shifted changepoint at random
    tau_position = np.random.choice(len(tau))
    # Calculate left and right changepoint positions
    left_tau_location,right_tau_location,left_y_position,right_y_position = get_left_right_cps_positions(tau,\
        tau_position,y_positions)
    # Propose the new location for the shifted changepoint
    if left_tau_location < right_tau_location:
        new_location = np.random.uniform(left_tau_location,right_tau_location)
    else: # If the first or last changepoint are shifted (allow to move to the other side of the circle)
        new_location = np.random.uniform(left_tau_location,2*pi+right_tau_location) % (2*pi)
    # Calculate the position tau_star
    tau_position_star = tau_position
    if tau_position in [0,len(tau)-1] and new_location < tau[0]:
        tau_position_star = 0
    elif tau_position in [0,len(tau)-1] and new_location > tau[len(tau)-1]: 
        tau_position_star = len(tau)
    # Calculate the new position
    new_y_position = np.searchsorted(y,new_location)
    # Update the likelihoods of the segment
    new_left_lhd = segment_likelihood(left_tau_location,new_location,left_y_position,new_y_position,len(y),eta)
    new_right_lhd = segment_likelihood(new_location,right_tau_location,new_y_position,right_y_position,len(y),eta)
    # Calculate the likelihood ratio
    lhd_ratio = new_left_lhd + new_right_lhd - likelihoods[tau_position] - (likelihoods[0] \
        if tau_position == (len(tau)-1) else likelihoods[tau_position+1])
    # Calculate the accept/reject
    accept = (lhd_ratio >= 0) or (np.random.exponential() > -lhd_ratio)
    # Updates    
    if accept and tau_position_star == tau_position:
        tau[tau_position] = new_location
        y_positions[tau_position] = new_y_position
        likelihoods[tau_position] = new_left_lhd
        if tau_position != (len(tau)-1): 
            likelihoods[tau_position+1] = new_right_lhd
        else: 
            likelihoods[0] = new_right_lhd
    if accept and tau_position_star != tau_position:
        ell = len(tau)
        if tau_position_star == 0:
            tau = np.insert(tau,tau_position_star,new_location)[:ell]
            y_positions = np.insert(y_positions,tau_position_star,new_y_position)[:ell]
            likelihoods = np.insert(likelihoods,tau_position_star,new_left_lhd)[:ell]
            likelihoods[1] = new_right_lhd  
        elif tau_position_star == ell:
            tau = np.insert(tau,tau_position_star,new_location)[1:]
            y_positions = np.insert(y_positions,tau_position_star,new_y_position)[1:]
            likelihoods = np.insert(likelihoods,tau_position_star,new_left_lhd)[1:]
            likelihoods[0] = new_right_lhd
    return accept, tau, len(tau), y_positions, likelihoods

## Proposal 2: insert changepoint
def propose_insert_changepoint(tau,y_positions,y,likelihoods,nu,eta):
    # Propose a changepoint uniformly
    location = np.random.uniform(0,2*pi)
    # tau_position is the position of the new location in the vector of taus
    tau_position = np.searchsorted(tau,location)
    # new_y_position is the position of the new location in the vector of observations
    new_y_position = np.searchsorted(y,location)
    # Obtain left and right tau location and y position when a new tau is added
    left_tau_location,right_tau_location,left_y_position,right_y_position = get_left_right_cps_positions(tau,\
        tau_position,y_positions,True)
    # Obtain the left and right likelihood
    new_left_lhd = segment_likelihood(left_tau_location,location,left_y_position,new_y_position,len(y),eta)
    new_right_lhd = segment_likelihood(location,right_tau_location,new_y_position,right_y_position,len(y),eta)
    # Compute the likelihood ratio
    lhd_ratio = new_left_lhd + new_right_lhd - (likelihoods[0] if tau_position==len(tau) else \
        likelihoods[tau_position]) + np.log(1-nu) - np.log(len(tau)+1)
    # If there are no changepoints, then the likelihood ratio has a particular form
    # if len(tau) == 2:
    if len(tau) == 1:
        lhd_ratio += np.log(2.0/3)
    # Accept if the likelihood ratio is larger than 0 or when the condition runif < alpha is verified
    accept = (lhd_ratio>=0) or (np.random.exponential() > -lhd_ratio)    
    # Updates
    if accept:
        tau = np.insert(tau,tau_position,location)
        y_positions = np.insert(y_positions,tau_position,new_y_position)
        if tau_position != (len(tau)-1):
            likelihoods[tau_position] = new_left_lhd
            likelihoods = np.insert(likelihoods,tau_position+1,new_right_lhd)
        else:
            likelihoods = np.insert(likelihoods,tau_position,new_left_lhd) 
            likelihoods[0] = new_right_lhd
    return accept, tau, len(tau), y_positions, likelihoods

## Proposal 3: delete changepoint
def propose_delete_changepoint(tau,y_positions,y,likelihoods,nu,eta):
    # Propose at random from the taus
    tau_position = np.random.choice(len(tau))
    # Obtain left and right tau location and x position when a new tau is removed
    left_tau_location,right_tau_location,left_y_position,right_y_position = get_left_right_cps_positions(tau,\
        tau_position,y_positions)
    # Obtain the updated likelihood
    new_lhd = segment_likelihood(left_tau_location,right_tau_location,left_y_position,right_y_position,len(y),eta)
    # Compute the likelihood ratio
    lhd_ratio = new_lhd - likelihoods[tau_position] - (likelihoods[0] if tau_position==len(tau)-1 else \
        likelihoods[tau_position+1]) - np.log(1-nu) + np.log(len(tau))
    # If there is only one changepoint, then the likelihood ratio has a particular form
    if len(tau)==1:
        lhd_ratio -= np.log(2.0/3)
    # Accept if the likelihood ratio is larger than 0 or when the condition runif < alpha is verified
    accept = (lhd_ratio >= 0) or (np.random.exponential() > -lhd_ratio)
    # Updates
    if accept:
        tau = np.delete(tau,tau_position)
        y_positions = np.delete(y_positions,tau_position)
        if tau_position != len(tau): 
            likelihoods = np.delete(likelihoods,tau_position+1)
        else: 
            likelihoods = np.delete(likelihoods,0)
        likelihoods[tau_position if tau_position != len(tau) else len(likelihoods)-1] = new_lhd
    return accept, tau, len(tau), y_positions, likelihoods
