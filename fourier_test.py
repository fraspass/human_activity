#!/usr/bin/env python
import sys
import argparse
import numpy as np
from collections import Counter
from scipy.signal import periodogram
from fourier import detect_periodicity

## Parser for delta
full_day = False
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--delta", type=float, dest="delta", default=0.0, \
    help="Resolution of the observations. Default: calculated as the maximal resolution of the observations.")
parser.add_argument("-f", "--fullday", action = 'store_true', dest = "full_day", default = full_day,
    help = 'Calculate the periodicity considering full days of observation.')
args = parser.parse_args()
delta = args.delta
if args.full_day:
    full_day = True

## Import the times
t = np.loadtxt(sys.stdin)

## Calculate delta
if delta == 0.0:
    delta = 10 ** float(-np.max(np.array([len(str(x).split('.')[1].strip('0')) for x in t])))
print 'Delta: ' + str(delta)

## Calculate the periodicity 
out = detect_periodicity(t, delta = delta, full_day = full_day)

## Print output
print 'Estimated periodicity: ' + str(out['p'])
print 'p-value (Jenkins & Pristley, 1957): ' + str(out['pval'])
print 'p-value (Rubin-Delanchy & Heard, 2014): ' + str(out['pval_nick'])