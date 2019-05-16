#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from collections import Counter
from scipy.signal import periodogram

## Import the times
t = np.loadtxt(sys.stdin)

## Determine the range in seconds
start = int((np.min(t) // 86400) * 86400)
end = int(((np.max(t) // 86400) + 1) * 86400)
time_seconds = np.arange(start, end)

## Obtain the process dN
q = Counter(time_seconds[np.searchsorted(time_seconds, t)])
dN = np.array([q[x] for x in time_seconds])
dN_star = dN - len(t)/len(time_seconds)

## Calculate the periodogram 
fk, Sk = periodogram(dN_star)
m = len(Sk)

## Calculate g-score, estimated p-values and periodicity
g_fish = np.max(Sk) / np.sum(Sk)
p = 1/fk[np.argmax(Sk)]
p_val1 = (1 - (1 - np.exp(-m * g_fish)) ** m)
p_val2 = 1 - np.exp(-m * np.exp(-g_fish * len(time_seconds) * (m - 1 - np.log(m)) / (m - g_fish * len(time_seconds))))

## Print output
print('Estimated periodicity:', p)
print('p-value (Jenkins & Pristley, 1957):', p_val1)
print('p-value (Rubin-Delanchy & Heard, 2014):', p_val2)