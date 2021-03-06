#!/usr/bin/env python
import sys
import argparse
import numpy as np
from collections import Counter
from scipy.signal import periodogram

def detect_periodicity(t, delta=1.0, full_day=True):
	## Determine the range in seconds
	start = int((np.min(t) // 86400) * 86400) if full_day else int(np.floor(np.min(t)))
	if not full_day:
		t -= start
		start = 0
	end = int(((np.max(t) // 86400) + 1) * 86400) if full_day else int(np.floor(np.max(t))+1)
	time_seconds = np.arange(start, end+1, step=delta)

	## Obtain the process dN
	q = Counter(time_seconds[np.searchsorted(time_seconds, t)])
	dN = np.array([q[x] for x in time_seconds])
	dN_star = dN - len(t)/len(time_seconds)

	## Calculate the periodogram 
	fk, Sk = periodogram(dN_star)
	m = len(Sk)

	## Calculate g-score, estimated p-values and periodicity
	g_fish = np.max(Sk) / np.sum(Sk)
	p = delta / fk[np.argmax(Sk)]
	p_val1 = - np.expm1(m * np.log1p(- np.exp(-m * g_fish))) ## (1 - (1 - np.exp(-m * g_fish)) ** m)
	p_val2 = 1 - np.exp(-m * np.exp(-g_fish * len(time_seconds) * (m - 1 - np.log(m)) / (m - g_fish * len(time_seconds))))

	## Return output
	out = {}
	out['p'] = p
	out['pval'] = p_val1
	out['pval_nick'] = p_val2

	return out 