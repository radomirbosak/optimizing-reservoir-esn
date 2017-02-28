#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
mcopte.py
Created 18.2.2015

Goal: Calculate the memory capacity depending on changing parameters
	sigma and tau, which determine the recurrent and input weight distributions.
	A difference from mcopt.py is that this script fixes tau and produced only
	1D graph (but with error-bars)
"""


import numpy as np
from numpy import sum, random, meshgrid, zeros, linspace, average, std, sqrt
from matplotlib import pyplot as plt

from library.mc3 import memory_capacity
from library.ortho import orthogonality

q = 100

taus = [10**-7, 10**-8]
posun = 0

#taus = [0.00001, 0.000001, 10**-7, 10**-8]
#posun = 4
#taus = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 10**-7, 10**-8]

np.savetxt('saved/taus.txt', taus)

sigmas = linspace(0.08, 0.095, 20)
np.savetxt('saved/sigmas.txt', sigmas)

#tau = 0.0001
INSTANCES = 1000

Y = zeros(len(sigmas))
Yerr = zeros(len(sigmas))

for ti, tau in enumerate(taus):
	print(tau, "of", str(taus))
	for i, sigma in enumerate(sigmas):
		mcs = zeros(INSTANCES)
		for inm in range(INSTANCES):
			W = random.normal(0, sigma, [q, q])
			WI = random.uniform(-tau, tau, [q, 1])
			mcs[inm] = sum(memory_capacity(W, WI, memory_max=q, runs=1, iterations=15000, iterations_skipped=10000, iterations_coef_measure=1000)[0])
		Y[i] = average(mcs)
		Yerr[i] = std(mcs) # / sqrt(INSTANCES)
		print(i,"of", len(sigmas))

	np.savetxt('saved/avg-t'+str(ti + posun)+'.txt', Y)
	np.savetxt('saved/std-t'+str(ti + posun)+'.txt', Yerr)

	plt.errorbar(sigmas, Y, yerr=Yerr, label=("$\\tau={0}$".format(tau)))

plt.grid(True)
plt.xlabel("sigma: $W = N(0, \\sigma)$")
plt.ylabel("MC, errbar: $1 \\times \\sigma$")
plt.legend(loc=3)
#plt.title("tau = {0}".format(tau))


plt.show()

def replot():
	plt.errorbar(sigmas, Y, yerr=Yerr, label=("$\\tau={0}$".format(tau)))
	plt.grid(True)
	plt.xlabel("sigma: $W = N(0, \\sigma)$")
	plt.ylabel("MC, errbar: $1 \\times \\sigma$")
	plt.legend(loc=3)
	#plt.title("tau = {0}".format(tau))


	plt.show()



