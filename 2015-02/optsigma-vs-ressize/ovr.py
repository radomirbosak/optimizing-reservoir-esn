#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
ovr.py
Created 25.2.2015

Goal: Calculate the memory capacity depending on changing parameters
	sigma for various reservoir sizes
"""


import numpy as np
from numpy import sum, random, meshgrid, zeros, linspace, average, std, sqrt
from matplotlib import pyplot as plt

from library.mc3 import memory_capacity
from library.aux import try_save_fig

#q = 100

#@FIXME pridat maxima

tau = 0.01
posun = 0

#taus = [0.00001, 0.000001, 10**-7, 10**-8]
#posun = 4
#taus = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 10**-7, 10**-8]

res_sizes = [16, 36, 49, 64, 100, 225]
LINES = len(res_sizes)

itskip = max( max(res_sizes) + 1, 200)


np.savetxt('saved/ressizes.txt', res_sizes)


sigmas =  list(linspace(0.05, 0.10, 20)) + list(linspace(0.11, 0.25, 20))
np.savetxt('saved/sigmas.txt', sigmas)

#tau = 0.0001
INSTANCES = 30

Y = [None]* LINES
Yerr = [None] * LINES

for line in range(LINES):
	Y[line] = zeros(len(sigmas))
	Yerr[line] = zeros(len(sigmas))

for qi, q in enumerate(res_sizes):
	print(q, "of", str(res_sizes))
	for i, sigma in enumerate(sigmas):
		mcs = zeros(INSTANCES)
		for inm in range(INSTANCES):
			W = random.normal(0, sigma, [q, q])
			WI = random.uniform(-tau, tau, [q, 1])
			mcs[inm] = sum(memory_capacity(W, WI, memory_max=q, runs=1, iterations=1200, iterations_skipped=itskip, iterations_coef_measure=100)[0])
		Y[qi][i] = average(mcs)
		Yerr[qi][i] = std(mcs) # / sqrt(INSTANCES)
		print(i,"of", len(sigmas))

	np.savetxt('saved/avg-t'+str(qi + posun)+'.txt', Y[qi])
	np.savetxt('saved/std-t'+str(qi + posun)+'.txt', Yerr[qi])

	plt.errorbar(sigmas, Y[qi], yerr=Yerr[qi], label=("res. size={0}".format(q)))

plt.grid(True)
plt.xlabel("sigma: $W = N(0, \\sigma)$")
plt.ylabel("MC, errbar: $1 \\times \\sigma$")
plt.legend(loc=3)
#plt.title("tau = {0}".format(tau))


try_save_fig()
plt.show()


def replot():
	for line in range(LINES):
		plt.errorbar(sigmas, Y[line], yerr=Yerr[line], label=("res. size={0}".format(q)))
		plt.grid(True)
		plt.xlabel("sigma: $W = N(0, \\sigma)$")
		plt.ylabel("MC, errbar: $1 \\times \\sigma$")
		plt.legend(loc=3)
	#plt.title("tau = {0}".format(tau))

	try_save_fig()
	plt.show()



