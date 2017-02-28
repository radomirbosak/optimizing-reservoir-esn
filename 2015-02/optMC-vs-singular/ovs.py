#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
ovs.py
Created 26.2.2015

Goal: Calculate the memory capacity depending on largest singular value
of the recurrent matrix
"""


import numpy as np
from numpy import sum, random, meshgrid, zeros, \
	linspace, average, std, sqrt, linalg, max, argmax
from matplotlib import pyplot as plt

from library.mc3 import memory_capacity
from library.aux import try_save_fig

#q = 100

colors = [
		[0, 0, 0],
		[0, 0, 1],
		[0, 1, 0],
		[1, 0, 0],
		[0, 1, 1],
		[1, 0, 1],
		[0, 1, 1],
	]

#@FIXME pridat maxima

tau = 0.01
posun = 0

#taus = [0.00001, 0.000001, 10**-7, 10**-8]
#posun = 4
#taus = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 10**-7, 10**-8]

res_sizes = [16, 36, 49, 64, 100, 225]
LINES = len(res_sizes)
itskip = max( [max(res_sizes) + 1, 200])
np.savetxt('saved/ressizes.txt', res_sizes)

#svals = list(linspace(0.5, 4, 20))
svals = list(linspace(1.5, 2.5, 20))
np.savetxt('saved/svals.txt', svals)

#sigmas =  list(linspace(0.05, 0.10, 20)) + list(linspace(0.11, 0.25, 20))
#np.savetxt('saved/sigmas.txt', sigmas)

#tau = 0.0001
INSTANCES = 10

Y = [None]* LINES
Yerr = [None] * LINES

maxinlineX = zeros(LINES)
maxinlineY = zeros(LINES)

for line in range(LINES):
	Y[line] = zeros(len(svals))
	Yerr[line] = zeros(len(svals))
	#Yerr[line] = zeros(len(sigmas))

for qi, q in enumerate(res_sizes):
	print(q, "of", str(res_sizes))
	for i, sval in enumerate(svals):
		mcs = zeros(INSTANCES)
		for inm in range(INSTANCES):
			W = random.normal(0, 1, [q, q])
			# find the largest singular value and rescale the matrix 
			current_sv = linalg.svd(W, compute_uv=0)[0]
			W = W * (sval / current_sv)

			WI = random.uniform(-tau, tau, [q, 1])
			mcs[inm] = sum(memory_capacity(W, WI, memory_max=q, runs=1, iterations=1200, iterations_skipped=itskip, iterations_coef_measure=100)[0])
		Y[qi][i] = average(mcs)
		Yerr[qi][i] = std(mcs)
		print(i,"of", len(svals))

	maxinlineX[qi] = svals[argmax(Y[qi])]
	maxinlineY[qi] = max(Y[qi])

	np.savetxt('saved/mcs-t'+str(qi + posun)+'.txt', Y[qi])
	np.savetxt('saved/stds-t'+str(qi + posun)+'.txt', Yerr[qi])

	plt.errorbar(svals, Y[qi], label=("res. size={0}".format(q)), yerr=Yerr[qi])
	#plt.scatter(X[qi], Y[qi], label=("res. size={0}".format(q)), c=(colors[qi % len(colors)]))

plt.plot(maxinlineX, maxinlineY, c=(0,0,0), label="maxima")
plt.grid(True)
plt.xlabel("$s_{max}$")
plt.ylabel("MC, errbar: $1 \\times \\sigma$")
plt.legend(loc=1)
#plt.title("tau = {0}".format(tau))


try_save_fig()
plt.show()



def replot():
	for qi in range(LINES):
		plt.errorbar(svals, Y[qi], label=("res. size={0}".format(res_sizes[qi])), yerr=Yerr[qi])
	plt.plot(maxinlineX, maxinlineY, c=(0,0,0), label="maxima")
	plt.grid(True)
	plt.xlabel("$s_{max}$")
	plt.ylabel("MC, errbar: $1 \\times \\sigma$")
	plt.legend(loc=1)
	#plt.title("tau = {0}".format(tau))

	try_save_fig()
	plt.show()


