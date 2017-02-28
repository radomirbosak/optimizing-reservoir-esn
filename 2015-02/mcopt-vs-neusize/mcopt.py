#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
mcopt.py
Created 2.2.2015

Goal: Calculate the memory capacity depending on changing parameters
	sigma and tau, which represent
"""

"""
Optimálna MC v závislosti od:
- veľkosti reservoira
- sila vstupnej matice vs. sila rekurentnej matice
- závisí to lepšie od sigmy alebo od eigenvalue radius?
"""

import numpy
from numpy import sum, random, meshgrid, zeros, linspace, average
from matplotlib import pyplot as plt

from library.mc3 import memory_capacity
from library.ortho import orthogonality

q = 100

sigmas = linspace(0.08, 0.095, 20)
taus = linspace(0.0001, 0.01, 20)
INSTANCES = 5 * 6

X, Y = meshgrid(sigmas, taus)
Z = zeros(X.shape)
i = 0



for sigma_r, tau_r in zip(X,Y):

	j = 0
	for sigma, tau in zip(sigma_r, tau_r):
		mcs = zeros(INSTANCES)
		for inm in range(INSTANCES):
			W = random.normal(0, sigma, [q, q])
			WI = random.uniform(-tau, tau, [q, 1])
			mcs[inm] = sum(memory_capacity(W, WI, memory_max=q)[0])

		Z[i,j] = average(mcs)
		print(sigma, tau)
		j += 1

	i += 1
cmap = plt.get_cmap('PiYG')
c = plt.pcolormesh(X, Y, Z, cmap=cmap)
plt.colorbar()
plt.ylim([0,0.010])
plt.xlim([0.08, 0.095])
plt.xlabel("sigma: $W = N(0, \\sigma)$")
plt.ylabel("tau: $W^I = U(-\\tau, \\tau)$")
plt.show()
	