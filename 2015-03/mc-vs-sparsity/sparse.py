#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparse.py - Memory capacity vs. matrix sparsity
Created 8.3.2015

Goal: Measure the effect of matrix sparsity on memory capacity
"""

from matplotlib import pyplot as plt
import numpy as np
import itertools
import random

from library.aux import try_save_fig
from library.mc3 import memory_capacity

q = 100
tau = 0.01

#sigmas = np.linspace(0.080, 0.5, 20)

radiuses = np.linspace(0.90, 1.1, 20)

sparsities = [0, 0.1, 0.2, 0.5, 0.8, 0.9] # np.linspace(0, 0.90, 10)

ITERATIONS = 1000 # bolo tu aj 10000

Y = [None] * len(sparsities)
Yerr = [None] * len(sparsities)

for li, sparsity in enumerate(sparsities):
	print('sp. {} of {}'.format(li, len(sparsities)))
	Y[li] = np.zeros(len(radiuses))
	Yerr[li] = np.zeros(len(radiuses))
	for si, radius in enumerate(radiuses):
		print('point {} of {}'.format(si, len(radiuses)))
		mcs = np.zeros(ITERATIONS)
		for it in range(ITERATIONS):
			W = np.random.normal(0, 1, [q, q])
			WI = np.random.uniform(-tau, tau, [q, 1])

			for i, j in itertools.product(range(q), range(q)):
				if random.random() < sparsity:
					W[i, j] = 0

			current_radius = np.max(np.abs(np.linalg.eig(W)[0]))
			W = W * (radius / current_radius)

			mcs[it] = sum(memory_capacity(W, WI, iterations=1200, iterations_skipped=q, iterations_coef_measure=100)[0])
		Y[li][si] = np.average(mcs)
		Yerr[li][si] = np.std(mcs)

def replot():
	for li in range(len(sparsities)):
		plt.errorbar(radiuses, Y[li], yerr=Yerr[li], label="sparsity = {:1.2f}".format(sparsities[li]))

	plt.grid(True)
	plt.legend(loc=3)
	plt.xlabel("$|\lambda|_{max}$")
	plt.ylabel("MC")

	try_save_fig()
	plt.show()

replot()