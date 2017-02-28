#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gerplot-test.py - General errorbar plotter - tester
Created 8.3.2015

Goal: Demonstrate usage of general errorbar plotter
"""

from matplotlib import pyplot as plt
import numpy as np
import itertools
import random

from library.aux import try_save_fig
from library.mc3 import memory_capacity
from library.gerplot import gep_parallel_save_errlines

"""
Needs:
	xTicks
	LineTicks
	Iterations
	[other parameters]
Returns:
	Lines, with std
"""

"""
Usage:
"""

sigmas = np.exp(np.linspace(np.log(0.02), np.log(100), 100))
sparsities = [0.9, 0.99, 0.999, 0.9999] # np.linspace(0, 0.90, 10)
ITERATIONS = 100 # bolo tu aj 10000
q = 100
tau = 0.01
savedir = "saved3"


def rv(xval, linepar):
	W = np.random.normal(0, xval, [q, q])
	WI = np.random.uniform(-tau, tau, [q, 1])

	for i, j in itertools.product(range(q), range(q)):
		if random.random() < linepar:
			W[i, j] = 0

	#current_radius = np.max(np.abs(np.linalg.eig(W)[0]))
	#W = W * (xval / current_radius)

	return sum(memory_capacity(W, WI, iterations=1200, iterations_skipped=q, iterations_coef_measure=100)[0])

#Y, Yerr = get_errlines(sigmas, sparsities, rv, iterations=ITERATIONS)
#plot_lines(Y, Yerr, sigmas, sparsities, linelabel="sp. = {lineval}", xlabel="$\sigma$", ylabel="MC")

gep_parallel_save_errlines(sigmas, sparsities, rv, savedir=savedir, iterations=ITERATIONS)
