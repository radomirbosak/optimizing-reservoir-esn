#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
uununn.py - uniform/uniform, normal/uniform, normal/normal
Created: 21.3.2015

Goal: How is MC affected by selecting input weights from normal distribution? 
	Or reservoir weights from uniform distribution?

"""

from library.mc3 import memory_capacity # computed memory capacity
from library.aux import try_save_fig    # saves figure to a file

from matplotlib import pyplot as plt

import numpy as np

q = 100
sigmas = np.linspace(0.08, 0.1, 20)



tauhats = np.linspace(0.14, 0.17, 20)
ITERATIONS = 100

def getmc(W, WI):
	return np.sum(memory_capacity(W, WI, memory_max=q, runs=1, iterations=1200, iterations_skipped=(q+1))[0])

Y = np.zeros(len(sigmas))
Yerr = np.zeros(len(sigmas))
for i, sigma in enumerate(sigmas):
	mcs = np.zeros(ITERATIONS)
	for it in range(ITERATIONS):
		WI = np.random.uniform(0, tau, [q, 1])
		W = np.random.normal(0, sigma, [q, q])
		mcs[it] = getmc(W, WI)
	Y[i] = np.average(mcs)
	Yerr[i] = np.std(mcs)
	print(i,'of', len(sigmas))

Y2 = np.zeros(len(tauhats))
Yerr2 = np.zeros(len(tauhats))
for i, tauhat in enumerate(tauhats):
	mcs = np.zeros(ITERATIONS)
	for it in range(ITERATIONS):
		WI = np.random.uniform(0, tau, [q, 1])
		W = np.random.uniform(-tauhat, tauhat, [q, q])
		mcs[it] = getmc(W, WI)
	Y2[i] = np.average(mcs)
	Yerr2[i] = np.std(mcs)
	print(i,'of', len(tauhats))



def replot():
	lims = [20, 50]
	fs = 24

	plt.subplot(121)
	plt.xlabel("$\\sigma$", fontsize=fs)
	plt.ylabel("MC")
	plt.ylim(lims)
	plt.grid(True)
	plt.title("normal W, it={}".format(ITERATIONS))
	plt.errorbar(sigmas, Y, Yerr)

	plt.subplot(122)
	plt.xlabel("$\\hat\\tau$", fontsize=fs)
	plt.ylabel("MC")
	plt.ylim(lims)
	plt.grid(True)
	plt.title("uniform W, it={}".format(ITERATIONS))
	plt.errorbar(tauhats, Y2, Yerr2)

	fig = plt.gcf()
	fig.set_size_inches(14, 6)

	try_save_fig(ext="png")
	try_save_fig(ext="eps")
	plt.show()

r = replot