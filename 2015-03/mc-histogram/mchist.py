#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mchist.py - MC Histogram
Created: 12.3.2015
Goal: To display how are memory capacity values distributed for specific
	parameter choice
"""

from library.mc3 import memory_capacity
from library.aux import try_save_fig
from matplotlib import pyplot as plt
import numpy as np

reservoir_size = 100

sigmas = [0.085, 0.09, 0.095, 0.100, 0.105]
sigmas = np.linspace(0.08, 0.11, 7)
tau = 0.01
ITERATIONS = 1000

bins = 20

hists = [None] * len(sigmas)

for i, sigma in enumerate(sigmas):
	
	print(sigma,'of', str(sigmas))
	mcs = np.zeros(ITERATIONS)
	for it in range(ITERATIONS):
		W = np.random.normal(0, sigma, [reservoir_size, reservoir_size])
		WI = np.random.uniform(-tau, tau, [reservoir_size, 1])
		mc = sum(memory_capacity(W, WI, memory_max=reservoir_size, runs=1)[0])
		mcs[it] = mc
		print("\r{} of {}".format(it, ITERATIONS), end="")
	print()
	hists[i] = mcs
	

def replot():
	together= True
	if together:
		max_yticks = 2
		fig = plt.figure()
		fig.subplots_adjust(hspace=0.4)
		
	bins = np.linspace(0, 60, 60)
	for i, sigma in enumerate(sigmas):
		if together:
			sp = plt.subplot(len(sigmas)*100 + 10 + 1 + i)
			if i < len(sigmas) - 1:
				plt.setp( sp.get_xticklabels(), visible=False)

			yloc = plt.MaxNLocator(max_yticks)
			sp.yaxis.set_major_locator(yloc)

		#sp.yaxis.set_ticks(np.arange(70000,80000,2500))
		plt.hist(hists[i], bins=bins, label="$\sigma={:0.3f}$".format(sigma), normed=True, alpha=0.5)
		plt.legend(loc=2,prop={'size':16})
		plt.grid(True)
		plt.xlim([0, 60])
		plt.yticks(fontsize=16)

	#fig.set_size_inches(4, 3)
	#plt.figure(figsize=(4,3))
	fig = plt.gcf()
	plt.xticks(fontsize=20)
	
	#fig.set_size_inches(4, 3)
	try_save_fig(ext="png")
	try_save_fig(ext="eps")
	#plt.savefig("figure3.eps")
	plt.show()

r = replot
replot()