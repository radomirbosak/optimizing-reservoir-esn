#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np

from library.aux import try_save_fig
from library.mc6forget import memory_capacity

# def memory_capacity(W, WI, memory_max=None, 
# 	iterations=1200, iterations_skipped=None, iterations_coef_measure=100, 
# 	runs=1, input_dist=(-1., 1.),
# 	use_input=False, target_later=False):

q = 20
tau = 0.01
sigmas = np.linspace(0.20, 0.25, 6)
#sigmas = [0.095]*7
mem_max = 60
ks = np.arange(1, mem_max + 1)

def calc():
	global mcks
	mcks = [None for _ in sigmas]

	for si, sigma in enumerate(sigmas):
		print(sigma, 'of', str(sigmas))
		W = np.random.normal(0, sigma, [q, q])
		WI = np.random.uniform(-tau, tau, [q])

		mcks[si] = memory_capacity(W, WI, memory_max=mem_max, 
			iterations=10000, iterations_skipped=1000, iterations_coef_measure=10000,
			runs=10, use_input=False, target_later=True)



def replot():
	for si, sigma in enumerate(sigmas):
		plt.plot(ks, mcks[si], label="$\\sigma = {:.3f}$".format(sigma))

	fig = plt.gcf()
	#plt.xticks(fontsize=20)
	
	fig.set_size_inches(6, 4.5)
	plt.grid(True)
	plt.legend(loc=1)
	plt.xlabel("$k$")
	plt.ylabel("$MC_k$")

	try_save_fig()
	try_save_fig(ext="eps")
	plt.show()
