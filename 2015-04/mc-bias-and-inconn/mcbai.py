#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mcbai.py - Memory Capacity Input Connections' effect
Created: 6.4.2015
"""

import numpy as np
from library.mc6 import memory_capacity
from matplotlib import pyplot as plt
from library.aux import try_save_fig

"""
mc function signature for reference
def memory_capacity(W, WI, memory_max=None, 
	iterations=1200, iterations_skipped=None, iterations_coef_measure=100, 
	runs=1, input_dist=(-1., 1.),
	use_input=False):
"""

q = 100

sigma = 0.089 #0.093
tau = 10**(-6)

matrices = 100

it_pinv = 1000
it_cc = 1000


mcs_input = np.zeros(matrices)
mcs_noinput = np.zeros(matrices)

def measure(**kwargs):
	mcs = np.zeros(matrices)
	for it in range(matrices):
		WI = np.random.uniform(-tau, tau, q)
		W = np.random.normal(0, sigma, [q, q])
		mc = memory_capacity(W, WI, memory_max=2*q, iterations=it_pinv, iterations_coef_measure=it_cc, **kwargs)
		mcs[it] = mc
	return mcs

mcs_jaeger = measure(use_input=True, target_later=True)
mcs_boedecker = measure(use_input=False, target_later=True)
mcs_ja = measure(use_input=False, target_later=False)
mcs_other = measure(use_input=True, target_later=False)

def r():
	""" replot the values """
	bar_width = 1
	def barit(where, what, label, c):
		av = np.average(what)
		plt.bar([where], [av], bar_width, yerr=[np.std(what)], label=label, color=c)
		plt.text(where + 0.5, av + 2, str(np.round(av, 2)), ha='center')

	barit(bar_width*0, mcs_jaeger, "jaeger", 'r')
	barit(bar_width*1, mcs_boedecker, "boedecker", 'b')
	barit(bar_width*2, mcs_ja, "me", 'g')
	barit(bar_width*3, mcs_other, "other", 'y')

	# plt.bar([0], [np.average(mcs_jaeger)], bar_width, yerr=[np.std(mcs_jaeger)], label="jaeger")
	# plt.bar([bar_width]  , [np.average(mcs_boedecker)], bar_width, yerr=[np.std(mcs_boedecker)], label="boedecker")
	# plt.bar([bar_width*2]  , [np.average(mcs_ja)], bar_width, yerr=[np.std(mcs_ja)], label="ja")
	# plt.bar([bar_width*3]  , [np.average(mcs_other)], bar_width, yerr=[np.std(mcs_other)], label="other")
	plt.legend(loc=3)
	plt.grid(True)
	plt.title("$\\tau={tau}$".format(tau=tau))

	try_save_fig(ext="png")
	try_save_fig(ext="eps")
	plt.show()

