#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mvo.py 
-(MC vs. orthogonality)
Created 7.2.2015
Goal: Measure how does the memory capacity correlate with orthogonality. Use orthogonalization process to adjust orthogonality.
"""

from library.ortho import orthogonality, learn_orthogonal, learn_orthogonal_rev
from library.mc3 import memory_capacity

from numpy import random, sum, floor

from matplotlib import pyplot as plt

from library.aux import try_save_fig


sigma = 0.092
tau = 0.001
q = 100
eta = 10**-2

ITERATIONS = 1000
ITERATIONS_REV = 100

LINES = 5

""" a 100-neuron reservoir usually begins with orthogonality 0.92. For every 1/100 of orthogonality we measure current MC
"""

def measure_mc(W, WI):
	return sum(memory_capacity(W, WI, memory_max=100, iterations=7000, iterations_skipped=200)[0])

def measure_og(W):
	return orthogonality(W)

for l in range(LINES):
	W = random.normal(0, sigma, [q, q])
	WI = random.uniform(-tau, tau, [q, 1])

	Wzal = W
	WIzal = WI

	oths = []
	mcs = []
	mc = measure_mc(W, WI)
	og = measure_og(W)

	print("before:", mc)

	mcs.append(mc)
	oths.append(og)

	last_oth = floor(og*100)
	last_oth2 = floor(og*1000)
	last_oth3 = floor(og*10000)

	for it in range(ITERATIONS):
		W = learn_orthogonal(W, eta)
		og = measure_og(W)
		if floor(og*100) != last_oth:
			last_oth = floor(og*100)
			mc = measure_mc(W, WI)	
			mcs.append(mc)
			oths.append(og)
		elif og>0.97 and floor(og*1000) != last_oth2:
			last_oth2 = floor(og*1000)
			mc = measure_mc(W, WI)
			mcs.append(mc)
			oths.append(og)
		elif og>0.999 and floor(og*10000) != last_oth3:
			last_oth3 = floor(og*10000)
			mc = measure_mc(W, WI)
			mcs.append(mc)
			oths.append(og)

	print("after:", mc)
	plt.plot(oths, mcs)

	# #reverse
	# W = Wzal
	# mcs, oths = [], []
	# mc = measure_mc(W, WI)
	# og = measure_og(W)

	# mcs.append(mc)
	# oths.append(og)

	# last_oth = floor(og*100)

	# for it in range(ITERATIONS_REV):
	# 	W = learn_orthogonal_rev(W, eta)
	# 	og = measure_og(W)
	# 	if floor(og*100) != last_oth:
	# 		last_oth = floor(og*100)
	# 		mc = measure_mc(W, WI)
	# 		mcs.append(mc)
	# 		oths.append(og)

	# plt.plot(oths, mcs)
	print("%d of %d" % (l, LINES-1))

plt.xlabel("orthogonality")
plt.ylabel("memory capacity")
plt.grid(True)
plt.title("change of MC during orthogonalization process")
try_save_fig("figures/figure")
plt.show()