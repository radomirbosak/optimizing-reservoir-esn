#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lavesing.py - Maximal lambda vs. Largest singular value
Created: 2.3.2015

Goal: Check, how depends the maximal absolute-value eigenvalue on largest singular value of a matrix
	for different matrix sizes
"""

from numpy import random, linalg, linspace, zeros, max, abs, average, std
from matplotlib import pyplot as plt
from library.aux import try_save_fig

res_sizes = [1, 16, 64, 100, 225]
#res_sizes = [1, 2, 4, 8, 16, 64, 128]
LINES  = len(res_sizes)

lambdas = list(linspace(0.5, 1.5, 20))

INSTANCES = 100
LAMBDA_MIN = 0.5
LAMBDA_MAX = 1.5

colors = [
		[0, 0, 0],
		[0, 0, 1],
		[0, 1, 0],
		[1, 0, 0],
		[0, 1, 1],
		[1, 0, 1],
		[0, 1, 1],
	]

def replot():
	for qi, q in enumerate(res_sizes):
		plt.errorbar(X[qi], Y[qi], yerr=Yerr[qi], label=("{0}".format(res_sizes[qi])))

	plt.grid(True)
	plt.xlabel("$\\rho$", fontsize=20, labelpad=-5)
	plt.ylabel("$s_{max}$", fontsize=20)
	plt.legend(loc=2)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	#plt.legend(loc=2,prop={'size':16})

	try_save_fig()
	try_save_fig(ext='pdf')
	try_save_fig(ext='eps')
	plt.show()


X = [None] * LINES
Y = [None] * LINES
Yerr = [None] * LINES

for qi, q in enumerate(res_sizes):
	print(q, "of", str(res_sizes))
	X[qi] = zeros(len(lambdas))
	Y[qi] = zeros(len(lambdas))
	Yerr[qi] = zeros(len(lambdas))
	for li, lambdamax in enumerate(lambdas):
		print("\r",li, "of", len(lambdas), end="")
		mcs = zeros(INSTANCES)
		for i in range(INSTANCES):
			W = random.normal(0, 1, [q, q])
			current_eig = max(abs(linalg.eig(W)[0]))
			current_sv = linalg.svd(W, compute_uv=0)[0]
			mcs[i] = current_sv * lambdamax / current_eig
		X[qi][li] = lambdamax
		Y[qi][li] = average(mcs)
		Yerr[qi][li] = std(mcs)
	print()

replot()