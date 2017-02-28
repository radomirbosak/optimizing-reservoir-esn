#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bidisivas.py - Bi-directional singular value scaling
============

* Created: 	14.4.2015
* Goal: 	Scale both largest and smallest singular value 
			for reservoir matrix and observe the MC
"""

from numpy import random, zeros, tanh, dot, linalg, \
	corrcoef, average, std, sqrt, hstack, meshgrid
import scipy.linalg
import scipy as sp
import numpy as np

from matplotlib import pyplot as plt
from library.aux import try_save_fig
from scipy.linalg import svd


def stretch_vec(L, newmin, newmax, oldmin, oldmax):
	if newmin > newmax:
		return L * 0
	return (L - oldmin) * (newmax / (oldmax - oldmin)) + \
		(L - oldmax) * (newmin / (oldmin - oldmax))

def main():
	global W, WI, U, L, V, L2, X, Y, Z
	q = 200
	sigma = 0.09
	tau = 0.1

	# boedecker's definition
	target_later = True
	use_input = False

	# smins = np.linspace(0, 1.0, 40)
	# smaxs = np.linspace(0, 2.0, 40)
	smins = np.linspace(0.8, 1.0, 40)
	smaxs = np.linspace(0.9, 1.3, 40)

	X, Y = meshgrid(smins, smaxs)
	Z = zeros(X.shape)

	W = sp.random.normal(0, sigma, [q, q])
	WI = sp.random.uniform(-tau, tau, q)

	U, L, V = svd(W)		
	smin_old = L[-1]
	smax_old = L[0]
	L2 = sp.zeros_like(L)

	#smins =  sp.linspace(smin, 0.4, 40)
	mcs = sp.zeros_like(smins)

	
	for ri, (sminr, smaxr) in enumerate(zip(X, Y)):
		for si, (smin, smax) in enumerate(zip(sminr, smaxr)):
			L2 = stretch_vec(L, smin, smax, smin_old, smax_old)
			W2 = sp.dot(U, sp.dot(sp.diag(L2), V))
			mc = memory_capacity(W2, WI, memory_max=2*q, 
				iterations_coef_measure=1000, iterations=1000, 
				use_input=use_input, target_later=target_later)
			mcv = sum(mc)
			if np.isnan(mcv):
				Z[ri, si] = -1
			else:
				Z[ri, si] = mcv
			#Z[si, ri] = random.random()
			print('.', end='')
		print(ri, 'of', len(X))
			#mcs[si] = sum(mc)

	#cmap = plt.get_cmap('PiYG')
	cmap = plt.get_cmap('nipy_spectral')

	c = plt.pcolormesh(X, Y, Z, cmap=cmap)
	plt.colorbar()
	#plt.ylim([0, 2])
	#plt.xlim([0, 2])
	plt.xlabel("smin")
	plt.ylabel("smax")
	try_save_fig()
	try_save_fig(ext="pdf")
	plt.show()
		

def memory_capacity(W, WI, memory_max=None, 
	iterations=1200, iterations_skipped=None, iterations_coef_measure=100, 
	runs=1, input_dist=(-1., 1.),
	use_input=False, target_later=False):
	"""Calculates memory capacity of a NN 
		[given by its input weights WI and reservoir weights W].
	W  = q x q matrix storing hidden reservoir weights
	WI = q x 1 vector storing input weights

	Returns: a non-negative real number MC
	MC: memory capacity sum for histories 1..MEMORY_MAX
	"""
	global S

	# matrix shape checks
	if len(WI.shape) != 1:
		raise Exception("input matrix WI must be vector-shaped!")
	q, = WI.shape
	if W.shape != (q, q):
		raise Exception("W and WI matrix sizes do not match")

	if memory_max is None:
		memory_max = q

	if iterations_skipped is None:
		iterations_skipped = max(memory_max, 100) + 1

	iterations_measured = iterations - iterations_skipped

	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)
	#	tanh = lambda x: x
	# vector initialization
	X = zeros(q)
	if use_input:
		S = zeros([q + 1, iterations_measured])
	else:
		S = zeros([q, iterations_measured])

	# generate random input
	u = dist_input() # all input; dimension: [iterations, 1]

	# run 2000 iterations and fill the matrices D and S
	for it in range(iterations):
		X = tanh(dot(W, X) + dot(WI, u[it]))

		if it >= iterations_skipped:
			# record the state of reservoir activations X into S
			if use_input:
				S[:, it - iterations_skipped] = hstack([X, u[it]])
			else:
				S[:, it - iterations_skipped] = X

	# prepare matrix D of desired values (that is, shifted inputs)
	assert memory_max < iterations_skipped
	D = zeros([memory_max, iterations_measured])
	if target_later:
		# if we allow direct input-output connections, there is no point in measuring 0-delay corr. coef. (it is always 1)
		for h in range(memory_max):
			D[h,:] = u[iterations_skipped - (h+1) : iterations - (h+1)] 
	else:
		for h in range(memory_max): 
			D[h,:] = u[iterations_skipped - h : iterations - h] 
	
	
	# calculate pseudoinverse S+ and with it, the matrix WO
	S_PINV = scipy.linalg.pinv(S)
	WO = dot(D, S_PINV)

	# do a new run for an unbiased test of quality of our newly trained WO
	# we skip memory_max iterations to have large enough window
	MC = zeros([runs, memory_max]) # here we store memory capacity
	for run in range(runs):
		u = random.uniform(input_dist[0], input_dist[1], iterations_coef_measure + memory_max)
		X = zeros(q)
		o = zeros([memory_max, iterations_coef_measure]) # 200 x 1000
		for it in range(iterations_coef_measure + memory_max):
			X = tanh(dot(W, X) + dot(WI, u[it]))
			if it >= memory_max:
				# we calculate output nodes using WO
				if use_input:
					o[:, it - memory_max] = dot(WO, hstack([X, u[it]]))
				else:
					o[:, it - memory_max] = dot(WO, X)

		# correlate outputs with inputs (shifted)
		for h in range(memory_max):
			k = h + 1
			if target_later:
				cc = corrcoef(u[memory_max - k : memory_max + iterations_coef_measure - k], o[h, : ]) [0, 1]
			else:
				cc = corrcoef(u[memory_max - h : memory_max + iterations_coef_measure - h], o[h, : ]) [0, 1]
			MC[run, h] = cc * cc

	return average(MC, axis=0)

if __name__ == '__main__':
	main()