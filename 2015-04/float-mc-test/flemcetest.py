#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flemcetest.py - Float MC test
Created: 13.4.2015

Goal: Try to scale minimal singular values to achieve greater MC

"""

from numpy import random, zeros, tanh, dot, linalg, \
	corrcoef, average, std, sqrt, hstack
import scipy.linalg
import scipy as sp
import numpy as np

from matplotlib import pyplot as plt
from library.aux import try_save_fig
from scipy.linalg import svd


def stretch_min(L, newmin, oldmin, oldmax):
	return oldmax + (L - oldmax) * (oldmax - newmin) / (oldmax - oldmin)

def main():
	global W, WI, U, L, V, L2
	q = 100
	sigma = 0.09
	tau = 0.1

	# boedecker's definition
	target_later = True
	use_input = False

	W = sp.random.normal(0, sigma, [q, q])
	WI = sp.random.uniform(-tau, tau, q)

	U, L, V = svd(W)		
	smin = L[-1]
	smax = L[0]
	L2 = sp.zeros_like(L)

	smins =  sp.linspace(smin, 0.4, 40)
	mcs = sp.zeros_like(smins)

	for si, new_smin in enumerate(smins):
		L2 = stretch_min(L, new_smin, smin, smax)
		W2 = sp.dot(U, sp.dot(sp.diag(L2), V))
		mc = memory_capacity(W2, WI, memory_max=2*q, 
			iterations_coef_measure=1000, iterations=1000, 
			use_input=use_input, target_later=target_later)
		mcs[si] = sum(mc)

	plt.plot(smins, mcs)
	plt.grid(True)
	plt.xlabel("new $s_{min}$")
	plt.ylabel("memory capacity")
	try_save_fig()
	try_save_fig(ext='pdf')
	plt.show()

def compare_defs():
	global q,sigma, tau, W, WI, mc

	target_later = False
	use_input = True

	q = 20
	sigma = 0.10
	tau = 0.1

	W = sp.random.normal(0, sigma, [q, q])
	WI = sp.random.uniform(-tau, tau, q)
	mc = memory_capacity(W, WI, memory_max=2*q, 
		iterations_coef_measure=100000, iterations=10000, 
		use_input=use_input, target_later=target_later)
	mcsum = sum(mc)
	svds = svd(W, compute_uv=0)

	print("MC = {}\nsmax={}\nsmin={}\nsspread={}".format(mcsum, svds[0], svds[-1], svds[0] / svds[-1]))
	plt.grid(True)
	plt.xlabel("$k$")
	plt.ylabel("$MC_k$")
	plt.title("starting k = {}; use_input = {}; MC = {:.3f}".format(
		int(target_later), use_input, mcsum))
	plt.plot(range(len(mc)), mc)
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
	tanh = lambda x: x
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