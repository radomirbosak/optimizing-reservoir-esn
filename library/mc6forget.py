#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc6.py
Created 21.3.2015
Based on mc5.py

Goal: Measuring Memory Capacity of reservoirs.

Changes:
	- removed correlation coefficient correction MC <- MC - q / iterations_coef_measure
	- added input-to-output connections

"""

from numpy import random, zeros, tanh, dot, linalg, \
	corrcoef, average, std, sqrt, hstack
import scipy.linalg

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


def main():
	print("I am a library. Please don't run me directly.")

if __name__ == '__main__':
	main()