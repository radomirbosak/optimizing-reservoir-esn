#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc5.py
Created 21.3.2015
Based on mc3.py

Goal: Measuring Memory Capacity of reservoirs.

Changes:
	- changed default values for iterations, iterations_skipped, iterations_coef_measure, runs
	- added correlation coefficient correction MC <- MC - q / iterations_coef_measure

"""

from numpy import random, zeros, tanh, dot, linalg, corrcoef, average, std, sqrt

def memory_capacity(W, WI, memory_max=None, iterations=1200, iterations_skipped=None, iterations_coef_measure=100, runs=1, input_dist=(-1., 1.), cc_correction=True):
	"""Calculates memory capacity of a NN 
		[given by its input weights WI and reservoir weights W].
	W  = q x q matrix storing hidden reservoir weights
	WI = q x p matrix storing input weights

	Returns: a tuple (MC, std)
	MC: memory capacity for history 0..(MEMORY_MAX - 1)
		[a vector of length MEMORY_MAX]
	std: standard deviation for each value of MC
	"""

	q, p = WI.shape

	if memory_max is None:
		memory_max = q

	if iterations_skipped is None:
		iterations_skipped = max(memory_max, 100) + 1

	iterations_measured = iterations - iterations_skipped

	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)

	# vector initialization
	X = zeros([q,1])	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q,iterations_measured])

	# generate random input
	u = dist_input() # all input; dimension: [iterations, 1]

	# run 2000 iterations and fill the matrices D and S
	for it in range(iterations):
		X = tanh(dot(W, X) + dot(WI, u[it]))

		if it >= iterations_skipped:
			# record the state of reservoir activations X into S
			S[:, it - iterations_skipped] = X[:,0]

	# prepare matrix D of desired values (that is, shifted inputs)
	assert memory_max < iterations_skipped
	D = zeros([memory_max, iterations_measured])

	for h in range(memory_max): # fill each row
		#FIXME maybe should be: 'iterations - (h+1)', it depends, whether we measure 0th iteration as well
		D[h,:] = u[iterations_skipped - h : iterations - h] 
	
	# calculate pseudoinverse S+ and with it, the matrix WO
	S_PINV = linalg.pinv(S)
	WO = dot(D, S_PINV)

	# do a new run for an unbiased test of quality of our newly trained WO
	# we skip memory_max iterations to have large enough window
	MC = zeros([runs, memory_max]) # here we store memory capacity
	for run in range(runs):
		u = random.uniform(input_dist[0], input_dist[1], iterations_coef_measure + memory_max)
		X = zeros([q,1])
		o = zeros([memory_max, iterations_coef_measure]) # 200 x 1000
		for it in range(iterations_coef_measure + memory_max):
			X = tanh(dot(W, X) + dot(WI, u[it]))
			if it >= memory_max:
				# we calculate output nodes using WO  ( @FIXME maybe not a column, but a row?)
				o[:, it - memory_max] = dot(WO, X)[:,0]

		# correlate outputs with inputs (shifted)
		for h in range(memory_max):
			k = h + 1
			cc = corrcoef(u[memory_max - h : memory_max + iterations_coef_measure - h], o[h, : ]) [0, 1]
			MC[run, h] = cc * cc

	correction = (q / iterations_coef_measure) if cc_correction else 0
	return sum(average(MC, axis=0)) - correction


def main():
	print("I am a library. Please don't run me directly.")

if __name__ == '__main__':
	main()