#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc3a.py
Created 21.2.2015
Based on mc3.py

Goal: Measuring Memory Capacity of reservoirs. Modified to include skipping iteration before correlation coefficient measure.

"""

from numpy import random, zeros, tanh, dot, linalg, corrcoef, average, std, sqrt

# default parameters, can be overriden by params dict in memory_capacity function
def_params = {
	'MEMORY_MAX': 50,
	'ITERATIONS': 2000,
	'ITERATIONS_SKIPPED': 1000,
	'ITERATIONS_COEF_MEASURE': 1000,
	'RUNS': 10,
	'INPUT_DIST': (-1., 1.),
}


# dist_WI = lambda: random.uniform(-0.1,0.1,[q,p])
# dist_W = lambda sigma: random.normal(0., sigma, [q,q]) 

def memory_capacity(W, WI, memory_max=50, iterations=2000, iterations_skipped=1000, iterations_coef_measure=1000, iterations_coef_measure_skipped=1000, runs=10, input_dist=(-1., 1.)):
	"""Calculates memory capacity of a NN 
		[given by its input weights WI and reservoir weights W].
	W  = q x q matrix storing hidden reservoir weights
	WI = q x p matrix storing input weights

	Returns: a tuple (MC, std)
	MC: memory capacity for history 0..(MEMORY_MAX - 1)
		[a vector of length MEMORY_MAX]
	std: standard deviation for each value of MC
	"""
	
	iterations_measured = iterations - iterations_skipped

	q, p = WI.shape

	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)

	# vector initialization
	X = zeros([q,1])	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q,iterations_measured])

	# generate random input
	u = dist_input() # all input; dimension: [iterations, 1]

	# run 2000 iterations and fill the matrices D and S
	u = random.uniform(input_dist[0], input_dist[1], iterations_skipped)
	for it in range(iterations_skipped):
		X = tanh(dot(W, X) + dot(WI, u[it]))

	u = random.uniform(input_dist[0], input_dist[1], memory_max + iterations_measured)
	for it in range(memory_max + iterations_measured):
		X = tanh(dot(W, X) + dot(WI, u[it]))

		if it >= memory_max:
			# record the state of reservoir activations X into S
			S[:, it - memory_max] = X[:,0]

	# prepare matrix D of desired values (that is, shifted inputs)
	#assert memory_max < iterations_skipped
	D = zeros([memory_max, iterations_measured])

	for h in range(memory_max): # fill each row
		#FIXME maybe should be: 'iterations - (h+1)', it depends, whether we measure 0th iteration as well
		D[h,:] = u[memory_max - h : memory_max + iterations_measured - h] 
	
	# calculate pseudoinverse S+ and with it, the matrix WO
	S_PINV = linalg.pinv(S)
	WO = dot(D, S_PINV)


	# do a new run for an unbiased test of quality of our newly trained WO
	# we skip memory_max iterations to have large enough window
	MC = zeros([runs, memory_max]) # here we store memory capacity
	for run in range(runs):
		
		X = zeros([q,1])
		o = zeros([memory_max, iterations_coef_measure]) # 200 x 1000

		u = random.uniform(input_dist[0], input_dist[1], iterations_coef_measure_skipped)
		for it in range(iterations_coef_measure_skipped):
			X = tanh(dot(W, X) + dot(WI, u[it]))

		u = random.uniform(input_dist[0], input_dist[1], iterations_coef_measure + memory_max)
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

	return (average(MC, axis=0), std(MC, axis=0) / sqrt(runs))


def main():
	print("I am a library. Please don't run me directly.")

if __name__ == '__main__':
	main()