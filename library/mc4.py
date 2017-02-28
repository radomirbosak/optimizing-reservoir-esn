#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc4.py
Created 5.11.2014
Based on mc3.py

Goal: Measuring Memory Capacity of reservoirs.

Modified to include activation function parameters

"""

from numpy import random, zeros, tanh, dot, linalg, corrcoef, average, std, sqrt, multiply

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

def tanh_ab(x, ab):
	""" Range: (0, 1) """
	return tanh(ab[:,0]*x + ab[:,1])

def sigmoid_ab(x, ab):
	""" Range: (-1, 1) """
	return 1 / (1 + exp(-ab[:,0] * x + ab[:,1]))

q = 150

def get_def_act_params(q):
	activation_function_params = zeros([q, 2])
	for i in range(q):
		activation_function_params[i, 0] = 1
		activation_function_params[i, 1] = 0
	return activation_function_params


def memory_capacity(W, WI, memory_max=50, iterations=2000, iterations_skipped=1000, 
					iterations_coef_measure=1000, runs=10, input_dist=(-1., 1.),
					activation_function=tanh_ab, activation_parameters=None):
	""" Calculate memory capacity of a NN 
		[given by its input weights WI and reservoir weights W].
	Args:
		W: q x q matrix storing hidden reservoir weights
		WI: q x p matrix storing input weights
		iterations: how many iterations are run to compute the pseudoinverse
		iterations_skipped: how many iterations are skipped to diminish initial state effect
		iterations_coef_measure: how many iterations are used to measure the correlation coeffitient
		runs: how many times is reservoir initialized a this test run 
		input_dist: from which interval to pick to input (uniform) distribution
		activation_function: function R^q x R^(q x nparam) -> R^q, acts element-wise (R x R^nparam -> R)
		activation_parameters: an q x nparam array

	Returns: 
		a tuple (MC, std)
		MC: memory capacity for history 0..(MEMORY_MAX - 1)
			[a vector of length MEMORY_MAX]
		std: standard deviation for each value of MC
	"""

	iterations_measured = iterations - iterations_skipped

	q, p = WI.shape

	if activation_parameters is None:
		activation_parameters = get_def_act_params(q)

	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)

	# vector initialization
	X = zeros([q,1])	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q,iterations_measured])

	# generate random input
	u = dist_input() # all input; dimension: [iterations, 1]

	# run 2000 iterations and fill the matrices D and S
	for it in range(iterations):

		X = activation_function(dot(W, X) + dot(WI, u[it]), activation_parameters)

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
			X = activation_function(dot(W, X) + dot(WI, u[it]), activation_parameters)
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