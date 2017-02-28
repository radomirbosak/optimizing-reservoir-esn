#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mctheano.py
Created 10.2.2015
Based on mc2.py

Goal: Measuring Memory Capacity of reservoirs.

"""

import numpy as np
from numpy import random, zeros, tanh, dot, linalg, corrcoef, average, std, sqrt
import time

import theano as t
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

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

def memory_capacity(W, WI, memory_max=50, iterations=2000, iterations_skipped=1000, iterations_coef_measure=1000, runs=10, input_dist=(-1., 1.)):
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

	q, p = WI.shape[0], 1

	XV = T.dvector('XV')
	uV = T.dscalar('uV')
	VV = T.tanh(T.dot(W, XV) + T.dot(WI, uV))
	f = function([XV, uV], VV)

	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)

	# vector initialization
	X = zeros(q)	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q,iterations_measured])

	# generate random input
	u = dist_input() # all input; dimension: [iterations, 1]

	p1 = time.time()

	# run 2000 iterations and fill the matrices D and S
	for it in range(iterations):
		#X = f(X, u[it])
		X = tanh(dot(W, X) + dot(WI, u[it]))

		if it >= iterations_skipped:
			# record the state of reservoir activations X into S
			S[:, it - iterations_skipped] = X

	p2 = time.time()

	# prepare matrix D of desired values (that is, shifted inputs)
	assert memory_max < iterations_skipped
	D = zeros([memory_max, iterations_measured])

	for h in range(memory_max): # fill each row
		#FIXME maybe should be: 'iterations - (h+1)', it depends, whether we measure 0th iteration as well
		D[h,:] = u[iterations_skipped - h : iterations - h] 
	
	# calculate pseudoinverse S+ and with it, the matrix WO
	S_PINV = linalg.pinv(S)
	WO = dot(D, S_PINV)

	p3 = time.time()

	# do a new run for an unbiased test of quality of our newly trained WO
	# we skip memory_max iterations to have large enough window
	MC = zeros([runs, memory_max]) # here we store memory capacity
	for run in range(runs):
		u = random.uniform(input_dist[0], input_dist[1], iterations_coef_measure + memory_max)
		X = zeros(q)
		o = zeros([memory_max, iterations_coef_measure]) # 200 x 1000
		for it in range(iterations_coef_measure + memory_max):
			#X = f(X, u[it])
			X = tanh(dot(W, X) + dot(WI, u[it]))
			if it >= memory_max:
				# we calculate output nodes using WO  ( @FIXME maybe not a column, but a row?)
				o[:, it - memory_max] = dot(WO, X)

		# correlate outputs with inputs (shifted)
		for h in range(memory_max):
			k = h + 1
			cc = corrcoef(u[memory_max - h : memory_max + iterations_coef_measure - h], o[h, : ]) [0, 1]
			MC[run, h] = cc * cc

	p4 = time.time()

	s = """ initial run: {0}
fill_desired + pinv: {1}
corrcoef run {2}
pretotal: {3}"""
	print(s.format(p2-p1, p3-p2, p4-p3, p4-p1))
	return (average(MC, axis=0), std(MC, axis=0) / sqrt(runs))


def main():
	q = 100
	W = random.normal(0, 0.095, [q, q])
	WI = random.uniform(-0.1, 0.1, q)

	a = sum(memory_capacity(W, WI, runs=1)[0])
	print(a)


if __name__ == '__main__':
	main()