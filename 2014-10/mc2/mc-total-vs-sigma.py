#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc2.py
Created 17.10.2014
Based on mc.py

Goal: Measuring Memory Capacity for random matrices
"""

from numpy import *
import matplotlib.pyplot as plt

p = 1 	# one input node
q = 100 # 100 reservoir nodes
r = 200 # 200 output nodes

params = {
	'MEMORY_MAX': 100,
	'ITERATIONS': 2000,
	'ITERATIONS_SKIPPED': 1000,
	'ITERATIONS_COEF_MEASURE': 1000,
	'RUNS': 1,
	'NETS': 100,
	'POINTS': 50
}

dist_WI = lambda: random.uniform(-0.1, 0.1,[q,p])
dist_W = lambda sigma: random.normal(0., sigma, [q,q]) 
dist_input = lambda: random.uniform(-1., 1., params['ITERATIONS']) # maybe [1,1] ?

def memory_capacity(W, WI, params):
	"""Calculates memory capacity of a NN 
		[given by its input weights WI and reservoir weights W].
	W  = q x q matrix storing hidden reservoir weights
	WI = q x p matrix storing input weights

	Returns: a tuple (MC, std)
	MC: memory capacity for history 0..(MEMORY_MAX - 1)
		[a vector of length MEMORY_MAX]
	std: standard deviation for each value of MC
	"""

	# load parameters to local variables for better readibility
	MEMORY_MAX = params['MEMORY_MAX']
	ITERATIONS = params['ITERATIONS']
	ITERATIONS_SKIPPED = params['ITERATIONS_SKIPPED'] 
	ITERATIONS_MEASURED = ITERATIONS - ITERATIONS_SKIPPED
	ITERATIONS_COEF_MEASURE = params['ITERATIONS_COEF_MEASURE']
	RUNS = params['RUNS']

	

	# vector initialization
	X = zeros([q,1])	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q,ITERATIONS_MEASURED])

	# generate random input
	u = dist_input() # all input; dimension: [ITERATIONS, 1]

	# run 2000 iterations and fill the matrices D and S
	for it in range(ITERATIONS):
		X = tanh(dot(W, X) + dot(WI, u[it]))

		if it >= ITERATIONS_SKIPPED:
			# record the state of reservoir activations X into S
			S[:, it - ITERATIONS_SKIPPED] = X[:,0]

	# prepare matrix D of desired values (that is, shifted inputs)
	assert MEMORY_MAX < ITERATIONS_SKIPPED
	D = zeros([MEMORY_MAX, ITERATIONS_MEASURED])

	for h in range(MEMORY_MAX): # fill each row
		#FIXME maybe should be: 'ITERATIONS - (h+1)', it depends, whether we measure 0th iteration as well
		D[h,:] = u[ITERATIONS_SKIPPED - h : ITERATIONS - h] 
	
	# calculate pseudoinverse S+ and with it, the matrix WO
	S_PINV = linalg.pinv(S)
	WO = dot(D, S_PINV)

	# do a new run for an unbiased test of quality of our newly trained WO
	# we skip MEMORY_MAX iterations to have large enough window
	MC = zeros([RUNS, MEMORY_MAX]) # here we store memory capacity
	for run in range(RUNS):
		u = dist_input()
		X = zeros([q,1])
		o = zeros([MEMORY_MAX, ITERATIONS_COEF_MEASURE]) # 200 x 1000
		for it in range(ITERATIONS_COEF_MEASURE + MEMORY_MAX):
			X = tanh(dot(W, X) + dot(WI, u[it]))
			if it >= MEMORY_MAX:
				# we calculate output nodes using WO  ( @FIXME maybe not a column, but a row?)
				o[:, it - MEMORY_MAX] = dot(WO, X)[:,0]

		# correlate outputs with inputs (shifted)
		for h in range(MEMORY_MAX):
			k = h + 1
			cc = corrcoef(u[MEMORY_MAX - h : MEMORY_MAX + ITERATIONS_COEF_MEASURE - h], o[h, : ]) [0, 1]
			MC[run, h] = cc * cc

	return (average(MC, axis=0), std(MC, axis=0) / sqrt(RUNS))

def kindofvector(vec):
	shp = vec.shape
	if len(shp) == 1:
		print('vector of length %d' % shp[0])
	else:
		if shp[0] == 1:
			print('a long row (with %d columns)' % shp[1])
		elif shp[1] == 1:
			print('a long column (with %d rows)' % shp[0])
		elif shp[0] > shp[1]:
			print('a tall rectangle matrix (%d x %d)' % shp)
		elif shp[0] < shp[1]:
			print('a wide rectangle matrix (%d x %d)' % shp)
		elif shp[0] == shp[1]:
			print('a square matrix (%d x %d)' % shp)
		else:
			print('an alien matrix of shape: %s' % str(shp))

def main_plot_MCk():
	# plot it
	sigma = 0.10

	# initial setup
	W = dist_W(sigma)
	WI = dist_WI()

	# calculate MC for history 0..199
	MC, std = memory_capacity(W, WI, params)

	x = array(range(params['MEMORY_MAX']))
	y = MC
	y.shape = (y.size,)

	plt.errorbar(x, y, yerr=(std * 3))

	plt.grid(True)
	plt.ylabel('correlation coefficient')
	plt.xlabel('memory size')
	plt.ylim([0,1])
	plt.title('Memory capacity ($\sigma$ = %.3f) (confidence = $3\sigma$) (runs = %d) ' % (sigma, params['RUNS']))
	plt.show()



def main_plot_MC_sigma():
	# 0.13s na iteraciu (tu 4000)
	POINTS = params['POINTS']
	NETS = params['NETS']
	#sigmas = linspace(0.001, 0.2, POINTS)
	sigmas = linspace(0.075, 0.1, POINTS)
	#params['RUNS'] = 1

	y = zeros([NETS, POINTS])

	for i, sigma in enumerate(sigmas):
		for net in range(NETS):
			W = dist_W(sigma)
			WI = dist_WI()

			MC, _ = memory_capacity(W, WI, params)
			y[net, i] = sum(MC)
			print("\rsigma: %.3f (%d of %d), net: (%d of %d)" % (sigma, i, POINTS, net, NETS), end="")
	y, error = (average(y, axis=0), std(y, axis=0) / sqrt(NETS))

	x = sigmas
	plt.errorbar(x, y, yerr=(error * 3))
	plt.plot(sigmas, y)

	plt.grid(True)
	plt.ylabel('Memory capacity')
	plt.xlabel('$\sigma_{W^R}$')
	#plt.ylim([0,1])
	plt.title('Memory capacity (confidence = $3\sigma$) (runs = %d) (nets = %d) ' % (params['RUNS'], params['NETS']))
	plt.show()

def main():
	main_plot_MC_sigma()
	#WI = dist_WI()
	#print(WI.shape)
	#main_plot_MCk()

if __name__ == '__main__':
	main()