#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
act-hist.py
Created 17.10.2014
Based on mc-total-vs-sigma.py

Goal: Measuring activation values in reservoir
"""

from numpy import *
import matplotlib.pyplot as plt

from scipy.stats import norm, shapiro

# for legend
import matplotlib.lines as mlines

p = 1 	# one input node
q = 100 # 100 reservoir nodes
r = 200 # 200 output nodes

params = {
	'MEMORY_MAX': 100,
	'ITERATIONS': 1020,
	'ITERATIONS_SKIPPED': 1000,
	'ITERATIONS_COEF_MEASURE': 1000,
	'RUNS': 1,
	'NETS': 100,
	'POINTS': 50
}

dist_WI = lambda: random.uniform(-0.1, 0.1,[q,p])
dist_W = lambda sigma: random.normal(0., sigma, [q,q]) 
dist_input = lambda: random.uniform(-1., 1., params['ITERATIONS']) # maybe [1,1] ?

def compute_histogram(W, WI, params):
	# load parameters to local variables for better readibility
	MEMORY_MAX = params['MEMORY_MAX']
	ITERATIONS = params['ITERATIONS']
	ITERATIONS_SKIPPED = params['ITERATIONS_SKIPPED'] 
	ITERATIONS_MEASURED = ITERATIONS - ITERATIONS_SKIPPED
	ITERATIONS_COEF_MEASURE = params['ITERATIONS_COEF_MEASURE']
	RUNS = params['RUNS']

	# vector initialization
	X = zeros([q,1])	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q, ITERATIONS_MEASURED])
	PS = zeros([q, ITERATIONS_MEASURED])

	# generate random input
	u = dist_input() # all input; dimension: [ITERATIONS, 1]

	# run 2000 iterations and fill the matrices D and S
	for it in range(ITERATIONS):
		PX = dot(W, X) + dot(WI, u[it])
		X = tanh(PX)
		if it >= ITERATIONS_SKIPPED:
			# record the state of reservoir activations X into S
			PS[:, it - ITERATIONS_SKIPPED] = PX[:, 0]
			S[:, it - ITERATIONS_SKIPPED] = X[:, 0]

	#return histogram(S, bins=10, range=(0, 1), density=True)
	return S, PS


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

def main_plot_histogram():
	sigma = 0.10

	# initial setup
	W = dist_W(sigma)
	WI = dist_WI()

	#h, hist_edges = compute_histogram(W, WI, params)
	S, PS = compute_histogram(W, WI, params)
	S = S.flatten()
	PS = PS.flatten()
	#kindofvector(h)
	#kindofvector(hist_edges)
	#print(h)
	#print(hist_edges)
	#plt.plot(hist_edges, h)
	BINCNT = 100

	plt.hist(S,  bins=BINCNT, normed=True, histtype='step', alpha=1, label="act after tanh", color="b")
	plt.hist(PS, bins=BINCNT, normed=True, histtype='step', alpha=1, label="act before tanh", color="g")

	
	#W = shapiro(S)
	print("S size = ", S.size)
	print("shapiro S = ",shapiro(S))
	print("shapiro PS = ",shapiro(PS))
	stdS = std(S)
	print("stdS=",stdS)
	stdPS = std(PS)
	print("stdPS=", stdPS)

	x = linspace(-1, 1, 100)
	y = norm.pdf(x, loc=0, scale=stdS)
	plt.plot(x,y, color="b", alpha=0.2)

	y = norm.pdf(x, loc=0, scale=stdPS)
	plt.plot(x,y, color="g", alpha=0.2)


	#blue_line = mlines.Line2D([], [], color='blue', marker='.', markersize=15, label='Blue stars')
	

	plt.grid(True)
	plt.ylabel('density')
	plt.xlabel('activation value')
	plt.xlim([-1, 1])
	plt.title('activation distibution in reservoir ($\sigma_{blue}$=%.2f, $\sigma_{green}$=%.2f)' % (stdS, stdPS))

	plt.legend()
	plt.show()

def main():
	main_plot_histogram()

if __name__ == '__main__':
	main()