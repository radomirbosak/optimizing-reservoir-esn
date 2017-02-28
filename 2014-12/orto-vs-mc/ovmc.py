#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ovmc.py
Created 16.12.2014

Goal: Find the relation between matrix orthogonality and memory capacity
"""

from scipy import dot, random, eye, square, zeros, outer, array, sum
from numpy.linalg import norm

from matplotlib import pyplot as plt
from library.mc3 import memory_capacity

def grad_energy2(V):
	"""orthogonalization"""
	n = V.shape[0]
	I = eye(n)
	N = norm_cols(V)
	DN = (dot(N, N.transpose()) - I)

	DEDV = zeros([n,n])
	for i in range(n):
		vi = V[:, i]
		ni = N[:, i]
		DEDV[:, i] = dot(dot(I - outer(ni, ni), DN), ni) / norm(vi)
	return DEDV

def learn2(V, eta):
	return V - eta * grad_energy2(V)

def matrix_orthogonality(V):
	q = V.shape[0]
	N = norm_cols(V)
	VTV = dot(N, N.transpose())
	return 1 - (sum(abs(VTV)) - q)/(q*q - q)

def norm_cols(V):
	return V / norm(V, axis=0)

def plot_everything():
	pass

def main():
	

	ITERATIONS = 100
	mc = zeros(ITERATIONS)
	og = zeros(ITERATIONS)

	#farby = 
	QS = 10

	colors = [
		[0, 0, 0],
		[0, 0, 1],
		[0, 1, 0],
		[1, 0, 0],
		[0, 1, 1],
		[1, 0, 1],
		[0, 1, 1],
	]

	for qpre in range(QS):
		q =  qpre + 2
		for it in range(ITERATIONS):
			W = random.normal(0, 0.1, [q, q])
			WI = random.uniform(-.1, .1, [q, 1])
			mc[it] = sum(memory_capacity(W, WI, memory_max=200, runs=1, iterations_coef_measure=5000)[0][:q+2])
			og[it] = matrix_orthogonality(W)
			print(qpre, QS, it, ITERATIONS)
		plt.scatter(og, mc, marker='+', label=q, c=(colors[qpre % len(colors)]))

	plt.xlabel("orthogonality")
	plt.ylabel("memory capacity")
	plt.grid(True)
	plt.legend()
	plt.show()

def main2():
	ITERATIONS = 10
	q= 100
	for it in range(ITERATIONS):
		W = random.uniform(-0.1, 0.1, [q, q])
		WI = random.uniform(-.1, .1, [q, 1])
		mc = sum(memory_capacity(W, WI, memory_max=200, runs=1)[0])
		og = matrix_orthogonality(W)
		print(og)

if __name__ == '__main__':
	main()