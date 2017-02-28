#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocs.py
Created 7.2.2015

Goal: Measure convergence speed of orthogonalization for various learning rates
"""

from numpy import dot, eye, linalg, sum, abs, outer, random, zeros, real, imag, sort, absolute, angle
from matplotlib import pyplot as plt

from library.aux import try_save_fig

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
		DEDV[:, i] = dot(dot(I - outer(ni, ni), DN), ni) / linalg.norm(vi)
	return DEDV

def learn2(V, eta):
	return V - eta * grad_energy2(V)

def learn2_rev(V, eta):
	return V + eta * grad_energy2(V)

def matrix_orthogonality(V):
	q = V.shape[0]
	N = norm_cols(V)
	VTV = dot(N, N.transpose())
	return 1 - (sum(abs(VTV)) - q)/(q*q - q)

def norm_cols(V):
	return V / linalg.norm(V, axis=0)

def o_vs_it():
	# 1. generate random matrix
	# 2. use orthogonalization process on it
	# 3. track and plot its trajectory (plot orthogonality vs. iteration number)

	# 1.
	sigma = 0.095
	q = 100
	eta = 10**-2
	ITERATIONS = 1000
	MATRICES = 5

	for mind in [-1, -2, -3, -4]:
		eta = 10**mind
		W = random.normal(0, sigma, [q, q])
		orts = zeros(ITERATIONS + 1)
		orts[0] = matrix_orthogonality(W)
		#WI = random.uniform(-.1, .1, [q, 1])

		for it in range(ITERATIONS):
			W = learn2_rev(W, eta)
			orts[it + 1] = matrix_orthogonality(W)

		plt.plot(range(ITERATIONS + 1), orts, label=("$10^{%f}$" % mind ))
		print("{} of {}".format(mind, MATRICES))
	plt.legend()
	plt.grid(True)
	plt.show()

def eig_vs_it():
	global svs
	# 1. generate random matrix
	# 2. use orthogonalization process on it
	# 3. track and plot its trajectory (plot orthogonality vs. iteration number)

	# 1.
	sigma = 0.095
	q = 100
	eta = 10**-2
	ITERATIONS = 600

	W = random.normal(0, sigma, [q, q])
	orts = zeros(ITERATIONS + 1)
	orts[0] = matrix_orthogonality(W)

	f1 = absolute
	f2 = angle

	eigsr = zeros([q, ITERATIONS + 1])
	eigsi = zeros([q, ITERATIONS + 1])
	eigsb = zeros([q, ITERATIONS + 1])
	eigsn = zeros([q, ITERATIONS + 1])
	svs = zeros([q, ITERATIONS + 1])

	eigs = linalg.eig(W)[0]

	eigsr[:,0] = sort(real(eigs))
	eigsi[:,0] = sort(imag(eigs))
	eigsb[:,0] = sort(absolute(eigs))
	eigsn[:,0] = sort(angle(eigs))
	svs[:, 0] = linalg.svd(W, compute_uv=0)
	#WI = random.uniform(-.1, .1, [q, 1])

	for it in range(ITERATIONS):
		W = learn2(W, eta)
		orts[it + 1] = matrix_orthogonality(W)
		eigs = linalg.eig(W)[0]
		eigsr[:,it + 1] = sort(real(eigs))
		eigsi[:,it + 1] = sort(imag(eigs))
		eigsb[:,it + 1] = sort(absolute(eigs))
		eigsn[:,it + 1] = sort(angle(eigs))
		svs[:, it + 1] = linalg.svd(W, compute_uv=0)

	for line in range(q):
		plt.plot(range(ITERATIONS + 1), eigsr[line, :])
	
	plt.title("real parts of eigenvalues")
	plt.grid(True)
	plt.show()

	for line in range(q):
		plt.plot(range(ITERATIONS + 1), eigsi[line, :])
	
	plt.title("imaginary parts of eigenvalues")
	plt.grid(True)
	plt.show()

	for line in range(q):
		plt.plot(range(ITERATIONS + 1), eigsb[line, :])
	
	plt.title("absolute values of eigenvalues")
	plt.grid(True)
	plt.show()

	for line in range(q):
		plt.plot(range(ITERATIONS + 1), eigsn[line, :])
	
	plt.title("angles of eigenvalues")
	plt.grid(True)
	plt.show()

	for line in range(q):
		plt.plot(range(ITERATIONS + 1), svs[line, :])
	
	plt.title("singular values")
	plt.grid(True)
	try_save_fig()
	try_save_fig(ext="pdf")
	plt.show()

def main():
	eig_vs_it()

if __name__ == '__main__':
	main()