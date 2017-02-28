#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ipdebug.py
Created 11.12.2014

Goal: Try to find errors in IPGauss implementation (as in ip.py)
	Investigate bimodal convergence behavior (is it normal?).
Answer 11.12.2014 14:38: no, there was a bug in the code, I was feeding previous iteration to IPGauss instead of actual net.
"""

from scipy import random, dot, tanh, ones, zeros, std
from library.aux import kindofvector
from matplotlib import pyplot

q = 4
gridx, gridy = 2, 2

sigma = 0.1
ITERATIONS = 200000

eta = 10**-3
mu = 0

def ipgauss(x, y, a, b):
	db = -eta * (-mu/(sigma*sigma) + y/(sigma*sigma) * (2*(sigma*sigma) + 1 - y*y + mu*y))
	da =  eta / a + x * db
	return a + da, b + db


def main():
	"""
	Today's agenda:
		i)   generate initial matrices
		ii)  simulate
		iii) learn gauss
		iv)  simulate
		v)   plot & compare
	"""

	W = random.normal(0, sigma, [q, q])
	WI = random.uniform(-sigma, sigma, [q, 1])
	X = random.uniform(-1, 1, [q])
	

	a = ones([q])
	b = zeros([q])

	S = zeros([q, ITERATIONS])
	S2 = zeros([q, ITERATIONS])
	ahist = zeros([q, ITERATIONS])

	# i?) simulate
	U = random.uniform(-1, 1, [ITERATIONS])
	for it in range(ITERATIONS):
		net = dot(WI, U[it].reshape(1)) + dot(W, X)
		X = tanh( a * net + b)
		S[:, it] = X

	
	# iii) learn
	U = random.uniform(-1, 1, [ITERATIONS])
	for it in range(ITERATIONS):
		net = dot(WI, U[it].reshape(1)) + dot(W, X)
		Y = tanh( a * net + b)
		a, b = ipgauss(net, Y, a, b)
		ahist[:, it] = a
		X = Y

	# i?) simulate2
	U = random.uniform(-1, 1, [ITERATIONS])
	for it in range(ITERATIONS):
		net = dot(WI, U[it].reshape(1)) + dot(W, X)
		X = tanh( a * net + b)
		S2[:, it] = X

	# iv) view histogram
	BINCNT=10
	
	def show_histograms():
		for yplt in range(gridy):
			print("\r {}/{}".format(yplt, gridy), end="")
			for xplt in range(gridx):
				
				indx = yplt*gridx + xplt
				pyplot.subplot(gridy, gridx, indx)
				std1 = std(S[indx, :])
				std2 = std(S2[indx, :])
				pyplot.hist(S[indx,:],  bins=BINCNT, normed=True, label="{}:bfr std1={:6.4f}".format(indx,std1))
				pyplot.hist(S2[indx,:], bins=BINCNT, normed=True, label="{}:atr std2={:6.4f}".format(indx,std2))
				pyplot.grid(True)
				pyplot.legend()
				print("std1={0}, std2={1}".format(std1, std2))
		#pyplot.legend()
		
		pyplot.show()
	show_histograms()

	for ciara in range(q):
		pyplot.plot(range(ITERATIONS), ahist[ciara, :], label="%d"%ciara)
	pyplot.legend()
	pyplot.show()

	print("a = %s" % a)
	print("b = %s" % b)
	print("done.")

if __name__ == '__main__':
	main()