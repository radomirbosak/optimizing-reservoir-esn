#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lexp.py
Created 5.11.2014
Based on numpylja.py

Goal: Measuring Ljapunov exponent of reservoirs.
"""

from numpy import random, zeros, tanh, dot, sqrt, vdot, log, average

def ljapunov_exponent(W, WI, gamma0=10**-12, iterations=1000, input_dist=(-1., 1.)):
	q, p = WI.shape
	lambdasum = 0

	#nody = range(q)
	nody = [random.randint(q) for _ in range(15)]

	for mp, miesto_perturbacie in enumerate(nody):
		X = zeros(q)
		X2 = zeros(q)
		X2[miesto_perturbacie] += gamma0
		
		lambdas = zeros(iterations)
		for it in range(iterations):
			I = random.uniform(input_dist[0], input_dist[1], p)
			X = tanh(dot(W, X) + dot(WI, I)) 
			X2 = tanh(dot(W, X2) + dot(WI, I))

			difr = X2 - X
			gammaK = sqrt(difr.dot(difr))

			X2 = X + difr * (gamma0 / gammaK)
			lambdas[it] = log(gammaK / gamma0)
		
		lambdasum += average(lambdas)

	return lambdasum / len(nody)

def main():
	print("I am a library. Please don't run me directly.")

if __name__ == '__main__':
	main()