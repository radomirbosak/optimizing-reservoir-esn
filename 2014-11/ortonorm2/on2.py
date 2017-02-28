#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
on2.py
Created 26.11.2014

Goal: Implement matrix orthonormalization and orthogonalization 
	learning algorithm
"""

from scipy import dot, random, eye, square, zeros, outer
from numpy.linalg import norm

import scipy


def energy(V):
	I = eye(V.shape[0])
	return square(norm(dot(V.transpose(), V) - I))

def energy2(V):
	I = eye(V.shape[0])
	N = norm_cols(V)
	return energy(N)

def grad_energy(V):
	"""orthonormalization"""
	return dot(dot(V, V.transpose()), V) - V

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

def learn(V, eta):
	return V - eta * grad_energy(V)

def learn2(V, eta):
	return V - eta * grad_energy2(V)

def norm_cols(V):
	return V / norm(V, axis=0)

V = random.normal(0, 10, [3,3])
print(V)

E = energy(V)
print("energy before = {}".format(E))
print("norms before= {}".format(norm(V, axis=0)))


eta = 10**-1

minimum = scipy.inf
for it in range(10000):
	V = learn2(V, eta)
	E = energy2(V)
	minimum = min(E, minimum)
	#print(E)
	if E > 10:
		break

print(V)
print(dot(V.transpose(),V))
print("energy after = {}".format(energy2(V)))
print("norms after = {}".format(norm(V, axis=0)))
print('minimum was at it={}/1000 E={}'.format(it, minimum))