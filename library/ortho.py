#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortho.py
Created 7.2.2015

Goal: Measuring orthogonality and orthonormality. Provide orthonormalization and orthogonalization (gradient descent based) procedure.

"""

from numpy import dot, random, eye, square, zeros, outer, linalg, sum
#from numpy.linalg import norm

def norm_cols(V):
	return V / linalg.norm(V, axis=0)

"""
orthoNORMality
"""

def energy_orthonormal(V):
	I = eye(V.shape[0])
	return square(norm(dot(V.transpose(), V) - I))

def grad_energy_orthonormal(V):
	"""orthonormalization"""
	return dot(dot(V, V.transpose()), V) - V

def learn_orthonormal(V, eta):
	return V - eta * grad_energy_orthonormal(V)

"""
orthoGONality
"""

def energy_orthogonal(V):
	I = eye(V.shape[0])
	N = norm_cols(V)
	return energy(N)

def grad_energy_orthogonal(V):
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

def learn_orthogonal(V, eta):
	return V - eta * grad_energy_orthogonal(V)

def learn_orthogonal_rev(V, eta):
	return V + eta * grad_energy_orthogonal(V)

def orthogonality(V):
	"""
	A measure how orthogonal a matrix is (a number from [0,1])
	0 - column space has dimension 1
	1 - all columns are perpendicular to each other

	It is computed as an average $1 - cos(angle)$, where $angle$ is angle between two matrix columns
	"""
	q = V.shape[0]
	N = norm_cols(V)
	VTV = dot(N, N.transpose())
	return 1 - (sum(abs(VTV)) - q)/(q*q - q)

