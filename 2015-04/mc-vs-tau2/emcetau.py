#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emcetau.py - Memory capacity vs. tau (input matrix parameter)
Created: 9.4.2015

Goal: Measure MC for different input matrix parameters tau.
"""

import sys
import numpy as np
from library.mc6 import memory_capacity
from library.gerplot2 import compute, draw
import random
import itertools
#import ipdb

q = 100
#tau = 0.01

sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]
taus = [0.1, 0.01, 10**-4, 10**-6, 10**-7, 10**-8]

sigmas = np.linspace(0.05, 0.14, 20)
rhos = np.linspace(0.8, 1.1, 20)
smaxs = np.linspace(1, 2.5, 20)

savedir1 = 'collect-sigmas'
savedir2 = 'collect-rhos'
savedir3 = 'collect-smaxs'

def generate_input_matrix(tau, q):
	return np.random.uniform(-tau, tau, q)

def generate_matrix_from_sigma(sigma, q):
	return np.random.normal(0, sigma, [q, q])

def generate_matrix_from_rho(rho, q):
	M = np.random.normal(0, 1, [q, q])
	return M * (rho / np.max(np.abs(np.linalg.eig(M)[0])))

def generate_matrix_from_smax(smax, q):
	M = np.random.normal(0, 1, [q, q])
	return M * (smax / np.linalg.svd(M, compute_uv=0)[0])

def normalize_rho(W, rho):
	norm = np.max(np.abs(np.linalg.eig(W)[0]))
	if norm != 0:
		return W * (rho / norm)
	else:
		return W

def normalize_smax(W, smax):
	norm = np.max(np.linalg.svd(W, compute_uv=0))
	if norm != 0:
		return W * (smax / norm)
	else:
		return W

def make_sparse(W, sparsity):
	q = W.shape[0]
	for i, j in itertools.product(range(q), range(q)):
		if random.random() < sparsity:
			W[i, j] = 0

def rvsigma(xval, lineval):
	WI = generate_input_matrix(lineval, q)
	W = generate_matrix_from_sigma(xval, q)
	#make_sparse(W, lineval)
	
	mc = memory_capacity(W, WI, memory_max=150, iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)
	return mc

def rvrho(xval, lineval):
	WI = generate_input_matrix(lineval, q)
	W = generate_matrix_from_sigma(1, q)
	#make_sparse(W, lineval)
	#if np.max(np.abs(np.linalg.eig(W)[0])) == 0:
#		ipdb.set_trace()
	
	W = normalize_rho(W, xval)
	
	mc = memory_capacity(W, WI, memory_max=150, iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)
	return mc

def rvsmax(xval, lineval):
	WI = generate_input_matrix(lineval, q)
	W = generate_matrix_from_sigma(1, q)
	#make_sparse(W, lineval)
	#W = W * (xval / np.linalg.svd(W, compute_uv=0)[0])
	W = normalize_smax(W, xval)
	
	mc = memory_capacity(W, WI, memory_max=150, iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)
	return mc


measure1 = {
	'savedir': savedir1,
	'xticks': sigmas,
	'xticks_desc_name': 'sigmas',
	'xlabel': "$\\sigma$",
	'rv': rvsigma,
}

measure2 = {
	'savedir': savedir2,
	'xticks': rhos,
	'xticks_desc_name': 'rhos',
	'xlabel': "$\\rho$",
	'rv': rvrho,
}

measure3 = {
	'savedir': savedir3,
	'xticks': smaxs,
	'xticks_desc_name': 'smaxs',
	'xlabel': "$s_{max}$",
	'rv': rvsmax,
}

measures = {
	'sigma': measure1,
	'rho': 	measure2,
	'smax': measure3,
}

for m in measures.values():
	m['linelabels'] = ["{}".format(t) for t in taus]
	m['ylabel'] = "memory capacity"


"""
def compute(random_variable, savedir, xticks, lineticks, basic_data=""):

def draw(savedir, xticks, lineticks, xlabel="", ylabel="", linelabels=[], save=False):
"""

def main():
	if len(sys.argv) < 2:
		print("Usage: '{0:} sigma compute' or {0:} sigma draw".format(sys.argv[0]))
		return

	task = sys.argv[1]
	action = sys.argv[2]

	try:
		m = measures[task]
		savedir, xticks = m['savedir'], m['xticks']

		if action == 'compute':
			basic_data = "q={}\n{}={}\ntaus={}\n".format(q,m['xticks_desc_name'],repr(xticks),repr(taus))
			compute(m['rv'], savedir, xticks, taus, basic_data)
		elif action == 'draw':
			draw(savedir, xticks, taus, m['xlabel'], m['ylabel'], m['linelabels'], loc=1, save=True)
	except KeyError:
		print("unknown task")
		return


if __name__ == '__main__':
	main()