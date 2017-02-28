#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emceresi.py - Memory capacity vs. reservoir size
Created: 8.4.2015

Goal: Measure MC for different reservoir sizes.
"""

import sys
import numpy as np
from library.mc6 import memory_capacity
from library.gerplot2 import compute, draw

q = 100
tau = 0.01

reservoir_sizes = [16, 36, 49, 64, 100, 225]

sigmas = np.linspace(0.05, 0.15, 20)
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

measure1 = {
	'savedir': savedir1,
	'xticks': sigmas,
	'xticks_desc_name': 'sigmas',
	'xlabel': "$\\sigma$",
	'rv': lambda xval, lineval: rvgen(xval, lineval, generate_matrix_from_sigma),
}

measure2 = {
	'savedir': savedir2,
	'xticks': rhos,
	'xticks_desc_name': 'rhos',
	'xlabel': "$\\rho$",
	'rv': lambda xval, lineval: rvgen(xval, lineval, generate_matrix_from_rho),
}

measure3 = {
	'savedir': savedir3,
	'xticks': smaxs,
	'xticks_desc_name': 'smaxs',
	'xlabel': "$s_{max}$",
	'rv': lambda xval, lineval: rvgen(xval, lineval, generate_matrix_from_smax),
}

measures = {
	'sigma': measure1,
	'rho': 	measure2,
	'smax': measure3,
}

for m in measures.values():
	m['linelabels'] = ["{}".format(rs) for rs in reservoir_sizes]
	m['ylabel'] = "memory capacity"



def rvgen(xval, lineval, Wgen):
	WI = generate_input_matrix(tau, lineval)
	W = Wgen(xval, lineval)
	
	mc = memory_capacity(W, WI, memory_max=int(lineval*3/2), iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)
	return mc


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
			basic_data = "q={}\ntau={}\n{}={}\nreservoir_sizes={}\n".format(q,tau,m['xticks_desc_name'],repr(xticks),repr(reservoir_sizes))
			compute(m['rv'], savedir, xticks, reservoir_sizes, basic_data)
		elif action == 'draw':
			draw(savedir, xticks, reservoir_sizes, m['xlabel'], m['ylabel'], m['linelabels'], loc=1, save=True)
	except KeyError:
		print("unknown task")
		return

	


if __name__ == '__main__':
	main()