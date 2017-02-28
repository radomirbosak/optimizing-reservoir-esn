#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mvsrs.py - MC vs. Sigma, Rho, S_max
Created: 7.4.2015


"""
import numpy as np
from library.mc6 import memory_capacity
from library.aux import try_save_fig

from time import time
import itertools
import pickle
from matplotlib import pyplot as plt, rc

import os.path
import os
import sys

"""
def memory_capacity(W, WI, memory_max=None, 
	iterations=1200, iterations_skipped=None, iterations_coef_measure=100, 
	runs=1, input_dist=(-1., 1.),
	use_input=False, target_later=False):
"""

savedir = 'collect-closesig'
q = 100
tau = 0.01
xlabel = "$\\sigma$"

sigmas = np.linspace(0.05, 0.15, 20) #sigmas
rhos = np.linspace(0.8, 1.1, 20) #rhos
smaxs = np.linspace(1, 2.5, 20) # smaxs

sigclose = np.linspace(0.085, 0.095)
rhoclose = np.linspace(0.96, 0.99)

xticks = sigclose

def save_basic_data():
	save_string = "q = {}\n" "tau={}\n" "xticks={}\n"
	fullfile = os.path.join(savedir, "basic-data.txt")
	if not os.path.exists(fullfile):
		with open(fullfile, "w") as f:
			f.write(save_string.format(q, tau, repr(xticks)))

def save_values(values):
	i,fname, ext = 0, 'data', 'pickle'

	fullfile = os.path.join(savedir, fname) + str(i) + "." + ext
	while os.path.exists(fullfile):
		i += 1
		fullfile = os.path.join(savedir, fname) + str(i) + "." + ext

	with open(fullfile, "wb") as f:
		pickle.dump(values, f)

	save_basic_data()

def generate_input_matrix(tau):
	return np.random.uniform(-tau, tau, q)

def generate_matrix_from_sigma(sigma):
	return np.random.normal(0, sigma, [q, q])

def generate_matrix_from_rho(rho):
	M = np.random.normal(0, 1, [q, q])
	return M * (rho / np.abs(np.linalg.eig(M)[0][0]))

def generate_matrix_from_smax(smax):
	M = np.random.normal(0, 1, [q, q])
	return M * (smax / np.linalg.svd(M, compute_uv=0)[0])

def compute():
	global values

	if not os.path.isdir(savedir):
		print("dir {} does not exist, creating".format(savedir))
		os.mkdir(savedir)

	total_values = get_total_values()
	total_lengths = [len(total_values[si]) for si, _ in enumerate(xticks)]

	start = time()
	t = time()

	values = [list() for _ in xticks]
	try:
		for it in itertools.count():
			for si, sigma in enumerate(xticks):
				W = generate_matrix_from_sigma(sigma) #generate_matrix_from_sigma(sigma)
				WI = generate_input_matrix(tau)

				mc = memory_capacity(W, WI, memory_max=150, iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)
				values[si].append(mc)
			print("counted to", it)
			if time() > t + 10:
				xtickstds = np.zeros(len(xticks))
				for si, sigma in enumerate(xticks):
					xtickstds[si] = np.std(values[si]) / np.sqrt(len(values[si]) + total_lengths[si])
				print("max standard error of MC sample average:", np.max(xtickstds))
				t = time()
	except KeyboardInterrupt:
		save_values(values)

	print("total:", round(time() - start), 'seconds')


def get_total_values():
	total_values = [list() for _ in xticks]
	for file in os.listdir(savedir):
		if file.endswith(".pickle"):
			fullfile = os.path.join(savedir, file)
			newvalues = pickle.load(open(fullfile, "rb"))
			for si, _ in enumerate(xticks):
				total_values[si].extend(newvalues[si])
	return total_values


def compute_lines():
	Y = np.zeros(len(xticks))
	Yerr = np.zeros(len(xticks))
	for si, _ in enumerate(xticks):
		Y[si] = np.average(total_values[si])
		Yerr[si] = np.std(total_values[si])
	return Y, Yerr


def collect():
	global total_values, Y, Yerr

	total_values = get_total_values()
	Y, Yerr = compute_lines()
	draw(Y, Yerr)
	

def draw(Y, Yerr, save=True):
	global xmax, ymax
	ymax = np.max(Y)
	xmax = xticks[np.argmax(Y)]

	plt.errorbar(xticks, Y, yerr=Yerr)
	plt.xlabel(xlabel, fontsize=24, labelpad=-3)
	plt.ylabel("memory capacity", fontsize=24)
	#rc('xtick', labelsize=24) 
	#rc('ytick', labelsize=24) 
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.grid()
	#plt.text(xmax, ymax, "({:.3f}, {:.3f})".format(xmax, ymax), fontsize=24)
	if save:
		try_save_fig()
		try_save_fig(ext="pdf")
	plt.show()


def main():
	if len(sys.argv) < 2:
		print("compute or draw?")
		return

	action = sys.argv[1]

	if action == "compute":
		compute()
	elif action == "draw":
		collect()

if __name__ == '__main__':
	main()