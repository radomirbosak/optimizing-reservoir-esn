#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gerplot.py - General errorbar plotter
Created 8.3.2015

Goal: Tool for quickly plotting various MC dependencies
"""

from matplotlib import pyplot as plt
import numpy as np

from library.aux import try_save_fig, readable_seconds
from library.mc3 import memory_capacity

import os
import pickle
import time

"""
Needs:
	xTicks
	LineTicks
	Iterations
	[other parameters]
Returns:
	Lines, with std
"""

def gep_get_errlines(xticks, lineticks, random_variable, iterations=10):
	Y = [None] * len(lineticks)
	Yerr = [None] * len(lineticks)

	for li, linepar in enumerate(lineticks):
		print('line {} of {}'.format(li, len(lineticks)))
		Y[li] = np.zeros(len(xticks))
		Yerr[li] = np.zeros(len(xticks))
		for xi, xval in enumerate(xticks):
			print('point {} of {}'.format(xi, len(xticks)))
			mcs = np.zeros(iterations)
			for it in range(iterations):
				mcs[it] = random_variable(xval, linepar)

			Y[li][xi] = np.average(mcs)
			Yerr[li][xi] = np.std(mcs)
	return Y, Yerr


def gep_plot_lines(Y, Yerr, xticks, lineticks, xlabel="", ylabel="", loc=1, linelabel=""):
	for li in range(len(lineticks)):
		plt.errorbar(xticks, Y[li], yerr=Yerr[li], label=linelabel.format(lineval=lineticks[li]))

	plt.grid(True)
	plt.legend(loc=loc)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	try_save_fig()
	plt.show()

def islocked(filename):
	return os.path.exists(filename + ".lock")

def lock(filename):
	open(filename + ".lock", 'a').close()

def unlock(filename):
	os.remove(filename + ".lock")

def gep_parallel_save_errlines(xticks, lineticks, random_variable, savedir="", iterations=10):
	# create dir if not exist
	if not os.path.exists(savedir):
		os.mkdir(savedir)

	# count partial results in various states
	locked, written, nonexistent, total = 0, 0, 0, len(lineticks) * len(xticks)
	for li, linepar in enumerate(lineticks):
		for xi, xval in enumerate(xticks):
			filename = os.path.join(savedir, "l{0}p{1}.txt".format(li, xi))
			if islocked(filename):
				locked += 1
			elif os.path.exists(filename):
				written += 1
			else:
				nonexistent += 1

	# partial results report
	print("BLOCK | Total: {}, Written: {}, Locked: {}, Missing: {}".format(total, written, locked, nonexistent), end="\n\n")

	if nonexistent == 0:
		print('Nothing to do here.')
		return

	time_duration = 1
	locked_since_last_measurement = 0
	# examine and (maybe compute) each datapoint
	for li, linepar in enumerate(lineticks):
		for xi, xval in enumerate(xticks):
			filename = os.path.join(savedir, "l{0}p{1}.txt".format(li, xi))
			# lock or skip datapoint
			if islocked(filename):
				print('(S) l. {} of {}, p. {} of {}, locked, skipping'.format(li + 1, len(lineticks), xi + 1, len(xticks)))
				locked_since_last_measurement += 1
				continue
			elif os.path.exists(filename):
				print('(E) l. {} of {}, p. {} of {}, already exists, skipping'.format(li + 1, len(lineticks), xi + 1, len(xticks)))
				continue
			lock(filename)

			unworked_blocks = (len(lineticks) - li - 1) * len(xticks) + len(xticks) - xi
			expected_seconds = time_duration * unworked_blocks / (1 + locked_since_last_measurement)
			print('(W) l. {} of {}, p. {} of {}, ETA: {}, *starting*...'.format(li + 1, len(lineticks), xi + 1, len(xticks), readable_seconds(expected_seconds)), end='')
			locked_since_last_measurement = 0
			try:
				time_start = time.time()
				mcs = np.zeros(iterations)
				for it in range(iterations):
					mcs[it] = random_variable(xval, linepar)

				np.savetxt(filename, mcs)
				time_duration = time.time() - time_start
			except KeyboardInterrupt:
				print(' ... INTERRUPTED'.format(li, len(lineticks), xi, len(xticks)))
				try:
					os.remove(filename)
				except:
					pass
				return
			finally:
				unlock(filename)
			print(' ... finished')

def gep_parallel_obtain_results(xticks, lineticks, savedir=""):
	""" Returns Y, Yerr, X, is_complete: boolean """
	
	# create dir if not exist
	if not os.path.exists(savedir):
		raise Exception("nonexistend data-dir!")

	# count partial results in various states
	locked, written, nonexistent, total = 0, 0, 0, len(lineticks) * len(xticks)
	linecounts = np.zeros(len(lineticks))

	for li, linepar in enumerate(lineticks):
		for xi, xval in enumerate(xticks):
			filename = os.path.join(savedir, "l{0}p{1}.txt".format(li, xi))
			if islocked(filename):
				locked += 1
			elif os.path.exists(filename):
				written += 1
				linecounts[li] += 1
			else:
				nonexistent += 1

	print("BLOCK | Total: {}, Written: {}, Locked: {}, Missing: {}".format(total, written, locked, nonexistent), end="\n\n")
	if written != total:
		print("warning: showing partial results")

	Y    = [None] * len(lineticks)
	Yerr = [None] * len(lineticks)
	X    = [None] * len(lineticks)
	for li, linepar in enumerate(lineticks):
		X[li] = np.zeros(linecounts[li])
		Y[li] = np.zeros(linecounts[li])
		Yerr[li] = np.zeros(linecounts[li])
		#print(linecounts[li])
		xi_real = 0
		for xi, xval in enumerate(xticks):
			filename = os.path.join(savedir, "l{0}p{1}.txt".format(li, xi))

			if islocked(filename):
				continue
			elif not os.path.exists(filename):
				continue
			
			mcs = np.loadtxt(filename)
			#print(xi_real)
			X[li][xi_real] = xval
			Y[li][xi_real] = np.average(mcs)
			Yerr[li][xi_real] = np.std(mcs)

			xi_real += 1
			
	return X, Y, Yerr

def gep_parallel_plot_lines(X, Y, Yerr, lineticks, xlabel="", ylabel="", loc=1, linelabel=""):
	for li in range(len(lineticks)):
		plt.errorbar(X[li], Y[li], yerr=Yerr[li], label=linelabel.format(lineval=lineticks[li]))

	plt.grid(True)
	plt.legend(loc=loc)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	try_save_fig()
	try_save_fig(ext="eps")
	plt.show()


"""
Usage:

sigmas = np.linspace(0.05, 0.5, 20)
sparsities = [0, 0.1, 0.2, 0.5, 0.8, 0.9] # np.linspace(0, 0.90, 10)
ITERATIONS = 10 # bolo tu aj 10000
q = 100
tau = 0.01


def rv(xval, linepar):
	W = np.random.normal(0, xval, [q, q])
	WI = np.random.uniform(-tau, tau, [q, 1])

	for i, j in itertools.product(range(q), range(q)):
		if random.random() < linepar:
			W[i, j] = 0

	#current_radius = np.max(np.abs(np.linalg.eig(W)[0]))
	#W = W * (xval / current_radius)

	return sum(memory_capacity(W, WI, iterations=1200, iterations_skipped=q, iterations_coef_measure=100)[0])

Y, Yerr = get_errlines(sigmas, sparsities, rv, iterations=ITERATIONS)
plot_lines(Y, Yerr, sigmas, sparsities, linelabel="sp. = {lineval}", xlabel="$\sigma$", ylabel="MC")
"""