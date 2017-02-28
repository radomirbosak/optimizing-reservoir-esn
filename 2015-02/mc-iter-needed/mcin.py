#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mcin.py - MC iterations needed
Created: 21.2.2015

Goal: Estimate how many iterations are needed for precise computation of memory capacity.
	Tested: ITERATIONS, ITERATIONS_SKIPPED, ITERATIONS_COEF_MEASURE
	
	modify the memory_capacity function parameters 
	inthe main loop to test one of these dependencies
"""

from library.mc3a import memory_capacity
from matplotlib import pyplot as plt

from numpy import random, zeros, sum, linspace, average, std
from time import time

q = 100
sigma = 0.090
tau = 0.01

W  = random.normal (0,    sigma,  [q, q])
WI = random.uniform(-tau,   tau,  [q, 1])

MAX_ITERATIONS_MEASURED = 10000
MAX_ITERATIONS_SKIPPED = 10000
MAX_ITERATIONS_COEF_MEASURE = 10000
MAX_ITERATIONS_COEF_MEASURE_SKIPPED = 10000

RUNS = 10

def measure_mc(W, WI, **kwargs):
	return sum(memory_capacity(W, WI, **kwargs)[0])


# ---------------
# measure true MC
# ---------------
mcr = zeros(RUNS)
for run in range(RUNS):
	mc = measure_mc(W, WI, memory_max=q, 
		iterations=(MAX_ITERATIONS_SKIPPED+MAX_ITERATIONS_MEASURED),
		iterations_skipped=MAX_ITERATIONS_SKIPPED,
		iterations_coef_measure=MAX_ITERATIONS_COEF_MEASURE,
		iterations_coef_measure_skipped=MAX_ITERATIONS_COEF_MEASURE_SKIPPED,
		runs=1,
		)
	mcr[run] = mc
true_mc_avg = average(mcr)
true_mc_std = std(mcr)

# 1. ITERATIONS_MEA
iterations_measured_bag = list(linspace(2, 10, 9)) \
	+ list(linspace(15, 100 , 10)) \
	+ list(linspace(115, 200 , 10)) 
#	+ list(linspace(215, 1000 , 10))
"""
iterations_measured_bag = list(linspace(100, 1000, 20))
"""
"""
iterations_measured_bag = list(linspace(1000, 10000, 20))
"""

iterations_measured_bag = list(map(int, iterations_measured_bag))

mc_avg = zeros(len(iterations_measured_bag))
mc_std = zeros(len(iterations_measured_bag))

# main loop
mcr = zeros(RUNS)
for i, iterations in enumerate(iterations_measured_bag):
	for run in range(RUNS):
		mc = measure_mc(W, WI, memory_max=q, 
			iterations=(MAX_ITERATIONS_SKIPPED+iterations),
			iterations_skipped=MAX_ITERATIONS_SKIPPED,
			iterations_coef_measure=MAX_ITERATIONS_COEF_MEASURE,
			iterations_coef_measure_skipped=MAX_ITERATIONS_SKIPPED,
			runs=1,
			)
		mcr[run] = mc
	mc_avg[i] = average(mcr)
	mc_std[i] = std(mcr)
	print(i,' of ', len(iterations_measured_bag))


plt.errorbar(iterations_measured_bag, mc_avg, yerr=mc_std)
plt.grid(True)
plt.xlabel("iterations")
plt.ylabel("mc")

plt.show()

"""
# plot the error

plt.errorbar(iterations_measured_bag, mc_avg-true_mc_avg, yerr=mc_std)
plt.grid(True)
plt.xlabel("iterations")
plt.ylabel("mc error")

plt.show()
"""

def pnormal():
	plt.errorbar(iterations_measured_bag, mc_avg, yerr=mc_std)
	plt.grid(True)
	plt.xlabel("iterations")
	plt.ylabel("mc")

	plt.show()


def perror():
	plt.errorbar(iterations_measured_bag, mc_avg-true_mc_avg, yerr=mc_std)
	plt.grid(True)
	plt.xlabel("iterations")
	plt.ylabel("mc error")

	plt.show()