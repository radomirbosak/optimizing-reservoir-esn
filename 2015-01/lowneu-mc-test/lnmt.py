#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lnmt.py
Created 28.1.2015

Goal: 	Measure memory capacity for reservoirs with small number of neurons.
Test if they satisfy Jaeger's upper bound for memory capacity.
"""

from numpy import *
from library.mc3 import memory_capacity as mc3
from library.mc4 import memory_capacity as mc4
from matplotlib import pyplot as plt

memory_capacity = mc3



def plotit(sigma=0.1, q=100):
	#q = 100
	global x, W, WI, p, mc, err
	global mc
	p = 1

	MMAX = 300

	WI = random.uniform(-0.1, 0.1, [q, p])
	W = random.normal(0, sigma, [q, q])

	mc, err = memory_capacity(W, WI, memory_max=MMAX, runs=1)
	x  = range(MMAX)
	print('mc=',sum(mc))

	plt.errorbar(x, mc, yerr=err*3)
	plt.show()

def rast():
	q=100
	p = 1
	sigma=0.1
	WI = random.uniform(-0.1, 0.1, [q, p])
	W = random.normal(0, sigma, [q, q])

	velkosti = range(150,2000,100)
	pridane = zeros([len(velkosti)])
	for i, MMAX in enumerate(velkosti):
		pridane[i] = sum(memory_capacity(W, WI, memory_max=MMAX, runs=10)[0])
		print('.', end='')

	plt.plot(velkosti, pridane)
	plt.show()













