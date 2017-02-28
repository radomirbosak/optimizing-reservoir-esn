#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comp.py
Created 6.11.2014

Goal: Compare MC and LE for various randomly generated matrices
"""

from library.mc3 import memory_capacity
from library.lexp import ljapunov_exponent

from numpy import random, sum, eye, array, zeros

p = 1
q = 100

MEM_MAX = 200

def input_matrix():
	return random.uniform(-0.1, 0.1, [q, p])

def generate_random_matrix():
	W = random.normal(0, 0.1, [q, q])
	return W

def generate_permutation_matrix():
	I = eye(q) * 0.9
	random.shuffle(I)
	return I

def generate_special_perm_matrix():
	M = zeros([q,q])
	for i in range(q - 1):
		M[i+1, i] = 0.9
	return M

def generate_special_input():
	WI = zeros([q, p])
	WI[0,0] = 1
	return WI

def test_le_mc(W, WI):
	return ljapunov_exponent(W, WI), memory_capacity(W, WI, memory_max=MEM_MAX, runs=1)[0]

def main():
	from matplotlib import pyplot as plt

	# random matrix
	W, WI = generate_random_matrix(), input_matrix()
	le, mc = ljapunov_exponent(W, WI), memory_capacity(W, WI, memory_max=MEM_MAX, runs=1)[0]
	smc = sum(mc)
	print('RANDOM: le={0}, mc={1}'.format(le, smc))
	plt.subplot(2, 2, 1)
	plt.ylim([0,1])
	plt.title("normal W, uniform WI, mc=%.2f" % smc)
	plt.plot(mc)
	
	W = generate_permutation_matrix()
	le, mc = ljapunov_exponent(W, WI), memory_capacity(W, WI, memory_max=MEM_MAX, runs=1)[0]
	smc = sum(mc)
	print('PERMUT: le={0}, mc={1}'.format(le, smc))
	plt.subplot(2, 2, 2)
	plt.ylim([0,1])
	plt.title("perm W, uniform WI, mc=%.2f" % smc)
	plt.plot(mc)

	W = generate_special_perm_matrix()
	WI = input_matrix()
	le, mc = ljapunov_exponent(W, WI), memory_capacity(W, WI, memory_max=MEM_MAX, runs=1)[0]
	smc = sum(mc)
	print('PERMUT: le={0}, mc={1}'.format(le, smc))
	plt.subplot(2, 2, 3)
	plt.ylim([0,1])
	plt.title("special perm W, uniform WI, mc=%.2f" % smc)
	plt.plot(mc)

	W = generate_special_perm_matrix()
	WI = generate_special_input()
	le, mc = ljapunov_exponent(W, WI), memory_capacity(W, WI, memory_max=MEM_MAX, runs=1)[0]
	smc = sum(mc)
	print('PERMUT: le={0}, mc={1}'.format(le, smc))
	plt.subplot(2, 2, 4)
	plt.ylim([0,1])
	plt.title("special perm W, delta WI, mc=%.2f" % smc)
	plt.plot(mc)

	plt.show()




	print('maco')

if __name__ == '__main__':
	main()