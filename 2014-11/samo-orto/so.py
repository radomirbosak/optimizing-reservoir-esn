#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array, ones, transpose, dot, random, eye, diag, linalg, zeros, multiply, sum

from library.mc3 import memory_capacity

nu = .1


n = 100



I = eye(n)

O = (ones([n, n]) - I) * nu

def energy(M):
	N = dot(M.transpose(), M)
	P = multiply(N, N)
	#print(P)
	return sum(P) - P.trace()

def step(M):
	#return normalize_columns(dot(M, (I - O)))
	N =  (dot(M, (I - nu*dot(M.transpose(), M))))
	#print(N)
	#return N
	return normalize_columns(N)





print(O)
print(I - O)


def normalize_columns(M):
	N = zeros(M.shape)
	for i in range(M.shape[0]):
		N[:, i] = M[:, i] / linalg.norm(M[:, i])
	return N

def main():
	A = random.normal(0, 0.1 , [n, n])
	A = normalize_columns(A)
	# A = array([[3/5, 0], [4/5, 1]])
	# 
	# print(A)
	# print("energy = %f" % energy(A))
	# print(step(A))
	# return

	print(energy(A))

	for it in range(20):
		A = step(A)
		print(energy(A))
	print(A)

	WI = random.uniform(-.1, .1, [n, 1])
	#W = random.normal(0, 0.1, [100,100])
	mc, err = memory_capacity(A, WI, memory_max=200)
	print(sum(mc))
	from matplotlib import pyplot as plt
	plt.errorbar(range(len(mc)), mc, yerr=(err*3))
	plt.ylim([0, 1])
	plt.show()

if __name__ == '__main__':
	main()
	#M = array([[0.5,0.7],[-0.7,0.5]])
	#print(energy(M))
