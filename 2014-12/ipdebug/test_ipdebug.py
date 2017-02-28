#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import random, dot, tanh, ones, zeros, std, array

sigma = 1
mu = 0
eta = 1

def ipgauss(x, y, a, b):
	db = -eta * (-mu/(sigma*sigma) + y/(sigma*sigma) * (2*(sigma*sigma) + 1 - y*y + mu*y))
	da =  eta / a + x * db
	return da,db

def main():
	a = array([1, 1])
	b = array([0, 0])
	x = array([-0.5, 1])
	y = tanh( a * x + b)

	dbma = [-(-0.46 * 2.79), -(0.76*2.42)]


	print("ma byt:", dbma[0], dbma[1])
	print("je:", ipgauss(x, y, a, b)[0])





if __name__ == '__main__':
	main()