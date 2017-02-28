#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import random
from lexp import ljapunov_exponent

def stara_procedura():
	gammy = [10**-8, 10**-9,10**-10,10**-11, 10**-12, 10**-13,10**-14, 10**-15]
	NETS = 4
	print('logsigma;\tlambda')
	f = open("krivky.dat", "w")
	for logsigma in linspace(-1.25, -0.9, 8):
		sigma = 10**logsigma
		print("%s" % logsigma, file=f, end="")
		for gamma0 in gammy:
			lambdasum2 = 0
			print("gamma: %s" % gamma0)
			for net in range(NETS):
				WI = random.uniform(-.1, .1, [q,p]) 
				W = random.normal(0., sigma, [q,q])
				lambdasum2 += ljapexp(W, WI)
				print('\rnet %d' % net, end='')
			lambdaend = lambdasum2 / NETS
			print("\r%s;\t%s" % (logsigma, lambdasum2/NETS))
			print("\t%s" % lambdaend, file=f, end="")
		print("", file=f)
	f.close()

def main():
	WI = random.uniform(-0.1, 0.1, [100, 1])
	W = random.normal(0., 0.2, [100, 100])
	le = ljapunov_exponent(W, WI)
	print(le)

if __name__ == '__main__':
	main()