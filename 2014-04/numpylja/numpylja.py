#!/usr/bin/python3

from numpy import *
import sys

p = 1
q = 150

gamma0 = 10**-12
ITERS = 1000

def ljapexp(W, WI):
	lambdy = zeros([ITERS])

	lambdasum = 0
	#nody = range(q)
	nody = [random.randint(q) for _ in range(15)]
	for miesto_perturbacie in nody: # range(q)
		X = zeros([q,1])
		X2 = zeros([q,1])
		X2[miesto_perturbacie] += gamma0
		
		for it in range(ITERS):
			I = random.uniform(-1.,1.,[p,1])
			X = tanh(dot(W, X) + dot(WI, I)) 
			X2 = tanh(dot(W, X2) + dot(WI, I))

			difr = X2 - X
			gammaK = sqrt(vdot(difr, difr))

			X2 = X + difr * (gamma0 / gammaK)
			lambdasum += log(gammaK / gamma0)

			# space for optimization: dot(WI, I) computed twice) 
		print("\r\t\t%d/%d " % (miesto_perturbacie, q), end='')
		sys.stdout.flush()
	return lambdasum / ITERS / size(nody)

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
