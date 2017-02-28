#!/usr/bin/python3

from ljapun import ljapexp
#def ljapexp(q=150, iterations=40, gamma0=10**-8, sigma=0.1):

f = open('output.dat', 'w')

logsigma = -1.3
while logsigma < -0.7:
	sigma = 10**logsigma
	priemer = ljapexp(q=150, iterations=40, gamma0=10**-8, sigma=sigma)
	#priemer2, disperzia = ljapexp(q=100, iterations=40, gamma0=10**-8, sigma=sigma)
	print('%s\t%s' % (logsigma, priemer), file=f)
	print('\nlogsigma: %s' % logsigma)
	logsigma += 0.1

f.close()
