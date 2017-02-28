#!/usr/bin/python3

from random import random, uniform, normalvariate, randrange
import math
from matrix import random_matrix, random_vector, \
				   dot, plus, times, vectorize, vectostr, \
				   vector_len
from reservoir import Reservoir
import copy

# input, hidden and output layer size
p = 1 
q = 150
r = 1 #300

# strength of input activations
INPUT_STRENGTH = 10**0
def input_dist():
	return INPUT_STRENGTH*uniform(-1,1)

MU = 0
SIGMA = 3
def norm():
	return normalvariate(MU, SIGMA)

def normgen(mu, sigma):
	def n():
		return normalvariate(mu, sigma)
	return n

def unigen(a,b):
	def u():
		return uniform(a,b)
	return u

def zerogen():
	def zero():	
		return 0
	return zero

# perturbation size = initial separation
gamma0 = 10**-8
ITERATIONS = 400

# matrix and initial activation randomization
r = Reservoir(p, q, r)
r.randomize_matrices(norm)
r.WI = random_matrix(q,p+1, unigen(-0.1,0.1))
r.W  = random_matrix(q,q+1, normgen(0,10**(-0.8)))
r.randomize_vectors(normgen(0,1))

# make a copy of the reservoir
r2 = copy.deepcopy(r)

#perturbation = random_vector(q)
#perturbation = times(perturbation, gamma0 / vector_len(perturbation))
perturbation = [0]*q
perturbation[randrange(q)] = gamma0

#print(perturbation)
r2.X = plus(r2.X, perturbation)

difvec = plus(times(r.X, -1), r2.X)
#print("difvec = %s" % difvec)
f = open('output.dat','w')

exponentsum = 0

lambdy = [0]*ITERATIONS
for it in range(0, ITERATIONS):
	r.I = random_vector(r.p, input_dist)
	r2.I = r.I
	r.fire()
	r2.fire()
	difvec = plus(times(r.X, -1), r2.X)
	gammaK = vector_len(difvec)
	r2.X = plus(r.X, times(difvec, gamma0 / gammaK))
	mylog = math.log(gammaK / gamma0)
	lambdy[it] = mylog
	exponentsum += mylog
	print("\rit: %d;\tpriem: %.4f;\tlog: %.4f" % (it, (exponentsum / (it+1)), mylog), end="")
	if it>1:
		print("%d\t%.4f" % (it, mylog), file=f)

priemer = exponentsum / ITERATIONS

disperzia = sum([ (x-priemer)*(x-priemer) for x in lambdy]) / ITERATIONS

print("")
print("Exponentsum: %.4f" % exponentsum)
print("Ljapunovov exponent: %.4f" % priemer)
print("Disperzia sigma^2: %.4f" % disperzia)

f.close()
