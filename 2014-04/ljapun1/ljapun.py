from random import random, uniform, normalvariate, randrange
import math
from matrix import random_matrix, random_vector, \
				   dot, plus, times, vectorize, vectostr, \
				   vector_len
from reservoir import Reservoir
import copy


def ljapexp(q=150, iterations=40, gamma0=10**-8, sigma=0.1):
	# input, hidden and output layer size
	p = 1 
	#q = 150
	r = 1 #300

	# strength of input activations
	INPUT_STRENGTH = 10**0
	def input_dist():
		return INPUT_STRENGTH*uniform(-1,1)

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

	# matrix and initial activation randomization
	r = Reservoir(p, q, r)
	r.randomize_matrices(normgen(0,1))
	r.WI = random_matrix(q,p+1, unigen(-0.1,0.1))
	r.W  = random_matrix(q,q+1, normgen(0,sigma))
	#r.randomize_vectors(normgen(0,1))

	# make a copy of the reservoir
	r2 = copy.deepcopy(r)

	#perturbation = random_vector(q)
	#perturbation = times(perturbation, gamma0 / vector_len(perturbation))
	priemersum = 0
	for perturbed_node in range(q):
		r.randomize_vectors(normgen(0,1))
		perturbation = [0 for _ in range(q)] 
		perturbation[perturbed_node] = gamma0

		r2.X = plus(r2.X, perturbation)

		difvec = plus(times(r.X, -1), r2.X)

		exponentsum = 0
		lambdy = [0]*iterations
		for it in range(0, iterations):
			# introduce a random input
			r.I = random_vector(r.p, input_dist)
			r2.I = r.I
			
			# run one setp
			r.fire()
			r2.fire()

			# calculate the difference
			difvec = plus(times(r.X, -1), r2.X)
			gammaK = vector_len(difvec)

			# renormalize
			r2.X = plus(r.X, times(difvec, gamma0 / gammaK))
			
			# record lambda
			mylog = math.log(gammaK / gamma0)
			lambdy[it] = mylog

		priemer = sum(lambdy) / iterations
		disperzia = sum([ (x-priemer)*(x-priemer) for x in lambdy]) / iterations
		priemersum += priemer
		print("\rnodeindex=%d" % perturbed_node, end="")

	return priemersum / q
	#print("")
	#print("Exponentsum: %.4f" % exponentsum)
	#print("Ljapunovov exponent: %.4f" % priemer)
	#print("Disperzia sigma^2: %.4f" % disperzia)

	#f.close()
