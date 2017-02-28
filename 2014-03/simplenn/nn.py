from random import random, uniform
import math
from matrix import random_matrix, random_vector, dot, plus, times, vectorize

class Reservoir:
	def __init__(self, p, q, r):
		self.p = p
		self.q = q
		self.r = r

		self.I = range(p)
		self.X = range(q)
		self.Y = range(r)
		
		self.WI = random_matrix(q,p)
		self.W  = random_matrix(q,q)
		self.WO = random_matrix(r,q)
		
		self.alpha = 0.0

	def fire(self):
		# X = αX + (1-α)[WI*I + W*X]
		X = plus(times(X, self.alpha), times(plus( dot(WI,I), dot(W,X)), 1 - self.alpha))
		X = tanh(X)#sigmoid(X)
		
		# Y = WO*X
		Y = dot(WO, X)

@vectorize
def sigmoid(x):
	return 1/(1 + math.exp(-x))

@vectorize
def tanh(x):
	return math.tanh(x)
