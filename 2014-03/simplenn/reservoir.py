from random import random, uniform
import math
from matrix import random_matrix, random_vector, dot, plus, times, vectorize

def uni11():
	return uniform(-1,1)

class Reservoir:
	def __init__(self, p, q, r):
		self.p = p
		self.q = q
		self.r = r
		
		self.randomize_vectors(uni11)
		self.randomize_matrices(uni11)
		
		self.alpha = 0.0
	
	def randomize_matrices(self, distribution=uni11):
		self.WI = random_matrix(self.q,self.p + 1, distribution)
		self.W  = random_matrix(self.q,self.q + 1, distribution)
		self.WO = random_matrix(self.r,self.q + 1, distribution)
		
	def randomize_vectors(self, distribution=uni11):
		self.I = random_vector(self.p, distribution)
		self.X = random_vector(self.q, distribution)
		self.Y = random_vector(self.r, distribution)

	def fire(self):
		# X = αX + (1-α)[WI*I + W*X]
		self.X = plus(times(self.X, self.alpha), times(plus( dot(self.WI, [1]+self.I), dot(self.W,[1]+self.X)), 1 - self.alpha))
		self.X = tanh(self.X)
		# Y = WO*X
		self.Y = dot(self.WO, [1]+self.X)

@vectorize
def sigmoid(x):
	return 1/(1 + math.exp(-x))

@vectorize
def tanh(x):
	return math.tanh(x)
