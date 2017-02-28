from random import random, uniform
import math

def uni11():
	return uniform(-1,1)

def random_matrix(n,m=None,distribution=uni11):
	if m is None:
		m = n
	return [random_vector(m, distribution) for _ in range(n)]

def random_vector(n, distribution=uni11):
	return [distribution() for _ in range(n)]


def dot(matrix, vector):
	src_length = len(vector)
	dst_length = len(matrix)
	if not len(matrix[0]) == src_length:
		raise Exception("dimensions do not match")


	result = [0]*dst_length
	for j in range(dst_length):
		for i in range(src_length):
			result[j] += matrix[j][i] * vector[i]

	return result

def plus(vector1, vector2):
	if len(vector1) != len(vector2):
		raise Exception("dimensions do not match")
	return [vector1[i] + vector2[i] for i in range(len(vector1))]

def times(vector, factor):
	return [x*factor for x in vector]

def vectorize(fn):
	def inner(vec):
		return [fn(x) for x in vec]
	return inner

def vectostr(vec):
	return "\t".join(map(str,vec))

def vector_len(vec):
	return math.sqrt(vector_len2(vec))

def vector_len2(vec):
	return sum([x * x for x in vec])
