import numpy as np

class ESN2():
	def __init__(self, input, hidden, output):
		self.p, self.q, self.r = input, hidden, output

		# neuron preparation
		self.input = np.zeros(input)
		self.X = np.zeros(hidden)
		self.output = np.zeros(output)

		# synapse preparation
		self.gen_w(0.95)
		self.gen_wi(0.5)
		self.WO = np.zeros([output, hidden])

	def gen_w(self, radius=0.95):
		self.W = np.random.normal(0, 1, [self.q, self.q])
		self.W = self.W * 0.95 / np.max(np.abs(np.linalg.eigvals(self.W)))

	def gen_wi(self, tau=0.5):
		self.WI = np.random.uniform(-tau, tau, [self.q, self.p])

	def train(self, training_set, length, verbose=False):
		iterations_measured = length#len(training_set)
		p, q, r = self.p, self.q, self.r

		S = np.zeros([q, iterations_measured]) # dlhy obdlznik
		D = np.zeros([r, iterations_measured])

		for it, (input, desired) in enumerate(training_set):
			self.input = np.array([input])
			self.reservoir()
			S[:, it] = self.X
			D[:, it] = desired
			if it % 1000 == 0 and verbose:
				print("\r", it, 'of', iterations_measured, end="")

		if verbose:
			print()
		S_PINV = np.linalg.pinv(S)
		self.WO = np.dot(D, S_PINV)

	# def test(self, testing_set, length, verbose=False):
	# 	iterations_measured = length
	# 	difs = np.zeros(length)
	# 	for it, (input, desired) in enumerate(testing_set):
	# 		self.input = input
	# 		self.fire()
	# 		difs[it] = 0 if np.argmax(self.output) == np.argmax(desired) else 1
	# 		if it % 1000 == 0 and verbose:
	# 			print("\r", it, 'of', iterations_measured, end="")

	# 	if verbose:
	# 		print()
	# 	return np.average(difs)


	def fire(self):
		self.reservoir()
		self.reoutput()

	def reservoir(self):
		self.X = np.tanh(np.dot(self.W, self.X) + np.dot(self.WI, self.input))

	def reoutput(self):
		self.output = np.dot(self.WO, self.X)