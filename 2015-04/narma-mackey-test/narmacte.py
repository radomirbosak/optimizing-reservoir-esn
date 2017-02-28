#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from library.esn2 import ESN2

input_int = (-0.8, 0.8)

def generate_narma(x, length):
	y = np.zeros(length)
	#x = np.random.uniform(input_int[0], input_int[1], length)
	for t in range(29, length - 1):
		mysum = np.sum(y[t - 29 : t + 1])
		y[t + 1] = 0.2 * y[t] + 0.004 * y[t] * mysum + 1.5 * x[t - 29] + 0.001

	return y


def nrmse(yout, ydes):
	upper = np.average((yout - ydes) ** 2)
	lower = np.average((ydes - ydes.mean()) ** 2)

	return np.sqrt(upper / lower)

length = 10000
x = np.random.uniform(input_int[0], input_int[1], length)
ydes = generate_narma(x, length)

tau = 0.1
sigma = 0.05
q = 100

# now create a net and feed it the sequence
net = ESN2(1, q, 1)

training_set = zip(x, ydes)
net.WI = np.random.uniform(-tau, tau, [q, 1])
net.W = np.random.normal(0, sigma, [q, q])

def get_yout(net, x):
	yout = np.zeros(length)
	for it, inp in enumerate(x):
		net.input = np.array([inp])
		net.fire()
		yout[it] = net.output[0]
	return yout
