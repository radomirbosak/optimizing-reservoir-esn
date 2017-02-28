#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evestante.py - Even function Echo state network test

* Created: 	14.4.2015
* Goal: 	Check, if an echo state network is capable of simulating 
			an even function, such as x -> x^2
"""

from library.esn import ESN
from library.aux import try_save_fig

from matplotlib import pyplot as plt
import numpy as np

net = ESN(1, 20, 1)

ITERATIONS = 100

inputs = [np.random.uniform(-1, 1, 1) for _ in ITERATIONS] + [np.zeros(1) for _ in range(3)]
inputs2 = [x**2 for x in inputs]
desired = inputs2[-3:] + inputs2[:-3]

training_set = list(zip(inputs, desired))

net.train(training_set, ITERATIONS)

#inputs = [np.array()]
plotinputs = [x[0] for x in inputs]
plotdesired = [x[0] for x in desired]
outputs = np.zeros_like(inputs)

for ix, (x, x2) in enumerate(training_set):
	net.input = x
	net.fire()
	#print(net.output.shape)
	#raise Exception("wow")
	outputs[ix] = net.output[0]

plt.plot(plotinputs, inputs**2, 'o', label="desired")
plt.plot(plotinputs, outputs, 'o', label="output")
plt.legend()
plt.grid(True)

try_save_fig()
try_save_fig(ext="pdf")

plt.show()