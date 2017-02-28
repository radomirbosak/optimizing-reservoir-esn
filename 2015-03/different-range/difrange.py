#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from library.lexp import ljapunov_exponent
from library.mc5 import memory_capacity
from library.aux import try_save_fig

import numpy as np
from matplotlib import pyplot as plt

dists = [
	((-1.,  4.), 'k'),
	(( 0.,  5.), 'g'),
	(( 2.,  7.), 'r'),
	(( 5., 10.), 'b'),
]

ITERATIONS = 50
tau = 0.1
q = 150

logsigmas = [-1.5, -1.4, -1.3] + list(np.linspace(-1.2, -0.9, 16)) + [-0.8, -0.7, -0.6, -0.5]

total = len(logsigmas) * ITERATIONS

lines = [None for _ in range(len(dists))]

for li, dist in enumerate(dists):
	X = np.zeros(total)
	Y = np.zeros(total)
	for si, logsigma in enumerate(	logsigmas):
		sigma = np.power(10, logsigma)
		for it in range(ITERATIONS):
			W = np.random.normal(0, sigma, [q, q])
			WI = np.random.uniform(-tau, tau, [q, 1])

			mc = memory_capacity(W, WI, input_dist=dist[0], cc_correction=False)
			le = ljapunov_exponent(W, WI, iterations=100, input_dist=dist[0])

			X[it + ITERATIONS*si] = le
			Y[it + ITERATIONS*si] = mc
		print(li, 'of', len(dists), ';', si, 'of', len(logsigmas))

	lines[li] = [X, Y]


def r():
	for li, dist in enumerate(dists):
		interval = "input $\in [{}, {}]$".format(*dists[li][0])
		plt.scatter(lines[li][0], lines[li][1],marker='+', c=dists[li][1], label=interval)

	plt.legend(loc=3)
	plt.grid(True)
	plt.xlabel("LE")
	plt.ylabel("MC")
	plt.xlim([-1.1, 0.5])
	plt.ylim([0, 30])
	try_save_fig(ext="png")
	try_save_fig(ext="eps")
	plt.show()

r()