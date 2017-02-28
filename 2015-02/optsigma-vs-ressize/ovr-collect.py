#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from library.aux import try_save_fig

#taus = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 10**-7, 10**-8]

SAVED_DIR = 'saved/'

sigmas = np.loadtxt(SAVED_DIR + 'sigmas.txt')
res_sizes = np.loadtxt(SAVED_DIR + 'ressizes.txt')

LINES = 6

maxinlineX = np.zeros(LINES)
maxinlineY = np.zeros(LINES)

for line in range(LINES):
	q = res_sizes[line]
	Y = np.loadtxt(SAVED_DIR + 'avg-t'+str(line)+'.txt')
	Yerr = np.loadtxt(SAVED_DIR + 'std-t'+str(line)+'.txt')

	maxinlineX[line] = sigmas[np.argmax(Y)]
	maxinlineY[line] = np.max(Y)
	print("maximum #{} (q={}): ({:5.3f}, {:5.3f}), sigma^2 * q = {:5.3f}" \
		.format(line, q,
			maxinlineX[line], maxinlineY[line],
			maxinlineX[line]**2 * q))

	plt.errorbar(sigmas, Y, yerr=Yerr, label=("res. size $={0}$".format(res_sizes[line])))


plt.plot(maxinlineX, maxinlineY, label="maxima")

plt.grid(True)
plt.xlabel("sigma: $W = N(0, \\sigma)$")
plt.ylabel("MC, errbar: $1 \\times \\sigma$")
plt.legend(loc=1)
plt.xlim([0.05, 0.25])

try_save_fig()

plt.show()