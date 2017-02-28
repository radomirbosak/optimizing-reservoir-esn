
import numpy as np
import matplotlib.pyplot as plt

from library.mc3 import memory_capacity
from library.aux import try_save_fig

#sigmas = [0.8, 0.85, 0.9, 0.95]
#sigmas = np.linspace(0.06, .15, 20)
sigmas = np.linspace(0.08, 0.11, 7)
tau = 0.01
reservoir_size = 100

ITERATIONS = 100

line = []
lineerr = []

for sigma in sigmas:
	mcs = np.zeros(ITERATIONS)
	for i in range(ITERATIONS):
		W = np.random.normal(0, sigma, [reservoir_size, reservoir_size])
		WI = np.random.uniform(-tau, tau, [reservoir_size, 1])

		mc = np.sum(memory_capacity(W, WI, memory_max=reservoir_size, runs=1)[0])
		mcs[i] = mc

	print("sigma=%f" % sigma)
	line.append(np.average(mcs))
	lineerr.append(np.std(mcs))

plt.grid()
plt.xlim([0.075,0.115])
plt.errorbar(sigmas, line, yerr=lineerr)
try_save_fig()
plt.show()