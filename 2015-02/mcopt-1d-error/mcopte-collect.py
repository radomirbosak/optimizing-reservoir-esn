import numpy as np
from matplotlib import pyplot as plt

from library.aux import try_save_fig

taus = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 10**-7, 10**-8]

SAVED_DIR = 'saved-h1/'

sigmas = np.loadtxt(SAVED_DIR + 'sigmas.txt')

maxinlineX = np.zeros(8)
maxinlineY = np.zeros(8)

for line in range(8):
	Y = np.loadtxt(SAVED_DIR + 'avg-t'+str(line)+'.txt')
	Yerr = np.loadtxt(SAVED_DIR + 'std-t'+str(line)+'.txt')

	maxinlineX[line] = sigmas[np.argmax(Y)]
	maxinlineY[line] = np.max(Y)

	plt.errorbar(sigmas, Y, yerr=Yerr, label=("$\\tau={0}$".format(taus[line])))


plt.plot(maxinlineX, maxinlineY, label="maxima")

plt.grid(True)
plt.xlabel("sigma: $W = N(0, \\sigma)$")
plt.ylabel("MC, errbar: $1 \\times \\sigma$")
plt.legend(loc=3)

try_save_fig()

plt.show()