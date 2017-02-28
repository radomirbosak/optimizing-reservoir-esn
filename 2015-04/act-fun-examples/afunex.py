#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

identity = lambda x: x
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: np.tanh(x)
threshold = lambda xs: [0 if x <= 0.5 else 1 for x in xs]

boundary = 4
xrange = np.linspace(-boundary, boundary, 100)

def plotfn(fn, title,):
	plt.plot(xrange, fn(xrange))
	plt.grid(True)
	plt.ylim([-1.2, 1.2])
	#plt.title(title)
	draw()

def plotthresh(title):
	plt.grid(True)
	plt.ylim([-1.2, 1.2])
	#plt.title(title)
	plt.plot([-boundary, 1/2], [0, 0], 'b')
	plt.plot([1/2, boundary], [1, 1], 'b')
	plt.plot([0.5], [1], 'ob')
	plt.plot([0.5], [0], 'ow')
	draw()

def draw():
	save = True
	fig = plt.gcf()
	fig.set_size_inches(8,3)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	if save:
		try_save_fig()
		try_save_fig(ext="pdf")
		plt.clf()
	else:
		plt.show()

plotfn(identity, "identity")
plotfn(sigmoid, "sigmoid")
plotfn(tanh, "tanh")
plotthresh("threshold")
