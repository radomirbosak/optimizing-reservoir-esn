


import numpy as np
from library.aux import try_save_fig

from time import time
import itertools
import pickle
from matplotlib import pyplot as plt

import os.path
import os
import sys

"""
input:
- savedir
- rv()

- xticks
- lineticks
- xlabel
- ylabel
- linelabels 
"""
#import ipdb

def create_dir(savedir):
	""" Creates savedir, if it does not exist """
	if not os.path.isdir(savedir):
		print("dir {} does not exist, creating".format(savedir))
		os.mkdir(savedir)


def get_old_data_sizes(xticks, lineticks, savedir):
	total_values = get_total_values(savedir, xticks, lineticks)
	return [[len(total_values[li][si]) 
		for si, _ in enumerate(xticks)] 
			for li, _ in enumerate(lineticks)]

def get_max_std(values, old_data_sizes, xticks, lineticks):
	#ipdb.set_trace()
	xtickstds = np.zeros([len(lineticks), len(xticks)])
	for li, lineval in enumerate(lineticks):
		for xi, xval in enumerate(xticks):
			xtickstds[li][xi] = np.std(values[li][xi]) / np.sqrt(len(values[li][xi]) + old_data_sizes[li][xi])
	return np.max(xtickstds)

def compute(random_variable, savedir, xticks,
	lineticks, basic_data=""):

	create_dir(savedir)	
	old_data_sizes = get_old_data_sizes(xticks, lineticks, savedir)

	t = start = time()

	values = [[list() for _ in xticks] for _ in lineticks]
	try:
		for it in itertools.count():
			for li, lineval in enumerate(lineticks):
				for xi, xval in enumerate(xticks):
					mc = random_variable(xval, lineval)
					values[li][xi].append(mc)

			print("counted to", it)

			if time() > t + 10: # every 10 seconds we print max sample average std
				print("max standard error of MC sample average:", get_max_std(values, old_data_sizes, xticks, lineticks))
				t = time()
	except KeyboardInterrupt:
		pass
	save_values(values, savedir)
	save_basic_data(basic_data, savedir)

	print("total:", round(time() - start), 'seconds')


def save_basic_data(string, savedir):
	fullfile = os.path.join(savedir, "basic-data.txt")
	if not os.path.exists(fullfile):
		with open(fullfile, "w") as f:
			f.write(string)

def save_values(values, savedir):
	i, fname, ext = 0, 'data', 'pickle'

	fullfile = os.path.join(savedir, fname) + str(i) + "." + ext
	while os.path.exists(fullfile):
		i += 1
		fullfile = os.path.join(savedir, fname) + str(i) + "." + ext

	with open(fullfile, "wb") as f:
		pickle.dump(values, f)

	

def get_total_values(savedir, xticks, lineticks):
	total_values = [[list() for _ in xticks] for _ in lineticks]
	for file in os.listdir(savedir):
		if file.endswith(".pickle"):
			fullfile = os.path.join(savedir, file)
			newvalues = pickle.load(open(fullfile, "rb"))
			for li, _ in enumerate(lineticks):
				for si, _ in enumerate(xticks):
					total_values[li][si].extend(newvalues[li][si])
	return total_values


def compute_lines(values, xticks, lineticks):
	global Y, Yerr
	Y = [np.zeros(len(xticks)) for _ in lineticks]
	Yerr = [np.zeros(len(xticks)) for _ in lineticks]
	for li, _ in enumerate(lineticks):
		for si, _ in enumerate(xticks):
			Y[li][si] = np.average(values[li][si])
			Yerr[li][si] = np.std(values[li][si])
	return Y, Yerr


def draw(savedir, xticks, lineticks, xlabel="", ylabel="", linelabels=[], save=False, loc=1):
	total_values = get_total_values(savedir, xticks, lineticks)
	Y, Yerr = compute_lines(total_values, xticks, lineticks)
	plot(Y, Yerr, savedir, xticks, lineticks, xlabel=xlabel, ylabel=ylabel, linelabels=linelabels, save=save, loc=loc)
	

def plot(Y, Yerr, savedir, xticks, lineticks, xlabel="", ylabel="", linelabels=[], loc=1, save=False):
	xmax = np.zeros(len(lineticks))
	ymax = np.zeros(len(lineticks))
	for li, _ in enumerate(lineticks):
		ymax[li] = np.max(Y[li])
		xmax[li] = xticks[np.argmax(Y[li])]

	if not linelabels:
		linelabels = ["" for _ in lineticks]

	for li, _ in enumerate(lineticks):
		plt.errorbar(xticks, Y[li], yerr=Yerr[li], label=linelabels[li])
		#plt.text(xmax[li], ymax[li] + 3, "({:.3f}, {:.3f})".format(xmax[li], ymax[li]), fontsize=18, ha='center')

	plt.xlabel(xlabel, fontsize=24, labelpad=-3)
	plt.ylabel(ylabel, fontsize=24)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.grid(True)
	plt.legend(loc=loc)
	
	if save:
		figname = os.path.join(savedir, 'figure')
		try_save_fig(fname=figname)
		try_save_fig(fname=figname, ext="pdf")

	plt.show()


def main():
	print("I am but a simple library.")