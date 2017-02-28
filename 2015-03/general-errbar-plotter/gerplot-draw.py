
"""
gerplot-draw.py - General errorbar plotter - draws computed plots
Created 9.3.2015

Goal: Demonstrate usage of general errorbar plotter
"""
from matplotlib import pyplot as plt
import numpy as np
import itertools
import random

from library.aux import try_save_fig
from library.mc3 import memory_capacity
from library.gerplot import gep_parallel_obtain_results, gep_parallel_plot_lines


sigmas = np.exp(np.linspace(np.log(0.02), np.log(100), 100))
sparsities = [0.9, 0.99, 0.999, 0.9999] # np.linspace(0, 0.90, 10)
ITERATIONS = 100 # bolo tu aj 10000
q = 100
tau = 0.01
savedir = "saved3"


X, Y, Yerr = gep_parallel_obtain_results(sigmas, sparsities, savedir=savedir)
plt.xscale('log')
gep_parallel_plot_lines(X, Y, Yerr, sparsities, xlabel="$\sigma$", ylabel="MC", loc=1, linelabel="sp. = {lineval}")