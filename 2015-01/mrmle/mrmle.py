#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mrmle.py
Created 31.1.2015

Goal: Test if the MC formula diverges for more iterations
"""


from numpy import zeros, sum, random
from library.mc3 import memory_capacity as mc3
from matplotlib import pyplot as plt
import sys

q = 100
p = 1
sigma = 0.095

WI = random.uniform(-0.1, 0.1, [q, p])
W = random.normal(0, sigma, [q, q])

MMAX = 200

# more runs

mcnew = mc3(W,WI, memory_max=200, runs=1000, iterations=2000, iterations_skipped=1000,
 iterations_coef_measure=5000)[0]

plt.plot(range(200), mcnew)
plt.show()