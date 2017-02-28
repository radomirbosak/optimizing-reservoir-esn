#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ipsingle.py
Created 26.11.2014

Goal: Study behavior of single tanh neuron under IP learning.
"""

from scipy.stats import norm
from scipy import random, tanh, zeros, std

from matplotlib import pyplot as plt

def newtanh(x, a, b):
	return tanh(a * x + b)

a = 1
b = 0
eta = 10**-4
sigma = 0.1
mu = 0

SAMPLE_COUNT = 100000

INPUT_SIGMA = .10

x = random.normal(0, INPUT_SIGMA, [SAMPLE_COUNT])
y = newtanh(x, a, b)
std1 = std(y)
#print(y)

plt.xlim([-1, 1])
plt.hist(y, bins=10, normed=True, label="bf; std={}".format(std1))


def ip_gauss(x, y):
	global a, b
	db = -eta * ( - mu/(sigma*sigma) + y / (sigma*sigma) * (2*sigma*sigma + 1 - y*y + mu*y))
	a += eta / a + x * db
	b += db

s = zeros([SAMPLE_COUNT])
for it in range(SAMPLE_COUNT):
	x = random.normal(0, INPUT_SIGMA)
	y = newtanh(x, a, b)
	s[it] = y
	ip_gauss(x, y)

print (a, b)
std2 = std(s)

plt.hist(s, bins=10, normed=True, label="af; std={}".format(std2))
plt.legend()
plt.show()

#\begin{align}
#\Delta a &= \frac{\eta}{a} + x\Delta b \\
#\Delta b &= -\eta \left( - \frac{\mu}{\sigma^2} + \frac{y}{\sigma^2} (2\sigma^2 + 1 - y^2 + \mu y) \right)
#\end{align}

