#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mc3 import memory_capacity
from numpy import random

def main():
	WI = random.uniform(-0.1, 0.1, [100, 1])
	W = random.normal(0, 0.1, [100,100])
	mc, err = memory_capacity(W, WI)

	from matplotlib import pyplot as plt
	plt.errorbar(range(len(mc)), mc, yerr=(err*3))
	plt.ylim([0, 1])
	plt.show()

if __name__ == '__main__':
	main()