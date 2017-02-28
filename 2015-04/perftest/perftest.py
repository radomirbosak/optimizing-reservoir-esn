from library.mc6 import memory_capacity
import numpy as np

q = 100
sigma = 0.1
tau = 0.1

W = np.random.normal(0, sigma, [q, q])
WI = np.random.uniform(-tau, tau, q)


for i in range(10):
	mc = memory_capacity(W, WI)

