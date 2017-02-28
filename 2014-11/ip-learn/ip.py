#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ip.py
Created 19.11.2014

Goal: Implement IP learning for various matrices
	  and Compare its effect on memory capacity
"""

from matplotlib import pyplot

from library.mc4 import memory_capacity, get_def_act_params

from numpy import tanh, exp, array, abs, zeros, random, dot, multiply, std, ones
from numpy.linalg import eigvals

q = 100
gridx, gridy = 10, 10

def tanh_ab(x, ab):
	""" Range: (0, 1) """
	q = x.shape[0]
	vys = tanh(ab[:,0].reshape(q,1) * x + ab[:,1].reshape(q,1))
	return vys

def sigmoid_ab(x, ab):
	""" Range: (-1, 1) """
	return 1 / (1 + exp(-ab[:,0] * x + ab[:,1]))

activation_function_params = zeros([q, 2])
for i in range(q):
	activation_function_params[i, 0] = 1
	activation_function_params[i, 1] = 0

eta = 10**-4
mu = 0
sigma = 0.085
c = 0.1


def IP_gauss(X, Y, af_params):
	""" adjusts the parameters so that the activation distribution resembles Gaussian distribution
	Usable for activation ranges (-infinity, +infinity)
	(tanh)
	"""

	q = X.shape[0]
	delta_b = -eta * (-mu / (sigma*sigma) + Y / (sigma*sigma) * (2*sigma*sigma + 1 - Y*Y + mu*Y))
	delta_a = +eta / af_params[:, 0].reshape(q,1) + X * delta_b

	af_params[:, 0] += delta_a.flatten()
	af_params[:, 1] += delta_b.flatten()
	

def IP_laplace(X, Y, af_params):
	""" adjusts the parameters so that the activation distribution resembles Laplace distribution
	Usable for activation ranges (-infinity, +infinity)
	(tanh)
	"""
	q = X.shape[0]
	delta_b = -eta * (2 * Y + (Y * (1 - Y*Y + mu * Y) / (c * abs(Y - mu))))
	delta_a = eta / af_params[:, 0].reshape(q,1) + X * delta_b

	af_params[:, 0] += delta_a.flatten()
	af_params[:, 1] += delta_b.flatten()

def IP_exp(X, Y, af_params):
	""" adjusts the parameters so that the activation distribution resembles exponential distribution
	Usable for activation ranges [0, +infinity)
	(sigmoid)
	"""
	q = X.shape[0]
	delta_b = eta * (1 - (2 + 1 / mu) * Y + Y * Y / mu)
	delta_a = eta / af_params[:, 0].reshape(q,1) + X * delta_b

	af_params[:, 0] += delta_a.flatten()
	af_params[:, 1] += delta_b.flatten()

def IP_unif(X, Y, af_params):
	""" adjusts the parameters so that the activation distribution resembles uniform distribution
	Usable for bounded activation ranges e.g. [-1, 1]
	(tanh?)
	"""
	raise NotImplementedError()
	q = X.shape[0]
	delta_b = 0
	delta_a = eta / af_params[:, 0].reshape(q,1) + X * delta_b

	af_params[:, 0] += delta_a.flatten()
	af_params[:, 1] += delta_b.flatten()

def IP_learn(W, WI, af_params, IP_algorithm, iterations=1000, activation_function=tanh_ab, input_dist=(-1., 1.), af_hist=None):
	global eta
	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)
	u = dist_input() 
	X = zeros([q, 1])
	af_param_hist_a, af_param_hist_b = af_hist
	

	for it in range(iterations):
		if it % 1000 == 0:
			eta = eta / 1.4
		net = dot(W, X) + dot(WI, u[it])
		Y = activation_function(net, af_params)
		IP_algorithm(net, Y, af_params)
		af_param_hist_a[:, it] = af_params[:, 0]
		af_param_hist_b[:, it] = af_params[:, 1]
		X = Y
	print(af_params)


def get_activation_std(W, WI, af_params, iterations=1000, activation_function=tanh_ab, input_dist=(-1., 1.)):
	dist_input = lambda: random.uniform(input_dist[0], input_dist[1], iterations)
	u = dist_input() 
	X = zeros([q, 1])
	S = zeros([q, iterations])


	for it in range(iterations):
		Y = activation_function(dot(W, X) + dot(WI, u[it]), af_params)
		S[:, it] = Y[:, 0]
		X = Y

	return (std(S, axis=1), S)

def main():
	"""
	Today's agenda:
		i)   generate a random matrix
		ii)  compute its MC
		iii) improve act. fun. parameters using some IP learning mechanism
			a) each iteration, parameters are updated
		iv)  compute new MC
		v)   compare
	"""

	# i)
	global sigma

	assert gridx * gridy == q

	WI = random.uniform(-0.1, 0.1, [q, 1])
	W = random.normal(0, 0.1, [q, q])
	s = max(abs(eigvals(W)))
	W = W * (0.95 / s)
	print(max(abs(eigvals(W))))

	af_params = get_def_act_params(q)

	act_fun = tanh_ab
	ip_algorithm = IP_gauss

	# ii)
	print("computing mc before")
	mc, _ = memory_capacity(W, WI, activation_function=act_fun, activation_parameters=af_params, memory_max=q, runs=2)
	std1, S =  get_activation_std(W, WI, af_params, activation_function=act_fun, iterations=1000)
	
	# iii)
	print("adjusting activation function parameters")
	sigma = std1.reshape(q, 1)

	

	#sigma = 0.05 * ones([q, 1])
	print("choosing sigma={}".format(sigma))
	IPLEARN_ITERATIONS = 10000
	af_param_hist_a = zeros([q, IPLEARN_ITERATIONS])
	af_param_hist_b = zeros([q, IPLEARN_ITERATIONS])
	IP_learn(W, WI, af_params, ip_algorithm, activation_function=act_fun, iterations=IPLEARN_ITERATIONS, af_hist=(af_param_hist_a, af_param_hist_b))

	# iv)
	print("computing mc after")
	mc2, _ = memory_capacity(W, WI, activation_function=act_fun, activation_parameters=af_params, memory_max=q, runs=2)
	std2, S2 =  get_activation_std(W, WI, af_params, activation_function=act_fun, iterations=1000)


	std1 = std(std1)
	std2 = std(std2)
	# v)
	print("mc before = {:6.4f}, std before = {:6.4f}".format(sum(mc), std1))
	print("mc after  = {:6.4f}, std after  = {:6.4f}".format(sum(mc2), std2))
	if sum(mc) < sum(mc2):
		print("BETTER!")
	else:
		print("WORSE!")

	pyplot.plot(mc, label="before: mc={:6.4f}".format(sum(mc)))
	pyplot.plot(mc2, label="after: mc={:6.4f}".format(sum(mc2)))
	pyplot.legend()
	pyplot.show()

	BINCNT=10
	def show_histograms():
		print()
		for yplt in range(gridy):
			print("\r {}/{}".format(yplt, gridy), end="")
			for xplt in range(gridx):
				
				indx = yplt*gridx + xplt
				pyplot.subplot(gridy, gridx, indx)
				pyplot.hist(S[indx,:],  bins=BINCNT, normed=True, label="before std={:6.4f}".format(std1))
				pyplot.hist(S2[indx,:], bins=BINCNT, normed=True, label="after std={:6.4f}".format(std2))
		#pyplot.legend()
		pyplot.grid(True)
		pyplot.show()
	#show_histograms()

	for neuron in range(q):
		x = array(range(IPLEARN_ITERATIONS))
		y = af_param_hist_a[neuron, :]
		#print(x.shape, y.shape)
		
		pyplot.plot(x, y)
		
	pyplot.grid(True)
	pyplot.show()


if __name__ == '__main__':
	main()