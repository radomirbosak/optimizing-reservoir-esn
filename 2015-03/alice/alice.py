#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from library.esn import ESN
from matplotlib import pyplot as plt

# load alice
filename = 'pg11.txt'
alicetext = open(filename, "r").read()
alicetext = alicetext.replace("\n", " ")
alicetext = alicetext.lower()
for ch in [",", "'", ".", "-", "!", "*", ";", ":", "?", "(", ")", '"', "0", "1", "3"]:
	alicetext = alicetext.replace(ch, " ")
alicetext = " ".join(alicetext.split())

c = Counter(alicetext)

symbols = list(c.keys())
count_symbols = len(symbols)

indexer = {symbol: i for i, symbol in enumerate(symbols)}

expanded_dimension = 30

vectors = np.random.normal(0, 1, [expanded_dimension, count_symbols])


def letter_to_vector(letter):
	ret = np.zeros(count_symbols)
	ret.fill(-1)
	ret[indexer[letter]] = 1
	return ret


def vector_to_letter(vector):
	return symbols[np.argmax(vector)]


# net = ESN(count_symbols, 1000, count_symbols)
# alicetext = alicetext[:10000]

net = ESN(count_symbols, 200, count_symbols)

alicetext = alicetext[:10000]

length = len(alicetext) - 1


def generate_training_set():
	return zip(map(letter_to_vector, alicetext[:-1]), map(letter_to_vector, alicetext[1:]))


def train():
	global training_set
	training_set = generate_training_set()
	net.train(training_set, length)


def test():
	global training_set, chyba
	training_set = generate_training_set()
	chyba = net.test(training_set, length)
	print("chyba = {}".format(chyba))


def tt():
	train()
	test()


def parameter_test():
	global taus, fines
	taus = np.linspace(.7, 1.0, 16)
	fines = np.zeros(len(taus))
	fineserr = np.zeros(len(taus))
	net.gen_w(0.95)

	iterations = 30
	zats = np.zeros(iterations)
	for ri, tau in enumerate(taus):

		for it in range(iterations):
			net.gen_wi(tau)
			training_set = generate_training_set()
			net.train(training_set, length)

			training_set = generate_training_set()
			chyba = net.test(training_set, length)
			zats[it] = chyba

		fines[ri] = np.average(zats)
		fineserr[ri] = np.std(zats)
		print("tau = {0:.2f}, chyba = {1:.3f}".format(tau, fines[ri]))

	plt.errorbar(taus, fines, yerr=fineserr)
	plt.show()


prepare_text = alicetext
i = 0


def feed_letter():
	global i
	print("i = {}".format(i))
	symbol = prepare_text[i]
	net.input = letter_to_vector(symbol)
	net.fire()
	generated_letter = vector_to_letter(net.output)
	i += 1
	return generated_letter, prepare_text[i + 1]
	

def more():
	celi = list(zip(*[feed_letter() for _ in range(10)]))
	return ''.join(celi[0]), ''.join(celi[1])


def only_gen():
	global i
	text = ""
	for _ in range(100):
		symbol = prepare_text[i]
		net.input = letter_to_vector(symbol)
		net.fire()
		generated_letter = vector_to_letter(net.output)
		i +=1 
		text += generated_letter
	print(text)


def only_gen2():
	text = ""
	for _ in range(100):
		net.fire()
		generated_letter = vector_to_letter(net.output)
		net.input = letter_to_vector(generated_letter)
		text += generated_letter
	print(text)


gamelen = 10


def game():
	textin = input("> ").lower()
	for l in textin:
		net.input = letter_to_vector(l)
		net.fire()

	textout = ""
	generated_letter = ""
	times = 0
	while times < gamelen:
		generated_letter = vector_to_letter(net.output)
		net.input = letter_to_vector(generated_letter)
		net.fire()
		textout += generated_letter
		if generated_letter in [" ", ".", "!", "?"]:
			times += 1
	print(textout)


def gameinf():
	try:
		while True:
			game()
	except KeyboardInterrupt:
		pass