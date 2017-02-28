#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from library.mc3 import memory_capacity as mc3
from library.mc4 import memory_capacity as mc4
from mcbetter import memory_capacity as mcb
from mctheano import memory_capacity as mct

from numpy import random

import timeit

sigma = 0.095
tau = 0.1

q = 100

W = random.normal(0, sigma, [q, q])
WI = random.uniform(-tau, tau, [q, 1])


def run_tests():
	NUMBER = 1

	t1 = timeit.timeit('mc3(W,WI, runs=1)', setup="from __main__ import mc3, W, WI", number=NUMBER)
	print('mc3:', t1 / NUMBER)

	#t2 = timeit.timeit('mc4(W,WI, runs=1)', setup="from __main__ import mc4, W, WI", number=NUMBER)
	#print('mc4:', t2 / NUMBER)

	t3 = timeit.timeit('mcb(W,WI, runs=1)', setup="from __main__ import mcb, W, WI", number=NUMBER)
	print('mcb:', t3 / NUMBER)

	t4 = timeit.timeit('mct(W,WI, runs=1)', setup="from __main__ import mct, W, WI", number=NUMBER)
	print('mcb:', t4 / NUMBER)

run_tests()

#mcb(W, WI, runs=1) 	