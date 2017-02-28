flemcetest.py
=============

Different definition effect on MC
---------------------------------

Testing different MC definitions on 20 neuron reservoir. Simulation parameters:

	q = 20
	sigma = 0.10
	tau = 0.1

	W = sp.random.normal(0, sigma, [q, q])
	WI = sp.random.uniform(-tau, tau, q)
	mc = memory_capacity(W, WI, memory_max=2*q, 
		iterations_coef_measure=100000, iterations=10000, 
		use_input=use_input, target_later=target_later)

![Boedecker's definition](difdef/figure0.png)

![My definition](difdef/figure1.png)

![Jaeger's definintion](difdef/figure2.png)

![The fourth definition](difdef/figure3.png)


Increasing minimal singular value
---------------------------------

I am investigating, what effect on MC has the minimal singular value. To doing this, a random reservoir matrix was generated, and them the singular values were stretched (through affine transform) so that the minimal singular value had prescribed value.

The optimal $s_{min}$ seems to be from $0.14$ to $0.20$.

![](figure0.png)

![](figure1.png)

![](figure2.png)