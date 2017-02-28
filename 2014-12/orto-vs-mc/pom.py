ITERATIONS = 100
mc = zeros(ITERATIONS)
og = zeros(ITERATIONS)

#farby = 
QS = 10

colors = [
	[0, 0, 0],
	[0, 0, 1],
	[0, 1, 0],
	[1, 0, 0],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 0],
]

for qpre in range(QS):
	q =  qpre + 2
	for it in range(ITERATIONS):
		W = random.normal(0, 0.1, [q, q])
		WI = random.uniform(-.1, .1, [q, 1])
		mc[it] = sum(memory_capacity(W, WI, memory_max=200, runs=1, iterations_coef_measure=5000)[0][:q+2])
		og[it] = matrix_orthogonality(W)
		print(qpre, QS, it, ITERATIONS)
	plt.scatter(og, mc, label=q, c=(colors[qpre % len(colors)]))

plt.xlabel("orthogonality")
plt.ylabel("memory capacity")
plt.grid(True)
plt.legend()
plt.show()