#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from numpy import linspace, random
from scipy.stats import norm, shapiro
import matplotlib.pyplot as plt

print("range")
print(shapiro(list(range(5000))))

x = linspace(-1, 1, 100)
y = norm.pdf(x, loc=0, scale=.14)

print()
print("normal")
print(shapiro(y))

x = linspace(-1, 1, 100)
y = norm.pdf(x, loc=0, scale=.12)

print()
print("normal2")
print(shapiro(y))

y = random.normal(0, 0.1, [5000])


print()
print("normal3")
print(shapiro(y))

y = norm.pdf(x, loc=0, scale=0.1)
plt.plot(x,y)
plt.xlim([-1, 1])
plt.show()

#x = [shapiro(random.normal(0, 1, [1000]))[1] for i in range(10000)]





#plt.hist(x, normed=True)
#plt.show()
