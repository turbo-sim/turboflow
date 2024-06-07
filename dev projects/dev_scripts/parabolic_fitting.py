# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:12:55 2023

@author: laboan
"""

import numpy as np
import matplotlib.pyplot as plt

x1 = 0.81
x2 = 0.97
y1 = -61.6
y2 = -62.3

a = (y1 - y2) / (x1**2 - 2 * x1 * x2 + x2**2)
b = -2 * a * x2
c = y2 + a * x2**2

y = lambda x: a * x**2 + b * x + c

x = np.linspace(0.8, 1, 20)

fig, ax = plt.subplots()
ax.plot(x, y(x))
ax.scatter([x1, x2], [y1, y2])
