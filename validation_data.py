# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:50:45 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Validation data of kofskey et al. 1972
df = pd.read_csv("validation_data.txt", header = None, sep = ",")

T0   = 295.6
T_st = 288.15
p0   = 13.8e4
p_st = 101352

fac = np.sqrt(T0/T_st)/(p0/p_st)

x1 = df[0].values
y1 = df[1].values/fac
x2 = df[2].values
y2 = df[3].values/fac
x3 = df[4].values
y3 = df[5].values/fac
x4 = df[6].values
y4 = df[7].values/fac


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.plot(x1,y1,'o',label = '110')
    ax.plot(x2,y2,'v',label = '100')
    ax.plot(x3,y3,'s',label = '90')
    ax.plot(x4,y4,'D',label = '70')
    ax.legend(title = 'Percent of design speed')
    ax.set_xlabel("Total-to-static pressure ratio")
    ax.set_ylabel("Mass flow rate")
    ax.set_title("Kofskey et al 1972, one stage")