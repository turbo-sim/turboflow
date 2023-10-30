# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:25:59 2023

@author: laboan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

name_aungier = 'Performance_data_2023-10-30_10-05-19.xlsx'
name_metal = 'Performance_data_2023-10-30_10-03-22.xlsx'

data_aungier = pd.read_excel(name_aungier, sheet_name=['overall', 'cascade', 'plane'])
data_metal = pd.read_excel(name_metal, sheet_name=['overall', 'cascade', 'plane'])
pr = data_aungier["overall"]["pr_ts"]

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["overall"]["m"], label = 'Aungier')
ax.plot(pr, data_metal["overall"]["m"], label = 'metal')
ax.set_ylabel('Mass flow rate [kg/s]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["plane"]["d_6"], label = 'Aungier')
ax.plot(pr, data_metal["plane"]["d_6"], label = 'metal')
ax.set_ylabel('Density [kg/m^3]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["plane"]["w_6"], label = 'Aungier')
ax.plot(pr, data_metal["plane"]["w_6"], label = 'metal')
ax.set_ylabel('Velocity [m/s]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["plane"]["beta_6"], label = 'Aungier')
ax.plot(pr, data_metal["plane"]["beta_6"], label = 'metal')
ax.set_ylabel('Exit relative flow angle [rad]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["cascade"]["d_crit_2"], label = 'Aungier')
ax.plot(pr, data_metal["cascade"]["d_crit_2"], label = 'metal')
ax.set_ylabel('Exit relative flow angle [rad]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["cascade"]["w_crit_2"], label = 'Aungier')
ax.plot(pr, data_metal["cascade"]["w_crit_2"], label = 'metal')
ax.set_ylabel('Exit relative flow angle [rad]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["cascade"]["m_crit_2"], label = 'Aungier')
ax.plot(pr, data_metal["cascade"]["m_crit_2"], label = 'metal')
ax.set_ylabel('Exit relative flow angle [rad]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()

fig, ax = plt.subplots()
ax.plot(pr, data_aungier["plane"]["alpha_4"], label = 'Aungier')
ax.plot(pr, data_metal["plane"]["alpha_4"], label = 'metal')
ax.set_ylabel('Exit relative flow angle [rad]')
ax.set_xlabel('Total-to-static pressure ratio')
ax.legend()