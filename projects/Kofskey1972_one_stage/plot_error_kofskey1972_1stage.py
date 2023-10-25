# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:10:28 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load validation file
Case = 8
folder = 'results/' + f'Case {Case}/'
filename = 'validation.xlsx'
filename = folder+filename
data = pd.read_excel(filename, sheet_name = ['Mass flow rate', 'Torque', 'Efficiency', 'alpha'])
mass_flow = data["Mass flow rate"]
torque = data["Torque"]
eta = data["Efficiency"]
alpha = data["alpha"] 

omega = np.sort(mass_flow["omega"].unique())

colormap = 'Blues'
colors = plt.get_cmap(colormap)(np.linspace(0.4, 1, len(omega))) 

fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))


for i in range(len(omega)):
    ax1.scatter(mass_flow[mass_flow["omega"] == omega[i]]["Experiments"], mass_flow[mass_flow["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])))
    ax2.scatter(torque[torque["omega"] == omega[i]]["Experiments"], torque[torque["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])))
    ax3.scatter(eta[eta["omega"] == omega[i]]["Experiments"], eta[eta["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])))
    ax4.scatter(alpha[alpha["omega"] == omega[i]]["Experiments"], alpha[alpha["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])))


error = 0.05
N = 10
ax1.set_xlim([2.55,2.8])
ax1.set_ylim([2.41, 2.9])
xlims = ax1.get_xlim()
ax1.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax1.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax1.legend(loc = "lower right", title = "Percent of design \n angular speed")
fig1.tight_layout(pad=1, w_pad=None, h_pad=None)
ax1.set_ylabel('Model value')
ax1.set_xlabel('Measured value')

ax2.set_xlim([50, 160])
ax2.set_ylim([50, 160])
xlims = ax2.get_xlim()
ax2.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax2.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax2.legend(loc = "lower right", title = "Percent of design \n angular speed")
fig2.tight_layout(pad=1, w_pad=None, h_pad=None)
ax2.set_ylabel('Model value')
ax2.set_xlabel('Measured value')

ax3.set_xlim([21, 90])
ax3.set_ylim([21, 90])
xlims = ax3.get_xlim()
ax3.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax3.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax3.legend(loc = "lower right", title = "Percent of design \n angular speed")
fig3.tight_layout(pad=1, w_pad=None, h_pad=None)
ax3.set_ylabel('Model value')
ax3.set_xlabel('Measured value')

ax4.set_xlim([-60, 0])
ax4.set_ylim([-60, 0])
xlims = ax4.get_xlim()
ax4.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax4.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax4.legend(loc = "lower right", title = "Percent of design \n angular speed")
fig4.tight_layout(pad=1, w_pad=None, h_pad=None)
ax4.set_ylabel('Model value')
ax4.set_xlabel('Measured value')


fig1.savefig(folder + "Error_mass_flow_rate.png", bbox_inches = 'tight')
fig2.savefig(folder + "Error_torque.png", bbox_inches = 'tight')
fig3.savefig(folder + "Error_total-to-static_efficiency.png", bbox_inches = 'tight')
fig4.savefig(folder + "Error_alpha_exit.png", bbox_inches = 'tight')


