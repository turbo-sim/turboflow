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
    ax1.scatter(mass_flow[mass_flow["omega"] == omega[i]]["Experiments"], mass_flow[mass_flow["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])/100)+' $\omega_{\mathrm{des}}$')
    ax2.scatter(torque[torque["omega"] == omega[i]]["Experiments"], torque[torque["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])/100)+' $\omega_{\mathrm{des}}$')
    ax3.scatter(eta[eta["omega"] == omega[i]]["Experiments"], eta[eta["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])/100)+' $\omega_{\mathrm{des}}$')
    ax4.scatter(alpha[alpha["omega"] == omega[i]]["Experiments"], alpha[alpha["omega"] == omega[i]]["Model"], color = colors[i], label = str(int(omega[i])/100)+' $\omega_{\mathrm{des}}$')


error = 0.05
N = 10
ax1.set_xlim([2.56, 2.85])
ax1.set_ylim([2.56, 2.85])
xlims = ax1.get_xlim()
ax1.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax1.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax1.legend(loc = "lower right")
fig1.tight_layout(pad=1, w_pad=None, h_pad=None)
ax1.set_ylabel('Predicted mass flow rate [kg/s]')
ax1.set_xlabel('Measured mass flow rate [kg/s]')
ax1.set_aspect('equal', adjustable='box')

ax2.set_xlim([50, 160])
ax2.set_ylim([50, 160])
xlims = ax2.get_xlim()
ax2.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax2.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax2.legend(loc = "lower right")
fig2.tight_layout(pad=1, w_pad=None, h_pad=None)
ax2.set_ylabel('Predicted torque [Nm]')
ax2.set_xlabel('Measured torque [Nm]')
ax2.set_aspect('equal', adjustable='box')

ax3.set_xlim([21, 90])
ax3.set_ylim([21, 90])
xlims = ax3.get_xlim()
ax3.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax3.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax3.legend(loc = "lower right")
fig3.tight_layout(pad=1, w_pad=None, h_pad=None)
ax3.set_ylabel('Predicted total-to-static efficiency [%]')
ax3.set_xlabel('Measured total-to-static efficiency [%]')
ax3.set_aspect('equal', adjustable='box')

ax4.set_xlim([-59, 0])
ax4.set_ylim([-59, 0])
xlims = ax4.get_xlim()
ax4.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--', linewidth=1.25)
ax4.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax4.legend(loc = "lower right")
fig4.tight_layout(pad=1, w_pad=None, h_pad=None)
ax4.set_ylabel('Predicted rotor exit absolute flow flow angle [deg]')
ax4.set_xlabel('Measured rotor exit absolute flow flow angle [deg]')
ax4.set_aspect('equal', adjustable='box')


fig1.savefig(folder + "Error_mass_flow_rate.png", bbox_inches = 'tight')
fig2.savefig(folder + "Error_torque.png", bbox_inches = 'tight')
fig3.savefig(folder + "Error_total-to-static_efficiency.png", bbox_inches = 'tight')
fig4.savefig(folder + "Error_alpha_exit.png", bbox_inches = 'tight')


