# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:40:03 2024

@author: lasseba
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
import os
import meanline_axial as ml

# Script options
plot_data = True
interpolation = True
check_interpolation = False
plot_mass_flow = True
plot_deviation = False

def get_data(filename):
    file_dir = 'output'
    filepath = os.path.join(file_dir, filename)
    data = pd.read_excel(filepath)
    return data

if interpolation:
    filename = 'mass_flux_curves.xlsx'
    data = get_data(filename)
    Y = data["Loss coefficient"].unique()

    # Extract critical mach
    mach_crit = np.zeros(len(Y))
    phi_max = np.zeros(len(Y))
    for i in range(len(Y)):
        data_subset = data[data["Loss coefficient"] == Y[i]]
        phi_max[i] = data_subset["Mass flux"].max()
        mach_crit[i] = data_subset.loc[data_subset["Mass flux"].idxmax(), "Mach"]
        
    f = interpolate.interp1d(Y, mach_crit, kind = 'cubic', fill_value = 'extrapolate')
    f2 = interpolate.interp1d(mach_crit, phi_max, 'linear', fill_value = 'extrapolate')
    
    if check_interpolation:
        Y_interp = np.linspace(min(Y), 2.0, 10)
        mach_interp = f(Y_interp)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(Y_interp, mach_interp, label = 'Interpolated')
        ax.plot(Y, mach_crit, label = 'Dataset')
        ax.set_xlabel('Loss coefficent [Y]')
        ax.set_ylabel('Critical mach number [Ma]')
        ax.legend()
        plt.show()

if plot_data: 
    filename = 'mass_flux_curves_compact.xlsx'
    data = get_data(filename)
    Y = data["Loss coefficient"].unique()
    phi_max = data["Mass flux"].max()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(Y)))
    linestyles = ['-',':','--','-.']
    for i in range(len(Y)):
        data_subset = data[data["Loss coefficient"] == Y[i]]
        ax.plot(data_subset["Mach"], data_subset["Mass flux"]/phi_max, label = f'Y = {Y[i]:.1f}', color = colors[i], linestyle = linestyles[i])
        ax1.plot(data_subset["Pressure ratio"], data_subset["Mach"], label = f'Y = {Y[i]:.1f}', color = colors[i], linestyle = linestyles[i])
        ax2.plot(data_subset["Pressure ratio"], data_subset["Mass flux"]/phi_max, label = f'Y = {Y[i]:.1f}', color = colors[i], linestyle = linestyles[i])
    if interpolation:
        mach_crit_iterp = np.linspace(0.8, 1.05, 10)
        ax.plot(mach_crit_iterp, f2(mach_crit_iterp)/phi_max, color = 'k', linestyle = ':')
        ax.text(0.82, 1.02, 'Subsonic', fontsize=13, color='k')
        ax.text(1.05, 1.02, 'Supersonic', fontsize=13, color='k')

    ax.legend(title = 'Loss coefficient')
    ax.set_xlim([0.4, 1.6])
    ax.set_ylim([0.44, 1.1])
    ax.set_ylabel('Normalized mass flux [$\phi/\phi_\mathrm{isentropic, max}$]')
    ax.set_xlabel('Mach')

    ax1.legend(title = 'Loss coefficient')
    ax1.set_ylabel('Mach')
    ax1.set_xlabel('Total-to-static pressure ratio [$p_\mathrm{out}/p_{0, \mathrm{in}}$]')

    ax2.legend(title = 'Loss coefficient')
    ax2.set_ylabel('Mass flux [kg/$\mathrm{m}^2\cdot$s]')
    ax2.set_xlabel('Total-to-static pressure ratio [$p_\mathrm{out}/p_{0, \mathrm{in}}$]')

if plot_mass_flow:
    filename = 'mass_flux_curves_compact.xlsx'
    data = get_data(filename)
    Y = data["Loss coefficient"].unique()
    Area = 0.005

    # Find phi_max and phi_Ma = 1
    phi_Ma_1 = np.zeros(len(Y))
    phi_max = np.zeros(len(Y))
    mach_crit = np.zeros(len(Y))
    for i in range(len(Y)):
        data_subset = data[data["Loss coefficient"] == Y[i]]
        phi_max[i] = data_subset["Mass flux"].max()
        mach_crit[i] = data_subset.loc[data_subset["Mass flux"].idxmax(), "Mach"]
        index_first_mach_one_or_above = data_subset.index[data_subset['Mach'] >= 1][-1]
        phi_Ma_1[i] = data_subset.loc[index_first_mach_one_or_above, 'Mass flux']

    # plot mass flow rate for both choking conditions
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(Y)))
    linestyles = ['-',':','--','-.']

    for i in range(len(Y)):
        data_subset = data[data["Loss coefficient"] == Y[i]]
        pressure_ratio = data_subset["Pressure ratio"]
        mass_flow_crit = data_subset[data_subset["Mach"] < mach_crit[i]]["Mass flux"] 
        mass_flow_1 = data_subset[data_subset["Mach"] < 1]["Mass flux"] 
        n = len(pressure_ratio) - len(mass_flow_crit)
        n2 = len(pressure_ratio) - len(mass_flow_1)
        mass_flow_crit = np.concatenate((np.ones(n)*phi_max[i],mass_flow_crit))*Area
        mass_flow_1 = np.concatenate((np.ones(n2)*phi_Ma_1[i], mass_flow_1))*Area
        ax.plot(1/pressure_ratio, mass_flow_crit/phi_max[0]/Area, color = colors[i], linestyle = '-', label = f"Y = {Y[i]:.1f}")
        ax.plot(1/pressure_ratio, mass_flow_1/phi_max[0]/Area, color = colors[i], linestyle = '--')

    ax.set_ylabel('Normalized mass flow rate [$\dot{m}/\dot{m}_\mathrm{isentropic, max}$]')
    ax.set_xlabel('Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]')
    ax.set_xlim([1.0, 3.5])

    x1, x2, y1, y2 = 1.7, 2.4, 0.8675, 0.8975 # subregion of the original image 
    axins = ax.inset_axes(
    [0.23, 0.1, 0.4, 0.4],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.plot(1/pressure_ratio, mass_flow_crit/phi_max[0]/Area, color = colors[-1])
    axins.plot(1/pressure_ratio, mass_flow_1/phi_max[0]/Area, color = colors[-1], linestyle = '--')
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.legend(title = 'Loss coefficient')

if plot_deviation:
    filename = 'mass_flux_curves_compact.xlsx'
    data = get_data(filename)
    Y = data["Loss coefficient"].unique()

    # Define gauging angle
    geometry = {"A_throat" : 0.5,
            "A_out" : 1}
    
    # Find mach_crit
    mach_crit = np.zeros(len(Y))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(Y)))
    linestyles = ['-',':','--','-.']
    for i in range(len(Y)):
        data_subset = data[data["Loss coefficient"] == Y[i]]
        mach = data_subset["Mach"]
        mach_crit[i] = data_subset.loc[data_subset["Mass flux"].idxmax(), "Mach"]
        beta_aungier = ml.deviation_model.get_subsonic_deviation(
        mach, mach_crit[i], geometry, 'aungier'
        )
        ax.plot(mach, beta_aungier, color = colors[i], linestyle = linestyles[i], label = f"Y = {Y[i]:.1f}")

    ax.set_xlim([0.25, 1.0])
    ax.set_xlabel("Mach")
    ax.set_ylabel("Relative flow angle [deg]")
    ax.legend(title = 'Loss coefficient')
    
plt.show()