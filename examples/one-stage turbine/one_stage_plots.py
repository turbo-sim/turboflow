# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import turboflow as tf

RESULTS_PATH = "output"

# Get the name of the latest results file
filename = tf.utils.find_latest_results_file(RESULTS_PATH)

save_figs = False
show_figures = True

# Load performance data
data = tf.plot_functions.load_data(filename)

fig1, ax1 = tf.plot_functions.plot_lines(
    data,
    x_key="PR_ts",
    y_keys=["mass_flow_rate"],
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Mass flow rate [kg/s]",
    colors='k',
    filename = 'mass_flow_rate',
    outdir = "figures",
    save_figs=True,
)

# Print design speed line
labels = ["Stator inlet", "Stator exit", "Rotor inlet", "Rotor exit"]
fig1, ax1 = tf.plot_functions.plot_lines(
    data,
    x_key="PR_ts",
    y_keys=["p_1", "p_2", "p_3", "p_4"],
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Static pressure [Pa]",
    linestyles=["-", ":", "--", "-."],
    color_map='Reds',
    labels = labels,
    filename='static_pressure',
    outdir = "figures",
    save_figs=True,
)

# Print stacked losses
labels = ["Profile losses", "Tip clearance losses", "Secondary flow losses", "Trailing edge losses", "Incidence losses"]
fig1, ax1 = tf.plot_functions.plot_lines(
    data,
    x_key="PR_ts",
    y_keys=[
        "loss_profile_4",
        "loss_clearance_4",
        "loss_secondary_4",
        "loss_trailing_4",
        "loss_incidence_4",
    ],
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Loss coefficient [-]", 
    color_map='Reds',
    labels = labels,
    stack=True,
    filename="loss_coefficients",
    outdir="figures",
    save_figs = True,
)


if show_figures:
    plt.show()
