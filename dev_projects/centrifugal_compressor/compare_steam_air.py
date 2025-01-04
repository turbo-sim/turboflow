
import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np


save_figs = False
# filename_air = "output/performance_analysis_2024-11-18_15-41-41.xlsx" # Eckardt O
# filename_steam = "output/performance_analysis_2024-11-22_09-55-22.xlsx" # Eckardt O
filename_air = "output/performance_analysis_2024-11-22_12-02-11.xlsx" # Zhang
filename_steam = "output/performance_analysis_2024-11-22_12-33-09.xlsx" # Zhang

# Load performance data
data_air = tf.plot_functions.load_data(filename_air)
data_steam = tf.plot_functions.load_data(filename_steam)


# Define plot settings
color_map = "jet"
outdir = "figures"

# Plot mass flow rate
title = "Mass flow rate"
# subsets = ["omega"] + list(np.array([10000, 12000, 14000, 16000])*2*np.pi/60)
subsets = ["omega"] + list(52000*2*np.pi/60*np.array([0.75,0.9,0.95, 1.0]))
fig1, ax1 = tf.plot_functions.plot_lines(
    data_air,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency [%]",
    outdir=outdir,
    color_map=color_map,
    save_figs=save_figs,
)

fig1, ax1 = tf.plot_functions.plot_lines(
    data_steam,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    fig = fig1,
    ax = ax1,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency [%]",
    linestyles=["--"]*4,
    outdir=outdir,
    color_map=color_map,
    save_figs=save_figs,
)

fig2, ax2 = tf.plot_functions.plot_lines(
    data_air,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    outdir=outdir,
    color_map=color_map,
    save_figs=save_figs,
)

fig2, ax2 = tf.plot_functions.plot_lines(
    data_steam,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    fig = fig2,
    ax = ax2,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    linestyles=["--"]*4,
    outdir=outdir,
    color_map=color_map,
    save_figs=save_figs,
)

# ax1.set_ylim([0.0, 4.0])

plt.show()