
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turboflow as tf

show_figures = True
save_figs = False

# Load performance data
filename_1 = "output/performance_map_critical_mach.xlsx" # Critical mach
filename_2 = "output/performance_map.xlsx" # Critical mass flow

data_1 = tf.plot_functions.load_data(filename_1)
data_2 = tf.plot_functions.load_data(filename_2)

subsets = ["omega"] + list(1627*np.array([0.7, 1.0]))
colors = plt.get_cmap('magma')(np.linspace(0.2, 0.7, 2))
fig1, ax1 = tf.plot_functions.plot_lines(
    data_1,
    x_key="PR_ts",
    y_keys=["mass_flow_rate"],
    subsets=subsets,
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Mass flow rate [kg/s]",
    colors=colors,
    linestyles=['-']*2,
    filename = 'mass_flow_rate',
    outdir = "figures",
    save_figs=False,
)

fig1, ax1 = tf.plot_functions.plot_lines(
    data_2,
    x_key="PR_ts",
    y_keys=["mass_flow_rate"],
    subsets=subsets,
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Mass flow rate [kg/s]",
    colors=colors,
    linestyles=[":"]*2,
    fig = fig1,
    ax = ax1, 
    filename = 'mass_flow_rate',
    outdir = "figures",
    save_figs=False,
)

fig2, ax2 = tf.plot_functions.plot_lines(
    data_1,
    x_key="PR_ts",
    y_keys=["torque"],
    subsets=subsets,
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Torque [Nm]",
    colors=colors,
    linestyles=['-']*2,
    filename = 'torque',
    outdir = "figures",
    save_figs=False,
)

fig2, ax2 = tf.plot_functions.plot_lines(
    data_2,
    x_key="PR_ts",
    y_keys=["torque"],
    subsets=subsets,
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Torque [Nm]",
    colors=colors,
    linestyles=[":"]*2,
    fig = fig2,
    ax = ax2, 
    filename = 'torque',
    outdir = "figures",
    save_figs=False,
)

# Add experimental points
filename_exp = "experimental_data/experimental_data_Kofskey1972_1stage_interpolated.xlsx"
data_exp = pd.read_excel(filename_exp, sheet_name = ["Sheet1"])

subsets = ["speed_percent"] + list(np.array([70, 100]))
fig1, ax1 = tf.plot_functions.plot_lines(
    data_exp,
    x_key="pressure_ratio_ts",
    y_keys=["mass_flow_rate"],
    subsets=subsets,
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Mass flow rate [kg/s]",
    colors=colors,
    linestyles = ['none']*2,
    markers = ["x"]*2,
    filename = 'mass_flow_rate',
    outdir = "figures",
    fig = fig1,
    ax = ax1,
    save_figs=False,
)

subsets = ["speed_percent"] + list(np.array([70, 100]))
fig2, ax2 = tf.plot_functions.plot_lines(
    data_exp,
    x_key="pressure_ratio_ts",
    y_keys=["torque"],
    subsets=subsets,
    xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    ylabel="Torque [Nm]",
    colors=colors,
    linestyles = ['none']*2,
    markers = ["x"]*2,
    filename = 'torque',
    outdir = "figures",
    fig = fig2,
    ax = ax2,
    save_figs=False,
)

lables = ["$0.7 \cdot \Omega_\mathrm{des}$, Critical mach", "$1.0 \cdot \Omega_\mathrm{des}$, Critical mach", 
            "$0.7 \cdot \Omega_\mathrm{des}$, Critical mass", "$1.0 \cdot \Omega_\mathrm{des}$, Critical mass", 
            "$0.7 \cdot \Omega_\mathrm{des}$, Experimental data", "$1.0 \cdot \Omega_\mathrm{des}$, Experimental data"]
ax1.legend(labels = lables, ncols = 1, loc = 'lower right', fontsize = 13)
ax2.legend(labels = lables, ncols = 1, loc = 'lower right', fontsize = 13)

ax1.set_ylim([2.55, 2.85])
ax2.set_ylim([40, 140])

fontsize = 18
ax1.xaxis.label.set_fontsize(fontsize)
ax1.yaxis.label.set_fontsize(fontsize) 
ax2.xaxis.label.set_fontsize(fontsize)
ax2.yaxis.label.set_fontsize(fontsize) 
ax1.tick_params(labelsize=fontsize)
ax2.tick_params(labelsize=fontsize)

if save_figs:
    filename1 = 'figures/validation_critical_mach_mass_flow'
    filename2 = 'figures/validation_critical_mach_torque'
    tf.plot_functions.savefig_in_formats(fig1, filename1, formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig2, filename2, formats=[".png",".eps"])

if show_figures:
    plt.show()
