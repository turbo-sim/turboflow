
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turboflow as tf

show_figures = True
save_figs = True
error_bars = False

# Load performance data
filename_1 = "output/performance_analysis_2024-09-05_11-28-10.xlsx" # Critical mach
filename_2 = "output/performance_analysis_2024-03-13_23-14-55.xlsx" # Critical mass flow

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
filename_exp = "experimental_data_Kofskey1972_1stage_interpolated.xlsx"
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

lables = ["$0.7 \cdot \Omega_\mathrm{des}$, New model", "$1.0 \cdot \Omega_\mathrm{des}$, New model", 
            "$0.7 \cdot \Omega_\mathrm{des}$, Previous model", "$1.0 \cdot \Omega_\mathrm{des}$, Previous model", 
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

if error_bars:

    data_exp_70 = data_exp["Sheet1"][data_exp["Sheet1"]["speed_percent"] == 70]
    data_exp_100 = data_exp["Sheet1"][data_exp["Sheet1"]["speed_percent"] == 100]

    # Percentage error (as a parameter)
    percentage_error_mass_flow_70 = 1.75 # For example, 10%
    percentage_error_mass_flow_100 = 2.0  # For example, 10%
    percentage_error_torque_70 = 6.0  # For example, 10%
    percentage_error_torque_100 = 2.5  # For example, 10%


    # Calculate absolute errors based on the percentage error
    error_mass_flow_70 = (percentage_error_mass_flow_70 / 100) * data_exp_70["mass_flow_rate"]
    error_mass_flow_100 = (percentage_error_mass_flow_100 / 100) * data_exp_100["mass_flow_rate"]
    error_torque_70 = (percentage_error_torque_70 / 100) * data_exp_70["torque"]
    error_torque_100 = (percentage_error_torque_100 / 100) * data_exp_100["torque"]

    # Create the plot with error bars
    ax1.errorbar(data_exp_70["pressure_ratio_ts"], data_exp_70["mass_flow_rate"], yerr=error_mass_flow_70, fmt=' ',color = colors[0], alpha=0.25, capsize=5, label='Data with error bars')
    ax1.errorbar(data_exp_100["pressure_ratio_ts"], data_exp_100["mass_flow_rate"], yerr=error_mass_flow_100, fmt=' ',color = colors[1], alpha=0.25, capsize=5, label='Data with error bars')
    ax2.errorbar(data_exp_70["pressure_ratio_ts"], data_exp_70["torque"], yerr=error_torque_70, fmt=' ',color = colors[0], alpha=0.25, capsize=5, label='Data with error bars')
    ax2.errorbar(data_exp_100["pressure_ratio_ts"], data_exp_100["torque"], yerr=error_torque_100, fmt=' ',color = colors[1], alpha=0.25, capsize=5, label='Data with error bars')


if save_figs:
    filename1 = 'validation_mass_flow_kofskey_1972'
    filename2 = 'validation_torque_kofskey_1972'
    tf.plot_functions.savefig_in_formats(fig1, filename1, formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig2, filename2, formats=[".png",".eps"])

if show_figures:
    plt.show()
