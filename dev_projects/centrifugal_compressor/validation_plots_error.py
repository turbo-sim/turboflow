
import turboflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

impeller = "A"
save_figs = False

files_exp = {
    "A" : "validation_cases\Eckardt\eckardt_impeller_A\eckardt_impeller_A.xlsx",
    "O" : "validation_cases\Eckardt\eckardt_impeller_O\eckardt_impeller_O.xlsx",
}

files_sim = {
    "A" : {
        "oh" : {
            "PR" : "output\performance_analysis_2024-11-27_12-17-50.xlsx",
            "eta" : "output\performance_analysis_2024-11-27_12-22-24.xlsx"
        },
        "zhang" : {
            "PR" : "output\performance_analysis_2024-11-27_12-19-07.xlsx",
            "eta" : "output\performance_analysis_2024-11-27_12-21-35.xlsx"
        }
    },
    "O" : {
        "oh" : {
            "PR" : "output\performance_analysis_2024-11-27_12-24-47.xlsx",
            "eta" : "output\performance_analysis_2024-11-27_12-24-47.xlsx"
        },
        "zhang" : {
            "PR" : "output\performance_analysis_2024-11-27_12-25-47.xlsx",
            "eta" : "output\performance_analysis_2024-11-27_12-25-47.xlsx" 
        }
    }
}

filename_exp = files_exp[impeller]
filename_sim_oh = files_sim[impeller]["oh"]
filename_sim_zhang = files_sim[impeller]["zhang"]

data_exp = tf.load_data(filename_exp, ["pr_tt", "eta_tt"])
data_sim_PR_oh = tf.load_data(filename_sim_oh["PR"])
data_sim_eta_oh = tf.load_data(filename_sim_oh["eta"])
data_sim_PR_zhang = tf.load_data(filename_sim_zhang["PR"])
data_sim_eta_zhang = tf.load_data(filename_sim_zhang["eta"])

# Calculate error
err_eta_oh = (data_exp["eta_tt"]["eta_tt"].values - data_sim_eta_oh["overall"]["efficiency_tt"].values/100)/data_exp["eta_tt"]["eta_tt"].values*100
err_PR_oh = (data_exp["pr_tt"]["pr_tt"].values - data_sim_PR_oh["overall"]["PR_tt"].values)/data_exp["pr_tt"]["pr_tt"].values*100
err_eta_zhang = (data_exp["eta_tt"]["eta_tt"].values - data_sim_eta_zhang["overall"]["efficiency_tt"].values/100)/data_exp["eta_tt"]["eta_tt"].values*100
err_PR_zhang = (data_exp["pr_tt"]["pr_tt"].values - data_sim_PR_zhang["overall"]["PR_tt"].values)/data_exp["pr_tt"]["pr_tt"].values*100
data_exp["eta_tt"]["error_eta_oh"] = err_eta_oh
data_exp["pr_tt"]["error_pr_oh"] = err_PR_oh
data_exp["eta_tt"]["error_eta_zhang"] = err_eta_zhang
data_exp["pr_tt"]["error_pr_zhang"] = err_PR_zhang

# Eta
omegas = [10e3, 12e3, 14e3, 16e3]
color_map = 'viridis'
colors = [plt.get_cmap(color_map)(i / (len(omegas) - 1)) for i in range(len(omegas))]
markers = ["o", "^", "s", "*"]
subsets = ["rpm"] + omegas
fig1, ax1 = tf.plot_functions.plot_lines(
    data_exp,
    x_key="mass_flow_rate",
    y_keys=["error_eta_oh"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Efficiency error [%]",
    colors=colors,
    linestyles=["none"]*4,
    markers=["^"]*4,
)


fig1, ax1 = tf.plot_functions.plot_lines(
    data_exp,
    x_key="mass_flow_rate",
    y_keys=["error_eta_zhang"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Efficiency error [%]",
    colors=colors,
    linestyles=["none"]*4,
    markers=["s"]*4,
    fig = fig1,
    ax = ax1,
)

#Pressure ratio
subsets = ["rpm"] + omegas
fig2, ax2 = tf.plot_functions.plot_lines(
    data_exp,
    x_key="mass_flow_rate",
    y_keys=["error_pr_oh"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Pressure ratio error [%]",
    colors=colors,
    linestyles=["none"]*4,
    markers=["^"]*4,
)

fig2, ax2 = tf.plot_functions.plot_lines(
    data_exp,
    x_key="mass_flow_rate",
    y_keys=["error_pr_zhang"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Pressure ratio error [%]",
    colors=colors,
    linestyles=["none"]*4,
    markers=["s"]*4,
    fig = fig2,
    ax = ax2,
)

marker_handles = [
    Line2D([0], [0], marker = '^', color = 'w', markerfacecolor=colors[0], markersize=10, label = "Oh"),
    Line2D([0], [0], marker = 's', color = 'w', markerfacecolor=colors[0], markersize=10, label = "Zhang"),
]
color_handles = [
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[0], markersize=10, label = omegas[0]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[1], markersize=10, label = omegas[1]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[2], markersize=10, label = omegas[2]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[3], markersize=10, label = omegas[3]/1e3),
]
color_legend1 = ax1.legend(handles = color_handles)
color_legend2 = ax2.legend(handles = color_handles)
ax1.add_artist(color_legend1)
ax2.add_artist(color_legend2)
marker_legend1 = ax1.legend(handles=marker_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
marker_legend2 = ax2.legend(handles=marker_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
fig1.subplots_adjust(top=0.9)
fig2.subplots_adjust(top=0.9)

if save_figs:
    tf.plot_functions.savefig_in_formats(fig2, f"figures/error_impeller{impeller}_PR_tt", formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig1, f"figures/error_impeller{impeller}_eta_tt", formats=[".png",".eps"])
plt.show()
