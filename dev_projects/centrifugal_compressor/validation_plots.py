
import turboflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

impeller = "A" # A, O
save_figs = True
fontsize = 16
plt.rcParams['font.size'] = fontsize  # Replace 14 with the desired font size
plt.rcParams.update({
    'axes.labelsize': fontsize,    # Font size for x and y labels
    'xtick.labelsize': fontsize,   # Font size for x tick labels
    'ytick.labelsize': fontsize,   # Font size for y tick labels
    'legend.fontsize': fontsize,   # Font size for legends
})

files_exp = {
    "A" : "validation_cases\Eckardt\eckardt_impeller_A\eckardt_impeller_A.xlsx",
    "O" : "validation_cases\Eckardt\eckardt_impeller_O\eckardt_impeller_O.xlsx",
}

files_sim_exl =  {
    "A" : {
        "oh" : "output/performance_analysis_2024-11-18_14-14-09.xlsx",
        "zhang" : "output/performance_analysis_2024-11-18_15-30-05.xlsx",
    },
    "O" : {
        "oh" : "output/performance_analysis_2024-11-07_10-13-37.xlsx",
        "zhang" : "output/performance_analysis_2024-11-19_08-21-33.xlsx",
    }
}

files_sim_incl =  {
    "A" : {
        "oh" : "output/performance_analysis_2024-11-18_15-39-03.xlsx",
        "zhang" : "output/performance_analysis_2024-11-18_13-34-39.xlsx",
    },
    "O" : {
        "oh" : "output/performance_analysis_2024-11-18_15-41-41.xlsx",
        "zhang" : "output/performance_analysis_2024-11-19_08-18-06.xlsx",
    }
}

files_sim_condensed = {
    "A" : {
        "oh" : {
            "PR" : "output\performance_analysis_2024-11-27_12-17-50.xlsx",
            "eta" : "output\performance_analysis_2024-11-27_12-22-24.xlsx"
        },
        "zhang" : {

            "PR" : "output\performance_analysis_2024-12-03_11-06-19.xlsx",
            "eta" : "output\performance_analysis_2024-12-03_11-04-49.xlsx"
        }
    },
    "O" : {
        "oh" : {
            "PR" : "output\performance_analysis_2024-11-27_12-24-47.xlsx",
            "eta" : "output\performance_analysis_2024-11-27_12-24-47.xlsx"
        },
        "zhang" : {
            "PR" : "output\performance_analysis_2024-12-03_11-02-32.xlsx",
            "eta" : "output\performance_analysis_2024-12-03_11-02-32.xlsx" 
        }
    }
}

filename_exp = files_exp[impeller]
filename_oh = files_sim_condensed[impeller]["oh"]["PR"]
filename_zhang = files_sim_condensed[impeller]["zhang"]["PR"]

data_exp = tf.load_data(filename_exp, ["pr_tt", "eta_tt"])
data_oh = tf.load_data(filename_oh)
data_zhang = tf.load_data(filename_zhang)

# Plot simulation data
omegas = [10e3, 12e3, 14e3, 16e3]
color_map = 'magma'
colors = plt.get_cmap(color_map)(np.linspace(0.2, 0.8, len(omegas)))
subsets = ["omega"] + list(2*np.pi/60*np.array(omegas))

# Mass flow rate
fig1, ax1 = tf.plot_functions.plot_lines(
    data_oh,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    colors=colors,
)
fig1, ax1 = tf.plot_functions.plot_lines(
    data_zhang,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    fig = fig1,
    ax = ax1,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    linestyles=["--"]*4,
    colors=colors,
)
# ax1.legend(omegas, loc = 'best')

color_handles = [
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[0], markersize=10, label = omegas[0]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[1], markersize=10, label = omegas[1]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[2], markersize=10, label = omegas[2]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[3], markersize=10, label = omegas[3]/1e3),
]

linestyle_handles = [
    Line2D([0], [0], color=colors[0], linestyle='-', label = "Oh"),
    Line2D([0], [0], color=colors[0], linestyle='--', label = "Zhang"),
]

# ax2.legend(np.array(omegas)/1e3, ncols = 1, loc = 'best')
color_legend = ax1.legend(handles = color_handles)
linestyle_legend = ax1.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
ax1.add_artist(color_legend)
# ax1.add_artist(linestyle_legend)
ax1.set_xlim([2.1, 7.1])
ax1.set_ylim([1.2, 2.6])
plt.tight_layout()


# Efficiency
fig2, ax2 = tf.plot_functions.plot_lines(
    data_oh,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency",
    colors=colors,
)
fig2, ax2 = tf.plot_functions.plot_lines(
    data_zhang,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    fig = fig2,
    ax = ax2,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency [%]",
    linestyles=["--"]*4,
    colors=colors,
)
color_handles = [
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[0], markersize=10, label = omegas[0]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[1], markersize=10, label = omegas[1]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[2], markersize=10, label = omegas[2]/1e3),
    Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor=colors[3], markersize=10, label = omegas[3]/1e3),
]

linestyle_handles = [
    Line2D([0], [0], color=colors[0], linestyle='-', label = "Oh"),
    Line2D([0], [0], color=colors[0], linestyle='--', label = "Zhang"),
]

# ax2.legend(np.array(omegas)/1e3, ncols = 1, loc = 'best')
color_legend = ax2.legend(handles = color_handles, loc = "lower left", bbox_to_anchor=(0.0, -0.05))#loc = "lower center", bbox_to_anchor=(0.3,0))
linestyle_legend = ax2.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
ax2.add_artist(color_legend)
# ax2.add_artist(linestyle_legend)
ax2.set_xlim([2.1, 7.1])
ax2.set_ylim([75, 95])
plt.tight_layout()

# Losses 
subset = ["omega"] + [2*np.pi/60*14e3]
colors_losses = plt.get_cmap(color_map)(np.linspace(0.2, 0.8, 10))
fig3, ax3 = tf.plot_functions.plot_lines(
    data_oh,
    x_key="mass_flow_rate",
    y_keys=["incidence_out",
            "blade_loading_out",
            "skin_friction_out",
            "tip_clearance_out",
            "wake_mixing_out",
            "leakage_out",
            "recirculation_out",
            "disk_friction_out",
            ],
    subsets = subset,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Loss distribution",
    colors=colors_losses,
    stack = True
)
loss_labels = ["Incidence", "Blade Loading", "Skin Friction", "Tip Clearance", "Mixing", 
               "Leakage", "Recirculation", "Disk Friction",
               ]
ax3.legend(loss_labels, loc = 'upper right', ncols = 2)

# Losses 
subset = ["omega"] + [2*np.pi/60*14e3]
fig4, ax4 = tf.plot_functions.plot_lines(
    data_zhang,
    x_key="mass_flow_rate",
    y_keys=["incidence_out",
            "blade_loading_out",
            "skin_friction_out",
            "tip_clearance_out",
            "wake_mixing_out",
            "leakage_out",
            "recirculation_out",
            "disk_friction_out",
            "choke_out",
            "entrance_diffusion_out"
            ],
    subsets = subset,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Loss distribution",
    colors=colors_losses,
    stack = True
)
loss_labels = ["Incidence", "Blade Loading", "Skin Friction", "Tip Clearance", "Mixing", 
               "Leakage", "Recirculation", "Disk Friction",
               "Choke", 
               "Entrance diffusion"
               ]
ax4.legend(loss_labels, loc = 'upper right', ncols = 2)

# Plot experimental data
markers = ["x", "o", "v", "s"]
for i, rpm in zip(range(len(omegas)),omegas):
    data_exp_pr_tt = data_exp["pr_tt"][data_exp["pr_tt"]["rpm"] == rpm]
    data_exp_eta_tt = data_exp["eta_tt"][data_exp["eta_tt"]["rpm"] == rpm]
    ax1.plot(data_exp_pr_tt["mass_flow_rate"], data_exp_pr_tt["pr_tt"], linestyle = "none", color = colors[i], marker = markers[i])
    ax2.plot(data_exp_eta_tt["mass_flow_rate"], data_exp_eta_tt["eta_tt"]*100, linestyle = "none", color = colors[i], marker = markers[i])

plt.show()

if save_figs:
    tf.plot_functions.savefig_in_formats(fig1, f"figures/check_models_impeller{impeller}_PR_tt", formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig2, f"figures/check_models_impeller{impeller}_eta_tt", formats=[".png",".eps"])
    # tf.plot_functions.savefig_in_formats(fig3, f"figures/check_models_impeller{impeller}_losses_oh", formats=[".png",".eps"])
    # tf.plot_functions.savefig_in_formats(fig4, f"figures/check_models_impeller{impeller}_losses_zhang", formats=[".png",".eps"])