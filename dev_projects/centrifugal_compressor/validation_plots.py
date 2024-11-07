
import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load experimental and 
filename_exp = "validation_cases\Eckardt\eckardt_impeller_O\eckardt_impeller_O.xlsx"
filename_sim = "output/performance_analysis_2024-11-07_10-13-37.xlsx" # Oh
# filename_sim = "output/performance_analysis_2024-11-07_13-10-59.xlsx" # Zhang set 1, incl. external losses
data_exp = tf.load_data(filename_exp, ["pr_tt", "eta_tt"])
data_sim = tf.load_data(filename_sim)

# Plot simulation data
omegas = [10e3, 12e3, 14e3, 16e3]
color_map = 'viridis'
colors = [plt.get_cmap(color_map)(i / (len(omegas) - 1)) for i in range(len(omegas))]
subsets = ["omega"] + list(2*np.pi/60*np.array(omegas))

# Mass flow rate
fig1, ax1 = tf.plot_functions.plot_lines(
    data_sim,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    colors=colors,
)
ax1.legend(omegas, loc = 'upper left')

# Efficiency
fig2, ax2 = tf.plot_functions.plot_lines(
    data_sim,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency",
    colors=colors,
)
ax2.legend(omegas, loc = 'lower left')

# Losses 
subset = ["omega"] + [2*np.pi/60*14e3]
colors = [plt.get_cmap(color_map)(i / (8 - 1)) for i in range(8)]
fig3, ax3 = tf.plot_functions.plot_lines(
    data_sim,
    x_key="mass_flow_rate",
    y_keys=["incidence_out",
            "blade_loading_out",
            "skin_friction_out",
            "tip_clearance_out",
            "wake_mixing_out",
            # "leakage_out",
            # "recirculation_out",
            # "disk_friction_out",
            ],
    subsets = subset,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Loss distribution",
    colors=colors,
    stack = True
)
loss_labels = ["Incidence", "Blade Loading", "Skin Friction", "Tip Clearance", "Mixing", 
            #    "Leakage", "Recirculation", "Disk Friction",
               ]
ax3.legend(loss_labels, loc = 'upper right')

# Plot experimental data
markers = ["x", "o", "v", "s"]
for i, rpm in zip(range(len(omegas)),omegas):
    data_exp_pr_tt = data_exp["pr_tt"][data_exp["pr_tt"]["rpm"] == rpm]
    data_exp_eta_tt = data_exp["eta_tt"][data_exp["eta_tt"]["rpm"] == rpm]
    ax1.plot(data_exp_pr_tt["mass_flow_rate"], data_exp_pr_tt["pr_tt"], linestyle = "none", color = colors[i], marker = markers[i])
    ax2.plot(data_exp_eta_tt["mass_flow_rate"], data_exp_eta_tt["eta_tt"]*100, linestyle = "none", color = colors[i], marker = markers[i])

plt.show()