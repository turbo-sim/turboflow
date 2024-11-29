
import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np

impeller = "O" # A, O
model = "zhang" # oh, zhang
save_figs = False

# Map files
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


filename_exp = files_exp[impeller]
filename_incl = files_sim_incl[impeller][model]
filename_exl = files_sim_exl[impeller][model]

data_exp = tf.load_data(filename_exp, ["pr_tt", "eta_tt"])
data_sim_incl = tf.load_data(filename_incl)
data_sim_exl = tf.load_data(filename_exl)

# Plot simulation data
omegas = [10e3, 12e3, 14e3, 16e3]
color_map = 'viridis'
colors = [plt.get_cmap(color_map)(i / (len(omegas) - 1)) for i in range(len(omegas))]
subsets = ["omega"] + list(2*np.pi/60*np.array(omegas))

# Mass flow rate
fig1, ax1 = tf.plot_functions.plot_lines(
    data_sim_exl,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    colors=colors,
)
fig1, ax1 = tf.plot_functions.plot_lines(
    data_sim_incl,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    fig = fig1,
    ax = ax1, 
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    linestyles=["--"]*4,
    title = model,
    colors=colors,
)
ax1.legend(omegas, loc = 'upper left')

# Efficiency
fig2, ax2 = tf.plot_functions.plot_lines(
    data_sim_exl,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency",
    colors=colors,
)
fig2, ax2 = tf.plot_functions.plot_lines(
    data_sim_incl,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    fig = fig2,
    ax = ax2, 
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency",
    linestyles=["--"]*4,
    title = model,
    colors=colors,
)
ax2.legend(omegas, loc = 'lower left')
ax2.set_xlim([2.0, 7.5])
ax2.set_ylim([50, 100])

# Losses 
subset = ["omega"] + [2*np.pi/60*14e3]
colors_losses = [plt.get_cmap(color_map)(i / (10 - 1)) for i in range(10)]
fig3, ax3 = tf.plot_functions.plot_lines(
    data_sim_exl,
    x_key="mass_flow_rate",
    y_keys=["incidence_out",
            "blade_loading_out",
            "skin_friction_out",
            "tip_clearance_out",
            "wake_mixing_out",
            "choke_out",
            "entrance_diffusion_out"
            ],
    subsets = subset,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Loss distribution",
    title = "Excluding parasitic losses",
    colors=colors_losses,
    stack = True
)
loss_labels = ["Incidence", "Blade Loading", "Skin Friction", "Tip Clearance", "Mixing", 
            #    "Leakage", "Recirculation", "Disk Friction",
               "Choke", "Entrance diffusion"
               ]
ax3.legend(loss_labels, loc = 'upper right')
fig4, ax4 = tf.plot_functions.plot_lines(
    data_sim_incl,
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
    title = "Including parasitic losses",
    colors=colors_losses,
    stack = True
)
loss_labels = ["Incidence", "Blade Loading", "Skin Friction", "Tip Clearance", "Mixing", 
               "Leakage", "Recirculation", "Disk Friction",
               "Choke", "Entrance diffusion"
               ]
ax4.legend(loss_labels, loc = 'upper right')

# Plot experimental data
markers = ["x", "o", "v", "s"]
for i, rpm in zip(range(len(omegas)),omegas):
    data_exp_pr_tt = data_exp["pr_tt"][data_exp["pr_tt"]["rpm"] == rpm]
    data_exp_eta_tt = data_exp["eta_tt"][data_exp["eta_tt"]["rpm"] == rpm]
    ax1.plot(data_exp_pr_tt["mass_flow_rate"], data_exp_pr_tt["pr_tt"], linestyle = "none", color = colors[i], marker = markers[i])
    ax2.plot(data_exp_eta_tt["mass_flow_rate"], data_exp_eta_tt["eta_tt"]*100, linestyle = "none", color = colors[i], marker = markers[i])


plt.show()

if save_figs:
    tf.plot_functions.savefig_in_formats(fig1, f"figures/check_parasitic_{model}_impeller{impeller}_PR_tt", formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig2, f"figures/check_parasitic_{model}_impeller{impeller}_eta_tt", formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig3, f"figures/check_parasitic_{model}_impeller{impeller}_losses_exl", formats=[".png",".eps"])
    tf.plot_functions.savefig_in_formats(fig4, f"figures/check_parasitic_{model}_impeller{impeller}_losses_incl", formats=[".png",".eps"])