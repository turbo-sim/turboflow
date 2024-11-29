import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = "output"
save_figs = False

filename = tf.utilities.find_latest_results_file(RESULTS_PATH)
# filename = "output/performance_analysis_2024-11-22_12-33-09.xlsx"

# Load performance data
timestamp = tf.utilities.extract_timestamp(filename)
data = tf.plot_functions.load_data(filename)


# Define plot settings
color_map = "jet"
outdir = "figures"

# Plot mass flow rate
title = "Mass flow rate"
filename = title.lower().replace(" ", "_") + "_" + timestamp
subsets = ["omega"] + list(np.array([10000, 12000, 14000, 16000])*2*np.pi/60)
fig1, ax1 = tf.plot_functions.plot_lines(
    data,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total efficiency",
    color_map=color_map,
    save_figs=save_figs,
)

title = "Mass flow rate"
filename = title.lower().replace(" ", "_") + "_" + timestamp
fig1, ax1 = tf.plot_functions.plot_lines(
    data,
    x_key="mass_flow_rate",
    y_keys=["PR_tt"],
    subsets = subsets,
    xlabel="Mass flow rate [kg/s]",
    ylabel="Total-to-total pressure ratio",
    title=title,
    filename=filename,
    outdir=outdir,
    color_map=color_map,
    save_figs=save_figs,
)



# ax1.set_ylim([0.0, 4.0])

plt.show()
