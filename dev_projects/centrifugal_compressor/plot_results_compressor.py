import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = "output"
save_figs = False

filename = tf.utilities.find_latest_results_file(RESULTS_PATH)

# Load performance data
timestamp = tf.utilities.extract_timestamp(filename)
data = tf.plot_functions.load_data(filename)
# indices = data["overall"].index[data["overall"]["angular_speed"] == 1627]
# for key in data.keys():
#     data[key] = data[key].loc[indices]

# Define plot settings
color_map = "jet"
outdir = "figures"

# Plot mass flow rate
title = "Mass flow rate"
filename = title.lower().replace(" ", "_") + "_" + timestamp
subsets = ["omega"] + list(52000*2*np.pi/60*np.array([0.75, 0.9, 0.95, 1.0]))
fig1, ax1 = tf.plot_functions.plot_lines(
    data,
    x_key="mass_flow_rate",
    y_keys=["efficiency_tt"],
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
