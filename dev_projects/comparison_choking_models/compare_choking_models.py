import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import turboflow as tf
import os

filename_evaluate_cascade_critical = "output/evaluate_cascade_critical_results.xlsx"
filename_evaluate_cascade_throat = "output/evaluate_cascade_throat_results.xlsx"

case = 'plot'

if case == 'plot':

    color_map = 'Reds'
    savefig = True

    # Plot results for evaluate_cascade_critical
    data = tf.plot_functions.load_data(filename_evaluate_cascade_critical)
    title = "Mass flow rate"
    subset = ["omega"] + list(1627*np.array([0.9, 1.0, 1.1]))
    fig1, ax1 = tf.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["mass_flow_rate"],
        subsets = subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
        title=title,
        color_map=color_map,
        save_figs=False,
    )

    # Plot results for evaluate_cascade_critical
    data = tf.plot_functions.load_data(filename_evaluate_cascade_throat)
    subset = ["omega"] + list(1627*np.array([0.9, 1.0, 1.1]))
    fig1, ax1 = tf.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["mass_flow_rate"],
        fig = fig1,
        ax = ax1,
        subsets = subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
        linestyles=['--', '--', '--'],
        color_map=color_map,
        save_figs=False,
    )

    # Set legend
    legend_title = "Reference model  Proposed model"
    labels = [f"$\Omega$ = {val}"+"$\Omega_\mathrm{des}$" for val in [0.9, 1.0, 1.1]]*2
    ax1.legend(labels = labels, ncols = 2, title = legend_title, loc = 'lower right')

    # Set axis limits
    ax1.set_ylim([2.55, 2.77])

    if savefig:
        folder = 'figures'
        filename = 'choking_model_comparison'
        formats=[".png",".eps"]
        tf.plot_functions.savefig_in_formats(fig1, os.path.join(folder, filename), formats = formats)

    plt.show()

elif case == 'quantify':

    # Load data
    data_evaluate_cascade_critical = tf.plot_functions.load_data(filename_evaluate_cascade_critical)
    data_evaluate_cascade_throat = tf.plot_functions.load_data(filename_evaluate_cascade_throat)
    data_evaluate_cascade_critical = data_evaluate_cascade_critical["overall"]
    data_evaluate_cascade_throat = data_evaluate_cascade_throat["overall"]
    omegas = data_evaluate_cascade_critical["angular_speed"].unique()

    # Define dict to store results
    errors = {}

    # Print header
    print(f"{'Angular speed':>15} {'Average':>15} {'Max':>15}")

    # Calculate error and print results
    for omega in omegas:

        errors[str(omega)] = {}

        data_omega_evaluate_cascade_critical =  data_evaluate_cascade_critical[data_evaluate_cascade_critical["angular_speed"] == omega]
        data_omega_evaluate_cascade_throat =  data_evaluate_cascade_throat[data_evaluate_cascade_throat["angular_speed"] == omega]

        mass_flow_evaluate_cascade_critical = data_omega_evaluate_cascade_critical["mass_flow_rate"].values
        mass_flow_evaluate_cascade_throat = data_omega_evaluate_cascade_throat["mass_flow_rate"].values

        errors[str(omega)]["raw"] = (mass_flow_evaluate_cascade_critical - mass_flow_evaluate_cascade_throat)/mass_flow_evaluate_cascade_critical*100
        errors[str(omega)]["max"] = max(abs(errors[str(omega)]["raw"]))
        errors[str(omega)]["avg"] = np.mean(abs(errors[str(omega)]["raw"]))
    
        print(f"{omega:15.1f} {errors[str(omega)]['avg']:15.5f} {errors[str(omega)]['max']:15.5f}")
    
