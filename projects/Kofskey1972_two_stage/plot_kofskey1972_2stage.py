# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:12:16 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import meanline_axial as ml

RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1972_2stage.yaml"

cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 'error_plot' # performance_map/error_plot

# Get the name of the latest results file
# filename = ml.utils.find_latest_results_file(RESULTS_PATH)
filename = "output/performance_map_2stage.xlsx" 
save_figs = False
validation = True
show_figures = True

if Case == 'pressure_line':
    
    # Load performance data
    timestamp = ml.utils.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)

    # Define plot settings
    color_map = "jet"
    outdir = "figures"
    
    # Plot mass flow rate
    title = "Mass flow rate"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["mass_flow_rate"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

    # Plot efficiency curve
    title = "Total-to-static efficiency"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["efficiency_ts"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Total-to-static efficiency [%]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

    # Plot exit absolute flow angle
    title = "Rotor exit absolute flow angle"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["alpha_8"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Absolute flow angle [deg]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

    
    # Plot torque
    title = "Torque"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["torque"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Torque [Nm]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

elif Case == 'performance_map':

    timestamp = ml.utils.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)

    # Plot mass flow rate at different angular speed
    subsets = ["omega"] + list(
        np.array([0.7, 0.9, 1, 1.1]) * design_point["omega"]
    )
    fig1, ax1 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "mass_flow_rate",
        subsets,
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
        linestyles = ['-', ':', '--', '-.'],
        close_fig=False,
    )

    # Plot total-to-static efficiency at different angular speeds
    fig2, ax2 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "efficiency_ts",
        subsets,
        xlabel="Total-to-static pressure ratio",
        ylabel="Total-to-static efficiency [%]",
        linestyles = ['-', ':', '--', '-.'],
        close_fig=False,
    )

    # Plot relative flow angle as different angular speed
    fig3, ax3 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "beta_4",
        subsets,
        xlabel="Total-to-static pressure ratio",
        ylabel="Rotor exit relative flow angle [deg]",
        linestyles = ['-', ':', '--', '-.'],
        close_fig=False,
    )

    # Plot absolute flow angle as different angular speed
    fig4, ax4 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "alpha_8",
        subsets,
        xlabel="Total-to-static pressure ratio",
        ylabel="Rotor exit absolute flow angle [deg]",
        linestyles = ['-', ':', '--', '-.'],
        close_fig=False,
    )

    # Plot torque as different angular speed
    fig5, ax5 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "torque",
        subsets,
        xlabel="Total-to-static pressure ratio",
        ylabel="Torque [Nm]",
        linestyles = ['-', ':', '--', '-.'],
        close_fig=False,
    )

    # Validation plots
    if validation:
        filename = "experimental_data_Kofskey1972_2stage_interpolated.xlsx"
        validation_data = pd.read_excel(filename)

        # Define which angukar speed lines that should be plotted
        speed_percent = validation_data["speed_percent"].unique()
        speed_percent = np.flip(speed_percent[0:4])

        # Define colors and markers
        colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
        markers = ['x', 'o', '^', 's']

        # Define plot options
        legend_title = "Percent of design \n angular speed"

        for i in range(len(speed_percent)):

            # Load pressure ratio
            PR = validation_data[validation_data['speed_percent'] == speed_percent[i]]["pressure_ratio_ts"]

            # Mass flow rate
            m = validation_data[validation_data['speed_percent'] == speed_percent[i]]["mass_flow_rate"]
            ax1.plot(PR,m,marker = markers[i], color = colors[i], linestyle = "none")
            labels = [str(val) for val in speed_percent] * 2
            ax1.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )

            # Total-to-static efficiency
            eta_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]][
                "efficiency_ts"
            ]
            ax2.plot(PR, eta_ts,  marker = markers[i], color=colors[i], linestyle = "none")
            ax2.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )

            # Exit absolute* flow angle
            alpha = validation_data[validation_data["speed_percent"] == speed_percent[i]]["angle_exit_abs"]
            pr_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]]["pressure_ratio_ts"]
            ax4.plot(pr_ts, alpha, marker = markers[i], color=colors[i], linestyle = "none")
            ax4.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )

            # Torque
            tau = validation_data[validation_data["speed_percent"] == speed_percent[i]]["torque"]
            pr_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]]["pressure_ratio_ts"]
            ax5.plot(pr_ts, tau,  marker = markers[i], color=colors[i], linestyle = "none")
            ax5.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )
elif Case == 'error_plot':

    filename_sim = 'output\performance_analysis_2024-01-19_11-43-47.xlsx'
    filename_exp = './experimental_data_kofskey1972_2stage_raw.xlsx'    

    speed_percent =np.flip([110, 100, 90, 70])
    data_sim = pd.read_excel(filename_sim, sheet_name=['overall'])
    data_exp = pd.read_excel(filename_exp, sheet_name = ['scaled'])
    data_sim = data_sim["overall"]
    data_exp = data_exp["scaled"]

    data_mass_flow = data_sim[0:28]
    data_torque = data_sim[28:81]
    data_alpha = data_sim[81:]

    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))        
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    fig3, ax3 = plt.subplots(figsize=(4.8, 4.8))

    # Define colors and markers
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
    markers = ['x', 'o', '^', 's']
    for speed, color, marker in zip(speed_percent, colors, markers):

        fig1, ax1 = ml.plot_functions.plot_error(
            data_exp[(data_exp["speed_percent"] == speed) & (data_exp["mass_flow"]>0)]['mass_flow'].values,
            data_mass_flow[(data_mass_flow["speed_percent"] > speed-1) & (data_mass_flow["speed_percent"] < speed+1)]["mass_flow_rate"],
            fig = fig1,
            ax = ax1,
            color = color,
            marker = marker,
            label = str(speed),
        )

        fig2, ax2 = ml.plot_functions.plot_error(
            data_exp[(data_exp["speed_percent"] == speed) & (data_exp["torque"]>0)]['torque'].values,
            data_torque[(data_torque["speed_percent"] > speed-1) & (data_torque["speed_percent"] < speed+1)]["torque"],
            fig = fig2,
            ax = ax2,
            color = color,
            marker = marker,
            label = str(speed),
        )

        fig3, ax3 = ml.plot_functions.plot_error(
            data_exp[(data_exp["speed_percent"] == speed) & (data_exp["alpha"].notna())]['alpha'].values,
            data_alpha[(data_alpha["speed_percent"] > speed-1) & (data_alpha["speed_percent"] < speed+1)]["exit_flow_angle"],
            fig = fig3,
            ax = ax3,
            color = color,
            marker = marker,
            label = str(speed),
        )

    figs = [fig1, fig2]
    axs = [ax1, ax2]
    error_band = 2.5/100
    for fig, ax in zip(figs, axs):
        minima = ax.get_xlim()[0]
        maxima = ax.get_xlim()[-1]
        evenly_distributed_values = np.linspace(minima, maxima, num=5)

        lower_bound = evenly_distributed_values * (1-error_band)
        upper_bound = evenly_distributed_values * (1+error_band)

        ax.plot(evenly_distributed_values, lower_bound, 'k--')
        ax.plot(evenly_distributed_values, upper_bound, 'k--')

    delta_deg = 5
    minima = ax3.get_xlim()[0]
    maxima = ax3.get_xlim()[-1]
    ax3.plot([minima, maxima], [minima-delta_deg, maxima-delta_deg], 'k--')
    ax3.plot([minima, maxima], [minima+delta_deg, maxima+delta_deg], 'k--')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel('Measured mass flow rate [kg/s]')
    ax1.set_ylabel('Simulated mass flow rate [kg/s]')

    ax3.set_xlabel('Measured rotor exit absolute flow angle [deg]')
    ax3.set_ylabel('Simulated rotor exit absolute flow angle [deg]')

    ax2.set_xlabel('Measured torque [Nm]')
    ax2.set_ylabel('Simulated torque [Nm]')

    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')

    if save_figs:
        ml.plot_functions.save_figure(fig1, "figures/error_1972_2stage_mass_flow_rate_error.png")
        ml.plot_functions.save_figure(fig2, "figures/error_1972_2stage_error.png")
        ml.plot_functions.save_figure(fig3, "figures/error_1972_2stage_absolute_flow_angle_error.png")


# Show figures
if show_figures:
    plt.show()  