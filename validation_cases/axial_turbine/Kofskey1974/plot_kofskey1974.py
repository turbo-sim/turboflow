# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import turboflow as tf

# Get design point
CONFIG_FILE = "kofskey1974.yaml"
cascades_data = tf.read_configuration_file(CONFIG_FILE)
design_point = cascades_data["operation_points"]

# Get filename
Case = "error_plot"  # performance_map/error_plot
filename_map = "output/performance_map.xlsx" # Experimental points
filename_exp_points = "output/experimental_points.xlsx" # Experimental points
formats = [".png", ".eps"]

# Options
save_figs = False
show_figures = True

def highlight_design_point(ax, design_point, text_location, markersize=6):

    # Rename variables
    x_des = design_point[0]
    y_des = design_point[1]
    x_text = text_location[0]
    y_text = text_location[1]

    # Plot point
    ax.plot(
        x_des,
        y_des,
        linestyle="none",
        marker="^",
        color="k",
        markerfacecolor="k",
        markersize=markersize,
    )

    # Write text and draw arrow
    if y_des > y_text:
        ax.text(
            x_text,
            y_text,
            "Design point",
            fontsize=13,
            color="k",
            horizontalalignment="center",
            verticalalignment="top",
        )
    else:
        ax.text(
            x_text,
            y_text,
            "Design point",
            fontsize=13,
            color="k",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    ax.plot([x_text, x_des], [y_text, y_des], linestyle=":", color="k")

    return

if Case == "performance_map":

    timestamp = tf.extract_timestamp(filename_map)
    data = tf.plot_functions.load_data(filename_map)

    # Define colors
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, 4))

    # Plot mass flow rate at different angular speed
    subsets = ["omega"] + list(np.array([0.7, 0.9, 1, 1.05]) * design_point["omega"])
    fig1, ax1 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["mass_flow_rate"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Mass flow rate [kg/s]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        colors = colors, 
    )

    # Plot total-to-static efficiency at different angular speeds
    fig2, ax2 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["efficiency_ts"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Total-to-static efficiency [%]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        colors = colors, 
    )

    # Plot absolute flow angle as different angular speed
    fig3, ax3 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["alpha_4"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Rotor exit absolute flow angle [deg]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        colors = colors, 
    )

    # Plot torque as different angular speed
    fig4, ax4 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["torque"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Torque [Nm]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        colors = colors, 
    )

    # Validation plots
    filename = "experimental_data/experimental_data_Kofskey1974_raw.xlsx"
    validation_data = pd.read_excel(filename, sheet_name="Interpolated pr_ts")
    mass_flow_data = validation_data[~validation_data["mass_flow_rate"].isna()]
    torque_data = validation_data[~validation_data["torque"].isna()]
    alpha_data = validation_data[~validation_data["alpha"].isna()]

    # Define which angular speed lines that should be plotted
    speed_percent = [70, 90, 100, 105]

    # Define markers
    markers = ["x", "o", "^", "s"]
    legend_title = "Percent of design \n angular speed"

    # Find design point
    pressure_ratio_des = design_point["p0_in"] / design_point["p_out"]
    index_des = (
        (
            mass_flow_data[mass_flow_data["speed_percent"] == 100][
                "pressure_ratio_ts_interp"
            ]
            - pressure_ratio_des
        )
        .abs()
        .idxmin()
    )
    design_point_mass_flow = mass_flow_data.loc[index_des]
    index_des = (
        (
            torque_data[torque_data["speed_percent"] == 100][
                "pressure_ratio_ts_interp"
            ]
            - pressure_ratio_des
        )
        .abs()
        .idxmin()
    )
    design_point_torque = torque_data.loc[index_des]
    index_des = (
        (
            alpha_data[alpha_data["speed_percent"] == 100][
                "pressure_ratio_ts_interp"
            ]
            - pressure_ratio_des
        )
        .abs()
        .idxmin()
    )
    design_point_alpha = alpha_data.loc[index_des]

    # Plot experimental data for each angular speed 
    for i in range(len(speed_percent)):

        # Mass flow rate
        pr_ts = mass_flow_data[mass_flow_data["speed_percent"] == speed_percent[i]][
            "pressure_ratio_ts_interp"
        ]
        m = mass_flow_data[mass_flow_data["speed_percent"] == speed_percent[i]][
            "mass_flow_rate"
        ]
        ax1.plot(pr_ts, m, marker=markers[i], color=colors[i], linestyle="none")
        labels = [str(val) for val in speed_percent] * 2
        ax1.legend(labels=labels, ncols=2, title=legend_title, loc="lower right")

        # Exit absolute flow angle
        pr_ts = alpha_data[alpha_data["speed_percent"] == speed_percent[i]][
            "pressure_ratio_ts_interp"
        ]
        alpha = alpha_data[alpha_data["speed_percent"] == speed_percent[i]]["alpha"]
        ax3.plot(pr_ts, alpha, marker=markers[i], color=colors[i], linestyle="none")
        ax3.legend(
            labels=labels,
            ncols=2,
            title=legend_title,
        )

        # Torque
        pr_ts = torque_data[torque_data["speed_percent"] == speed_percent[i]][
            "pressure_ratio_ts_interp"
        ]
        tau = torque_data[torque_data["speed_percent"] == speed_percent[i]][
            "torque"
        ]
        ax4.plot(pr_ts, tau, marker=markers[i], color=colors[i], linestyle="none")
        ax4.legend(
            labels=labels,
            ncols=2,
            title=legend_title,
        )

    # Highlight design points
    design_point_coordinates = [
        design_point_mass_flow["pressure_ratio_ts_interp"],
        design_point_mass_flow["mass_flow_rate"],
    ]
    highlight_design_point(
        ax1,
        design_point_coordinates,
        [design_point_mass_flow["pressure_ratio_ts_interp"], 2.42],
    )
    design_point_coordinates = [
        design_point_torque["pressure_ratio_ts_interp"],
        design_point_torque["torque"],
    ]
    highlight_design_point(
        ax4,
        design_point_coordinates,
        [design_point_torque["pressure_ratio_ts_interp"], 92],
    )
    design_point_coordinates = [
        design_point_alpha["pressure_ratio_ts_interp"],
        design_point_alpha["alpha"],
    ]
    highlight_design_point(
        ax3,
        design_point_coordinates,
        [design_point_alpha["pressure_ratio_ts_interp"], 35],
    )

    ax1.set_xlim([1.3, 3.49])
    ax1.set_ylim([2.0, 2.5])

    ax2.set_xlim([1.3, 3.49])
    ax2.set_ylim([40, 90])

    ax3.set_xlim([1.3, 3.49])
    ax3.set_ylim([-40, 60])
    tick_interval = 10
    tick_limits = [-40, 60]
    ax3.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    ax4.set_xlim([1.3, 3.49])
    ax4.set_ylim([10, 100])
    tick_interval = 20
    tick_limits = [10, 110]
    ax4.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    # Ensure tight layout
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    if save_figs:
        tf.plot_functions.savefig_in_formats(
            fig1, "figures/1974_mass_flow_rate_new_Y", formats=formats
        )
        tf.plot_functions.savefig_in_formats(fig2, "figures/1974_efficiency", formats = [".eps"])
        tf.plot_functions.savefig_in_formats(
            fig3, "figures/1974_absolute_flow_angle_new_Y", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig4, "figures/1974_torque_new_Y", formats=formats
        )

elif Case == "error_plot":

    # Define speed lines
    speed_percent = np.flip([105, 100, 90, 70])

    # Get experimental data
    filename_exp = "experimental_data/experimental_data_kofskey1974_raw.xlsx"
    data_exp = pd.read_excel(filename_exp, sheet_name="Interpolated pr_ts")
    data_exp = data_exp[data_exp["speed_percent"].isin(speed_percent)]
    data_exp = data_exp[~data_exp["pressure_ratio_ts_interp"].isna()].reset_index()

    # Get simultions data
    data_sim = pd.read_excel(filename_exp_points, sheet_name=["overall"])
    data_sim = data_sim["overall"]

    # initalize figures
    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    fig3, ax3 = plt.subplots(figsize=(4.8, 4.8))

    # Define colors and markers
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
    markers = ["x", "o", "^", "s"]

    for speed, color, marker in zip(speed_percent, colors, markers):
        data_exp_temp = data_exp[data_exp["speed_percent"] == speed]
        mass_flow_rate_indices = data_exp_temp.loc[
            ~data_exp_temp["mass_flow_rate"].isna()
        ].index
        torque_indices = data_exp_temp.loc[~data_exp_temp["torque"].isna()].index
        alpha_indices = data_exp_temp.loc[~data_exp_temp["alpha"].isna()].index
        ax1.plot(
            data_exp.loc[mass_flow_rate_indices, "mass_flow_rate"].values,
            data_sim.loc[mass_flow_rate_indices, "mass_flow_rate"].values,
            color=color,
            marker=marker,
            label=str(speed),
            linestyle = 'none',
        )

        ax2.plot(
            data_exp.loc[torque_indices, "torque"].values,
            data_sim.loc[torque_indices, "torque"].values,
            color=color,
            marker=marker,
            label=str(speed),
            linestyle = 'none',
        )

        ax3.plot(
            data_exp.loc[alpha_indices, "alpha"].values,
            data_sim.loc[alpha_indices, "exit_flow_angle"].values,
            color=color,
            marker=marker,
            label=str(speed),
            linestyle = 'none',
        )

    # Add lines to plots
    minima = -200
    maxima = 200
    evenly_distributed_values = np.linspace(minima, maxima, num=5)

    # Add lines to mass flow rate plot
    ax1.plot(
        evenly_distributed_values, evenly_distributed_values * (1 - 2.5 / 100), "k--"
    )
    ax1.plot(
        evenly_distributed_values, evenly_distributed_values * (1 + 2.5 / 100), "k--"
    )
    ax1.plot(evenly_distributed_values, evenly_distributed_values, "k:")

    # Add lines to torque plot
    ax2.plot(
        evenly_distributed_values, evenly_distributed_values * (1 - 2.5 / 100), "k--"
    )
    ax2.plot(
        evenly_distributed_values, evenly_distributed_values * (1 + 2.5 / 100), "k--"
    )
    ax2.plot(
        evenly_distributed_values, evenly_distributed_values * (1 - 10 / 100), "k:"
    )
    ax2.plot(
        evenly_distributed_values, evenly_distributed_values * (1 + 10 / 100), "k:"
    )

    # Add lines to angle plot
    delta_deg = 5
    ax3.plot([minima, maxima], [minima - delta_deg, maxima - delta_deg], "k--")
    ax3.plot([minima, maxima], [minima + delta_deg, maxima + delta_deg], "k--")
    delta_deg = 10
    ax3.plot([minima, maxima], [minima - delta_deg, maxima - delta_deg], "k:")
    ax3.plot([minima, maxima], [minima + delta_deg, maxima + delta_deg], "k:")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel("Measured mass flow rate [kg/s]")
    ax1.set_ylabel("Simulated mass flow rate [kg/s]")

    ax3.set_xlabel("Measured rotor exit absolute flow angle [deg]")
    ax3.set_ylabel("Simulated rotor exit absolute flow angle [deg]")

    ax2.set_xlabel("Measured torque [Nm]")
    ax2.set_ylabel("Simulated torque [Nm]")

    # ax1.axis('equal')
    ax1_limits = [2.01, 2.49]
    ax1.set_xlim(ax1_limits)
    ax1.set_ylim(ax1_limits)

    ax2_limits = [50, 90]
    ax2.set_xlim(ax2_limits)
    ax2.set_ylim(ax2_limits)
    tick_interval = 10
    tick_limits = [55, 85]
    ax2.set_xticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))
    ax2.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    ax3_limits = [10, 110]
    ax3.set_xlim(ax3_limits)
    ax3.set_ylim(ax3_limits)
    tick_interval = 20
    tick_limits = [30, 90]
    ax3.set_xticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))
    ax3.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    # ax4.axis('equal')
    ax3_limits = [-40, 60]
    ax3.set_xlim(ax3_limits)
    ax3.set_ylim(ax3_limits)
    tick_interval = 10
    tick_limits = [-30, 50]
    ax3.set_xticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))
    ax3.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    # Ensure tight layout
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    if save_figs:
        formats=[".png", ".eps"]
        tf.plot_functions.savefig_in_formats(
            fig1, "figures/error_1974_mass_flow_rate_error_new_Y", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig2, "figures/error_1974_torque_error_new_Y", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig3, "figures/error_1974_absolute_flow_angle_error_new_Y", formats=formats
        )

if show_figures:
    plt.show()
