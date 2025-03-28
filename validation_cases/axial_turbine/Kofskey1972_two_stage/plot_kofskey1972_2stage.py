# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:12:16 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import turboflow as tf
import matplotlib.patches as patches

# Get design point
CONFIG_FILE = "kofskey1972_2stage.yaml"
cascades_data = tf.read_configuration_file(CONFIG_FILE)
design_point = cascades_data["operation_points"]

# Get filename
Case = "performance_map"  # performance_map/error_plot
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
    subsets = ["omega"] + list(np.array([0.7, 0.9, 1, 1.1]) * design_point["omega"])
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
        ["alpha_8"],
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
    filename = "experimental_data/experimental_data_Kofskey1972_2stage_interpolated.xlsx"
    validation_data = pd.read_excel(filename)

    # Define which angukar speed lines that should be plotted
    speed_percent = validation_data["speed_percent"].unique()
    speed_percent = np.flip(speed_percent[0:4])

    # Define markers
    markers = ["x", "o", "^", "s"]

    # Define plot options
    legend_title = "Percent of design \n angular speed"

    # Find design point
    pressure_ratio_des = design_point["p0_in"] / design_point["p_out"]
    index_des = (
        (
            validation_data[validation_data["speed_percent"] == 100][
                "pressure_ratio_ts"
            ]
            - pressure_ratio_des
        )
        .abs()
        .idxmin()
    )
    design_point_result = validation_data.loc[index_des]

    # Plot experimental data for each angular speed
    for i in range(len(speed_percent)):

        # Mass flow rate
        pr_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]][
            "pressure_ratio_ts"
        ]
        m = validation_data[validation_data["speed_percent"] == speed_percent[i]][
            "mass_flow_rate"
        ]
        ax1.plot(pr_ts, m, marker=markers[i], color=colors[i], linestyle="none")
        labels = [str(val) for val in speed_percent] * 2
        ax1.legend(
            labels=labels,
            ncols = 2,
            title=legend_title,
            loc = "lower right"
        )

        # Total-to-static efficiency
        eta_ts = validation_data[
            validation_data["speed_percent"] == speed_percent[i]
        ]["efficiency_ts"]
        ax2.plot(pr_ts, eta_ts, marker=markers[i], color=colors[i], linestyle="none")
        ax2.legend(
            labels=labels,
            ncols=2,
            title=legend_title,
        )

        # Exit absolute flow angle
        alpha = validation_data[
            validation_data["speed_percent"] == speed_percent[i]
        ]["angle_exit_abs"]
        pr_ts = validation_data[
            validation_data["speed_percent"] == speed_percent[i]
        ]["pressure_ratio_ts"]
        ax3.plot(pr_ts, alpha, marker=markers[i], color=colors[i], linestyle="none")
        ax3.legend(
            labels=labels,
            ncols=2,
            title=legend_title,
        )

        # Torque
        tau = validation_data[validation_data["speed_percent"] == speed_percent[i]][
            "torque"
        ]
        pr_ts = validation_data[
            validation_data["speed_percent"] == speed_percent[i]
        ]["pressure_ratio_ts"]
        ax4.plot(pr_ts, tau, marker=markers[i], color=colors[i], linestyle="none")
        ax4.legend(
            labels=labels,
            ncols=2,
            title=legend_title,
        )

    # Highlight design points
    design_point_coordinates = [
        design_point_result["pressure_ratio_ts"],
        design_point_result["mass_flow_rate"],
    ]
    highlight_design_point(ax1, design_point_coordinates, [4.25, 2.46])
    design_point_coordinates = [
        design_point_result["pressure_ratio_ts"],
        design_point_result["torque"],
    ]
    highlight_design_point(ax4, design_point_coordinates, [3.25, 165])
    design_point_coordinates = [
        design_point_result["pressure_ratio_ts"],
        design_point_result["angle_exit_abs"],
    ]
    highlight_design_point(ax3, design_point_coordinates, [3.25, -25])

    # Manual settings
    # ax1.set_xlim([2.1, 5.4])
    # ax1.set_ylim([2.3, 2.5])

    # ax2.set_xlim([2.1, 5.4])
    # ax2.set_ylim([60, 90])

    ax3.set_xlim([2.1, 5.4])
    ax3.set_ylim([-40, 50])

    ax4.set_xlim([2.1, 5.4])
    ax4.set_ylim([60, 180])

    # Ensure tight layout
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    if save_figs:
        tf.plot_functions.savefig_in_formats(
            fig1, "figures/1972_2stage_mass_flow_rate", formats=formats
        )
        tf.plot_functions.savefig_in_formats(fig2, "figures/1972_2stage_efficiency", formats = formats)
        tf.plot_functions.savefig_in_formats(
            fig3, "figures/1972_2stage_absolute_flow_angle", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig4, "figures/1972_2stage_torque", formats=formats
        )

elif Case == "error_plot":

    # Define speed lines
    speed_percent = np.flip([110, 100, 90, 70])

    # Get experminetal data
    filename_exp = "experimental_data/experimental_data_kofskey1972_2stage_raw.xlsx"
    data_exp = pd.read_excel(filename_exp, sheet_name=["scaled"])
    data_exp = data_exp["scaled"]

    # Get simulations data
    data_sim = pd.read_excel(filename_exp_points, sheet_name=["overall"])
    data_sim = data_sim["overall"]
    data_mass_flow = data_sim[0:28]
    data_torque = data_sim[28:81]
    data_alpha = data_sim[81:]

    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    fig3, ax3 = plt.subplots(figsize=(4.8, 4.8))
    fig4, ax4 = plt.subplots(figsize=(4.8, 4.8))

    # Define colors and markers
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
    markers = ["x", "o", "^", "s"]
    for speed, color, marker in zip(speed_percent, colors, markers):

        fig1, ax1 = tf.plot_functions.plot_error(
            data_exp[
                (data_exp["speed_percent"] == speed) & (data_exp["mass_flow"] > 0)
            ]["mass_flow"].values,
            data_mass_flow[
                (data_mass_flow["angular_speed"] > speed/100*design_point["omega"] - 1)
                & (data_mass_flow["angular_speed"] < speed/100*design_point["omega"] + 1)
            ]["mass_flow_rate"],
            fig=fig1,
            ax=ax1,
            color=color,
            marker=marker,
            label=str(speed),
        )

        fig2, ax2 = tf.plot_functions.plot_error(
            data_exp[(data_exp["speed_percent"] == speed) & (data_exp["torque"] > 0)][
                "torque"
            ].values,
            data_torque[
                (data_torque["angular_speed"] > speed/100*design_point["omega"] - 1)
                & (data_torque["angular_speed"] < speed/100*design_point["omega"] + 1)
            ]["torque"],
            fig=fig2,
            ax=ax2,
            color=color,
            marker=marker,
            label=str(speed),
        )

        fig3, ax3 = tf.plot_functions.plot_error(
            data_exp[
                (data_exp["speed_percent"] == speed) & (data_exp["alpha"].notna())
            ]["alpha"].values,
            data_alpha[
                (data_alpha["angular_speed"] > speed/100*design_point["omega"] - 1)
                & (data_alpha["angular_speed"] < speed/100*design_point["omega"] + 1)
            ]["exit_flow_angle"],
            fig=fig3,
            ax=ax3,
            color=color,
            marker=marker,
            label=str(speed),
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

    # Add lines to angle plot
    ax4.plot(
        evenly_distributed_values, evenly_distributed_values * (1 - 2.5 / 100), "k--"
    )
    ax4.plot(
        evenly_distributed_values, evenly_distributed_values * (1 + 2.5 / 100), "k--"
    )
    ax4.plot(evenly_distributed_values, evenly_distributed_values * (1 - 5 / 100), "k:")
    ax4.plot(evenly_distributed_values, evenly_distributed_values * (1 + 5 / 100), "k:")

    ax1.legend(loc="upper left")
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel("Measured mass flow rate [kg/s]")
    ax1.set_ylabel("Simulated mass flow rate [kg/s]")

    ax3.set_xlabel("Measured rotor exit absolute flow angle [deg]")
    ax3.set_ylabel("Simulated rotor exit absolute flow angle [deg]")

    ax2.set_xlabel("Measured torque [Nm]")
    ax2.set_ylabel("Simulated torque [Nm]")

    ax4.set_xlabel("Cosine of measured rotor exit absolute flow angle [-]")
    ax4.set_ylabel("Cosine of simulated rotor exit absolute flow angle [-]")

    # ax1.axis('equal')
    ax1_limits = [2.3, 2.5]
    ax1.set_xlim(ax1_limits)
    ax1.set_ylim(ax1_limits)
    tick_interval = 0.025
    tick_limits = [2.325, 2.475]
    ax1.set_xticks(np.linspace(tick_limits[0], tick_limits[1], 6))
    ax1.set_yticks(np.linspace(tick_limits[0], tick_limits[1], 6))

    ax2_limits = [60, 180]
    ax2.set_xlim(ax2_limits)
    ax2.set_ylim(ax2_limits)
    tick_interval = 20
    tick_limits = [80, 160]
    ax2.set_xticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))
    ax2.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    # ax3.axis('equal')
    ax3_limits = [-40, 50]
    ax3.set_xlim(ax3_limits)
    ax3.set_ylim(ax3_limits)
    tick_interval = 10
    tick_limits = [-30, 40]
    ax3.set_xticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))
    ax3.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    ax4_lims = [0.70, 1.05]
    ax4.set_ylim(ax4_lims)
    ax4.set_xlim(ax4_lims)

    # Ensure tight layout
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    if save_figs:
        tf.plot_functions.savefig_in_formats(
            fig1, "figures/error_1972_2stage_mass_flow_rate", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig2, "figures/error_1972_2stage_torque", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig3, "figures/error_1972_2stage_absolute_flow_angle", formats=formats
        )


# Show figures
if show_figures:
    plt.show()
