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

RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1974.yaml"

cascades_data = tf.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = "test"  # performance_map/error_plot
# Get the name of the latest results file
filename = tf.utils.find_latest_results_file(RESULTS_PATH)
# filename = 'output\performance_analysis_2024-07-11_11-43-46.xlsx' # New loss coefficient, performance_map
# filename = 'output\performance_analysis_2024-07-11_20-00-05.xlsx' # New loss coefficient, error_plot
# filename = "output\performance_analysis_2024-03-16_20-10-25.xlsx"  # performance map, 1963 REVS
# filename = 'output\performance_analysis_2024-03-16_20-20-51.xlsx' # experimental points 1963 REVS

save_figs = False
validation = True
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
    # arrow1 = patches.FancyArrowPatch((x_text, y_text), (x_des, y_des), arrowstyle='-', mutation_scale=15, color='k')
    # ax.add_patch(arrow1)
    ax.plot([x_text, x_des], [y_text, y_des], linestyle=":", color="k")

    return


if Case == "pressure_line":

    # Load performance data
    timestamp = tf.extract_timestamp(filename)
    data = tf.plot_functions.load_data(filename)

    # Change data to be only design angular speed
    sheets = data.keys()
    filtered_indices = data["overall"][data["overall"]["angular_speed"] == 1963].index
    for sheet in sheets:
        data[sheet] = data[sheet].loc[filtered_indices, :]

    # Define plot settings
    color_map = "Reds"
    outdir = "figures"

    # Plot mass flow rate
    title = "Mass flow rate"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = tf.plot_functions.plot_lines(
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

    title = "Entropy"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = tf.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["s_1", "s_throat_1", "s_2"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Entropy",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

    # Plot mach
    title = "Mach number"
    linstyles = ["-", "--"]*2
    color_map_1 = plt.get_cmap("Reds")(np.linspace(0.2, 1.0, 4))
    color_map_2 = plt.get_cmap("Blues")(np.linspace(0.2, 1.0, 4))
    colors = np.concatenate((color_map_1[1:3], color_map_2[1:3]))
    filename = title.lower().replace(" ", "_") + "_" + timestamp
    fig1, ax1 = tf.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["Ma_rel_2", "Ma_rel_4"],
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Cascade exit relative mach number [-]",
        filename=filename,
        outdir=outdir,
        colors=colors,
        linestyles=linstyles,
        save_figs=save_figs,
    )

    ax1.set_ylim([0.35, 1.05])
    ax1.set_xlim([1.3, 3.49])
    labels = ["Stator actual", "Rotor actual", "Stator critical", "Rotor critical"]
    ax1.legend(labels=labels, ncols=2, loc="lower right")

    ax1.plot([2.54, 2.54], [0.35, 0.978], "k--")
    ax1.plot([3.16, 3.16], [0.35, 0.94], "k--")
    ax1.text(2.61, 0.65, "Stator\nchokes", verticalalignment="center")
    ax1.text(3.21, 0.65, "Rotor\nchokes", verticalalignment="center")

    if save_figs:
        formats=[".png", ".eps"]

        tf.plot_functions.savefig_in_formats(
            fig1, "figures/1974_mach_new_Y", formats=formats
        ) 

    # Plot pressure
    # title = "Rotor pressure"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["w_crit_2"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Pressure [Pa]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # title = "Rotor relative flow angles"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["beta_4"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Flow angles [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # Group up the losses
    # df = data["cascade"]
    # losses = [col for col in df.columns if col.startswith("efficiency_drop_")]
    # stator_losses = [col for col in losses if col.endswith("_1")]
    # rotor_losses = [col for col in losses if col.endswith("_2")]
    # data["cascade"]["efficiency_ts_drop_stator"] = df[stator_losses].sum(axis=1)
    # data["cascade"]["efficiency_ts_drop_rotor"] = df[rotor_losses].sum(axis=1)

    # Plot the total-to-static efficiency distribution
    # title = "Total-to-static efficiency distribution"
    # filename = title.lower().replace(" ", "_") + "_" + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=[
    #         "efficiency_ts",
    #         "efficiency_ts_drop_kinetic",
    #         "efficiency_ts_drop_stator",
    #         "efficiency_ts_drop_rotor",
    #     ],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Total-to-static efficiency",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    #     stack=True,
    #     legend_loc="lower left",
    # )

    # Show figures
    if show_figures:
        plt.show()

elif Case == "performance_map":
    timestamp = tf.utils.extract_timestamp(filename)
    data = tf.plot_functions.load_data(filename)
    color_map = 'Reds'
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
        color_map = color_map,
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
        color_map = color_map,
    )

    # Plot relative flow angle as different angular speed
    fig3, ax3 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["beta_4"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Rotor exit relative flow angle [deg]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        color_map = color_map,
    )

    # Plot absolute flow angle as different angular speed
    fig4, ax4 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["alpha_4"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Rotor exit absolute flow angle [deg]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        color_map = color_map
    )

    # Plot torque as different angular speed
    fig5, ax5 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["torque"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Torque [Nm]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        color_map = color_map,
    )

    # Plot rotor inlet total pressure as different angular speed
    fig6, ax6 = tf.plot_functions.plot_lines(
        data,
        "PR_ts",
        ["Ma_rel_4"],
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Total pressure rotor inlet [Nm]",
        linestyles=["-", ":", "--", "-."],
        close_fig=False,
        color_map = color_map,
    )

    # fig7, ax7 = tf.plot_functions.plot_subsets(
    #     data,
    #     "PR_ts",
    #     "mass_flow_crit_throat_2",
    #     subsets,
    #     xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    #     ylabel="Critical mass flow rate rotor [Nm]",
    #     linestyles=["-", ":", "--", "-."],
    #     close_fig=False,
    # )

    # fig8, ax8 = tf.plot_functions.plot_subsets(
    #     data,
    #     "PR_ts",
    #     "Y_crit_throat_2",
    #     subsets,
    #     xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
    #     ylabel="Critical total loss coefficient [Nm]",
    #     linestyles=["-", ":", "--", "-."],
    #     close_fig=False,
    # )

    # # Plot stacked losses at stator on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = [ "Y_inc_3", "Y_p_3", "Y_te_3", "Y_s_3"]
    # fig, ax = ml.plot_functions.plot_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Losses", title = "Rotor losses", stack = True)

    # # Plot stacked losses at rotor on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = ["Y_inc_6", "Y_p_6", "Y_te_6", "Y_s_6", "Y_cl_6"]
    # fig, ax = ml.plot_functions.plot_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Losses", title = "Rotor losses", stack = True)

    # # Plot stacked losses at rotor on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = ["d_5", "d_6"]
    # fig, ax = ml.plot_functions.plot_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Relative flow angles", title = "Rotor losses", close_fig = False)

    # Validation plots
    if validation:
        filename = "experimental_data_Kofskey1974_raw.xlsx"
        validation_data = pd.read_excel(filename, sheet_name="Interpolated pr_ts")
        mass_flow_data = validation_data[~validation_data["mass_flow_rate"].isna()]
        torque_data = validation_data[~validation_data["torque"].isna()]
        alpha_data = validation_data[~validation_data["alpha"].isna()]

        # Define which angular speed lines that should be plotted
        speed_percent = [70, 90, 100, 105]

        # Define colors and markers
        colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
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

        for i in range(len(speed_percent)):

            # Mass flow rate
            PR = mass_flow_data[mass_flow_data["speed_percent"] == speed_percent[i]][
                "pressure_ratio_ts_interp"
            ]
            m = mass_flow_data[mass_flow_data["speed_percent"] == speed_percent[i]][
                "mass_flow_rate"
            ]
            ax1.plot(PR, m, marker=markers[i], color=colors[i], linestyle="none")
            labels = [str(val) for val in speed_percent] * 2
            ax1.legend(labels=labels, ncols=2, title=legend_title, loc="lower right")

            # Total-to-static efficiency
            # eta_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]][
            #     "efficiency_ts"
            # ]
            # ax2.plot(PR, eta_ts, marker = markers[i], color=colors[i], linestyle = 'none')
            # ax2.legend(
            #     labels=labels,
            #     ncols = 2,
            #     title=legend_title,
            # )

            # Exit absolute* flow angle
            pr_ts = alpha_data[alpha_data["speed_percent"] == speed_percent[i]][
                "pressure_ratio_ts_interp"
            ]
            alpha = alpha_data[alpha_data["speed_percent"] == speed_percent[i]]["alpha"]
            ax4.plot(pr_ts, alpha, marker=markers[i], color=colors[i], linestyle="none")
            ax4.legend(
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
            ax5.plot(pr_ts, tau, marker=markers[i], color=colors[i], linestyle="none")
            ax5.legend(
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
            ax5,
            design_point_coordinates,
            [design_point_torque["pressure_ratio_ts_interp"], 92],
        )
        design_point_coordinates = [
            design_point_alpha["pressure_ratio_ts_interp"],
            design_point_alpha["alpha"],
        ]
        highlight_design_point(
            ax4,
            design_point_coordinates,
            [design_point_alpha["pressure_ratio_ts_interp"], 35],
        )

    ax1.set_xlim([1.3, 3.49])
    ax1.set_ylim([2.0, 2.5])

    ax2.set_xlim([1.3, 3.49])
    ax2.set_ylim([40, 90])

    ax3.set_xlim([1.3, 3.49])
    # ax3.set_ylim([-56, -53])
    ax3.legend(labels=[70, 90, 100, 105], title=legend_title)

    ax4.set_xlim([1.3, 3.49])
    ax4.set_ylim([-40, 60])
    tick_interval = 10
    tick_limits = [-40, 60]
    ax4.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    ax5.set_xlim([1.3, 3.49])
    ax5.set_ylim([10, 100])
    tick_interval = 20
    tick_limits = [10, 110]
    ax5.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    if save_figs:
        formats=[".png", ".eps"]

        tf.plot_functions.savefig_in_formats(
            fig1, "figures/1974_mass_flow_rate_new_Y", formats=formats
        )
        # ml.plot_functions.savefig_in_formats(fig2, "figures/1974_efficiency", formats = [".eps"])
        # ml.plot_functions.savefig_in_formats(fig3, "figures/1974_relative_flow_angle", formats = [".eps"])
        tf.plot_functions.savefig_in_formats(
            fig4, "figures/1974_absolute_flow_angle_new_Y", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig5, "figures/1974_torque_new_Y", formats=formats
        )

elif Case == "error_plot":

    filename_exp = "./experimental_data_kofskey1974_raw.xlsx"

    speed_percent = np.flip([105, 100, 90, 70])
    data_sim = pd.read_excel(filename, sheet_name=["overall"])
    data_sim = data_sim["overall"]

    data_exp = pd.read_excel(filename_exp, sheet_name="Interpolated pr_ts")
    data_exp = data_exp[data_exp["speed_percent"].isin(speed_percent)]
    data_exp = data_exp[~data_exp["pressure_ratio_ts_interp"].isna()].reset_index()

    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    fig3, ax3 = plt.subplots(figsize=(4.8, 4.8))
    fig4, ax4 = plt.subplots(figsize=(4.8, 4.8))
    # fig5, ax5 = plt.subplots(figsize=(4.8, 4.8))

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

        # fig2, ax2 = ml.plot_functions.plot_error(
        #     data_exp[data_exp["speed_percent"] == speed]["efficiency_ts"],
        #     data_sim[data_sim["speed_percent"] == speed]["efficiency_ts"],
        #     fig = fig2,
        #     ax = ax2,
        #     color = color,
        #     marker = marker,
        #     label = str(speed),
        # )

        ax3.plot(
            data_exp.loc[torque_indices, "torque"].values,
            data_sim.loc[torque_indices, "torque"].values,
            color=color,
            marker=marker,
            label=str(speed),
            linestyle = 'none',
        )

        ax4.plot(
            data_exp.loc[alpha_indices, "alpha"].values,
            data_sim.loc[alpha_indices, "exit_flow_angle"].values,
            color=color,
            marker=marker,
            label=str(speed),
            linestyle = 'none',
        )

        # fig5, ax5 = ml.plot_functions.plot_error(
        #     data_exp[data_exp["speed_percent"] == speed]["cos_angle"],
        #     data_sim[data_sim["speed_percent"] == speed]["cos_alpha"],
        #     fig = fig5,
        #     ax = ax5,
        #     color = color,
        #     marker = marker,
        #     label = str(speed),
        # )

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
    ax3.plot(
        evenly_distributed_values, evenly_distributed_values * (1 - 2.5 / 100), "k--"
    )
    ax3.plot(
        evenly_distributed_values, evenly_distributed_values * (1 + 2.5 / 100), "k--"
    )
    ax3.plot(
        evenly_distributed_values, evenly_distributed_values * (1 - 10 / 100), "k:"
    )
    ax3.plot(
        evenly_distributed_values, evenly_distributed_values * (1 + 10 / 100), "k:"
    )

    # Add lines to efficiency plot
    # eta_error = 5
    # ax2.plot([minima, maxima], [minima - eta_error, maxima - eta_error], "k--")
    # ax2.plot([minima, maxima], [minima + eta_error, maxima + eta_error], "k--")

    # Add lines to angle plot
    delta_deg = 5
    ax4.plot([minima, maxima], [minima - delta_deg, maxima - delta_deg], "k--")
    ax4.plot([minima, maxima], [minima + delta_deg, maxima + delta_deg], "k--")
    delta_deg = 10
    ax4.plot([minima, maxima], [minima - delta_deg, maxima - delta_deg], "k:")
    ax4.plot([minima, maxima], [minima + delta_deg, maxima + delta_deg], "k:")

    # Add lines to angle plot
    # ax5.plot(evenly_distributed_values, evenly_distributed_values* (1-2.5/100) , 'k--')
    # ax5.plot(evenly_distributed_values, evenly_distributed_values* (1+2.5/100), 'k--')
    # ax5.plot(evenly_distributed_values, evenly_distributed_values* (1-5/100) , 'k:')
    # ax5.plot(evenly_distributed_values, evenly_distributed_values* (1+5/100), 'k:')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    ax1.set_xlabel("Measured mass flow rate [kg/s]")
    ax1.set_ylabel("Simulated mass flow rate [kg/s]")

    ax2.set_xlabel("Measured total-to-static efficiency [%]")
    ax2.set_ylabel("Simulated total-to-static efficiency [%]")

    ax4.set_xlabel("Measured rotor exit absolute flow angle [deg]")
    ax4.set_ylabel("Simulated rotor exit absolute flow angle [deg]")

    ax3.set_xlabel("Measured torque [Nm]")
    ax3.set_ylabel("Simulated torque [Nm]")

    # ax5.set_xlabel('Cosine of measured rotor exit absolute flow angle [-]')
    # ax5.set_ylabel('Cosine of simulated rotor exit absolute flow angle [-]')

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
    ax4_limits = [-40, 60]
    ax4.set_xlim(ax4_limits)
    ax4.set_ylim(ax4_limits)
    tick_interval = 10
    tick_limits = [-30, 50]
    ax4.set_xticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))
    ax4.set_yticks(range(tick_limits[0], tick_limits[1] + 1, tick_interval))

    # ax5_lims = [0.65, 1.05]
    # ax5.set_ylim(ax5_lims)
    # ax5.set_xlim(ax5_lims)

    if save_figs:
        formats=[".png", ".eps"]
        tf.plot_functions.savefig_in_formats(
            fig1, "figures/error_1974_mass_flow_rate_error_new_Y", formats=formats
        )
        # ml.plot_functions.savefig_in_formats(fig2, "figures/error_1974_efficiency_error", formats = [".eps"])
        tf.plot_functions.savefig_in_formats(
            fig3, "figures/error_1974_torque_error_new_Y", formats=formats
        )
        tf.plot_functions.savefig_in_formats(
            fig4, "figures/error_1974_absolute_flow_angle_error_new_Y", formats=formats
        )

elif Case == "test":

    # Load performance data
    filename_1 = "output/performance_map_evaluate_cascade_throat.xlsx"
    filename_2 = "output/performance_analysis_2024-03-16_20-10-25.xlsx"

    data_1 = tf.plot_functions.load_data(filename_1)
    data_2 = tf.plot_functions.load_data(filename_2)

    subsets = ["omega"] + list(1963*np.array([0.7, 1.0]))
    linestyles = ["-", ":"]
    colors = plt.get_cmap('magma')(np.linspace(0.2, 0.7, 2))
    fig1, ax1 = tf.plot_functions.plot_lines(
        data_1,
        x_key="PR_ts",
        y_keys=["mass_flow_rate"],
        subsets=subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Mass flow rate [kg/s]",
        colors=colors,
        # color_map = 'magma',
        linestyles=['-']*2,
        filename = 'mass_flow_rate',
        outdir = "figures",
        save_figs=False,
    )

    fig1, ax1 = tf.plot_functions.plot_lines(
        data_2,
        x_key="PR_ts",
        y_keys=["mass_flow_rate"],
        subsets=subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Mass flow rate [kg/s]",
        linestyles=[':']*2,
        colors=colors, 
        # color_map = 'Blues',       
        fig = fig1,
        ax = ax1, 
        filename = 'mass_flow_rate',
        outdir = "figures",
        save_figs=False,
    )

    fig2, ax2 = tf.plot_functions.plot_lines(
        data_1,
        x_key="PR_ts",
        y_keys=["torque"],
        subsets=subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Torque [Nm]",
        colors=colors,
        # color_map = 'magma',
        linestyles=['-']*2,
        filename = 'torque',
        outdir = "figures",
        save_figs=False,
    )

    fig2, ax2 = tf.plot_functions.plot_lines(
        data_2,
        x_key="PR_ts",
        y_keys=["torque"],
        subsets=subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Torque [Nm]",
        linestyles=[':']*2,
        colors=colors, 
        # color_map = 'Blues',       
        fig = fig2,
        ax = ax2, 
        filename = 'torque',
        outdir = "figures",
        save_figs=False,
    )

    # Add experimental points
    filename_exp = "experimental_data_Kofskey1974_raw.xlsx"
    validation_data = pd.read_excel(filename_exp, sheet_name="Interpolated pr_ts")
    data_mass_flow_exp = validation_data[~validation_data["mass_flow_rate"].isna()]
    torque_data = validation_data[~validation_data["torque"].isna()]
    alpha_data = validation_data[~validation_data["alpha"].isna()]

    subsets = ["speed_percent"] + list(np.array([70, 100]))
    fig1, ax1 = tf.plot_functions.plot_lines(
        {"data_exp" : data_mass_flow_exp},
        x_key="pressure_ratio_ts_interp",
        y_keys=["mass_flow_rate"],
        subsets=subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Mass flow rate [kg/s]",
        colors=colors,
        # color_map='Reds',
        linestyles = ['none']*2,
        markers = ["x"]*2,
        filename = 'mass_flow_rate',
        outdir = "figures",
        fig = fig1,
        ax = ax1,
        save_figs=False,
    )

    fig2, ax2 = tf.plot_functions.plot_lines(
        {"data_exp" : torque_data},
        x_key="pressure_ratio_ts_interp",
        y_keys=["torque"],
        subsets=subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Torque [Nm]",
        colors=colors,
        # color_map='Reds',
        linestyles = ['none']*2,
        markers = ["x"]*2,
        filename = 'torque',
        outdir = "figures",
        fig = fig2,
        ax = ax2,
        save_figs=False,
    )

    lables = ["$0.7\Omega_\mathrm{des}$, Critical mach", "$1.0\Omega_\mathrm{des}$, Critical mach", 
              "$0.7\Omega_\mathrm{des}$, Max mass flow rate", "$1.0\Omega_\mathrm{des}$, Max mass flow rate", 
              "$0.7\Omega_\mathrm{des}$, Exp. data", "$1.0\Omega_\mathrm{des}$, Exp. data"]
    ax1.legend(labels = lables, ncols = 1, loc = 'lower right')
    ax2.legend(labels = lables, ncols = 1, loc = 'lower right')
    ax1.set_ylim([2.0, 2.5])
    ax2.set_ylim([10, 110])

    if save_figs:
        filename1 = 'validation_mass_flow_kofskey_1974'
        filename2 = 'validation_torque_kofskey_1974'
        tf.plot_functions.savefig_in_formats(fig1, filename1, formats=[".png",".eps"])
        tf.plot_functions.savefig_in_formats(fig2, filename2, formats=[".png",".eps"])

if show_figures:
    plt.show()
