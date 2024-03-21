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
filename = "output/performance_analysis_2024-03-14_01-27-29.xlsx" # experimental point
# filename = "output/performance_analysis_2024-03-14_00-30-26.xlsx" # perfromance map

save_figs = False
validation = True
show_figures = True

if Case == 'pressure_line':

    # Load performance data
    timestamp = ml.utils.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)

    # Change data to be only design angular speed
    sheets = data.keys()
    filtered_indices = data["overall"][data["overall"]['angular_speed'] == 1627].index
    for sheet in sheets:
        data[sheet] = data[sheet].loc[filtered_indices, :]

    # Define plot settings
    color_map = "Reds"
    outdir = "figures"
    
    # Plot mass flow rate
    title = "Mass flow rate"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["mass_flow_rate"],
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Mass flow rate [kg/s]",
        filename=filename,
        outdir=outdir,
        color_map = color_map,
    )
    # ax1.set_ylim([2.38, 2.46])
    # ax1.set_ylim([2.2, 2.3])
    # ax1.legend(labels = ["$\dot{m}$", "$\dot{m}^*_1$", "$\dot{m}^*_2$", "$\dot{m}^*_3$", "$\dot{m}^*_4$"], loc = 'lower right')

    # Plot efficiency curve
    # title = "Total-to-static efficiency"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["efficiency_ts"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Total-to-static efficiency [%]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    # )

    # Plot exit absolute flow angle
    # title = "Rotor exit absolute flow angle"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["alpha_8"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Absolute flow angle [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    # )

    # Relative exit flow angles
    title = "Rotor exit relative flow angle"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["beta_8"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Relative flow angle [deg]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
    )

    # Relative exit flow angles
    # title = "Stator exit relative flow angle"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["beta_8"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Relative flow angle [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    # )

    # title = "Stator exit relative flow angle"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["beta_8"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Relative flow angle [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    # )

    # title = "Static pressure"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["p_2", "p_4", "p_6", "p_8"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Cascade exit static pressure [Pa]",
    #     title='',
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    # )
    # ax1.legend(labels = ['$1^\mathrm{st}$ stator', '$1^\mathrm{st}$ rotor', '$2^\mathrm{nd}$ stator', '$2^\mathrm{nd}$ rotor'], ncols = 1)

    
    # Plot torque
    # title = "Torque"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["torque"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Torque [Nm]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    # )

    # Plot mach
    title = "Mach"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    linestyles = ['-', ':','--','-.']
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["Ma_rel_2", "Ma_rel_4", "Ma_rel_6", "Ma_rel_8"],
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Cascade exit relative mach number [-]",
        filename=filename,
        outdir=outdir,
        linestyles = linestyles,
        color_map='Reds',
    )

    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["Ma_crit_throat_1", "Ma_crit_throat_2", "Ma_crit_throat_3", "Ma_crit_throat_4"],
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Cascade exit relative mach number [-]",
        fig = fig1,
        ax = ax1,
        filename=filename,
        outdir=outdir,
        linestyles = linestyles,
        color_map='Blues',
    )

    labels = ['$1^\mathrm{st}$ stator actual', '$1^\mathrm{st}$ rotor actual', '$2^\mathrm{nd}$ stator actual', '$2^\mathrm{nd}$ rotor actual', 
              '$1^\mathrm{st}$ stator critical', '$1^\mathrm{st}$ rotor critical', '$2^\mathrm{nd}$ stator critical', '$2^\mathrm{nd}$ rotor critical']
    ax1.legend(labels = labels, ncols = 2, loc = 'upper left')
    ax1.set_ylim([0.4,1.4])
    ax1.set_xlim([2.1,5.9])

    if save_figs:
        ml.plot_functions.save_figure(fig1, "figures/1972_2stage_mach.png")

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
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
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
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Total-to-static efficiency [%]",
        linestyles = ['-', ':', '--', '-.'],
        close_fig=False,
    )

    # Plot relative flow angle as different angular speed
    fig3, ax3 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "beta_8",
        subsets,
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
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
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
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
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
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
    
    # Manual settings
    ax1.set_xlim([2.1, 5.4])
    ax1.set_ylim([2.3, 2.5])

    ax2.set_xlim([2.1, 5.4])
    ax2.set_ylim([60, 90])

    ax3.set_xlim([2.1, 5.4])
    ax3.set_ylim([-50.0, -45.5])

    ax4.set_xlim([2.1, 5.4])
    ax4.set_ylim([-40, 50])

    ax5.set_xlim([2.1, 5.4])
    ax5.set_ylim([60, 180])

    if save_figs:
        ml.plot_functions.savefig_in_formats(fig1, "figures/1972_2stage_mass_flow_rate", formats = [".eps"])
        ml.plot_functions.savefig_in_formats(fig2, "figures/1972_2stage_efficiency", formats = [".eps"])
        ml.plot_functions.savefig_in_formats(fig3, "figures/1972_2stage_relative_flow_angle", formats = [".eps"])
        ml.plot_functions.savefig_in_formats(fig4, "figures/1972_2stage_absolute_flow_angle", formats = [".eps"])
        ml.plot_functions.savefig_in_formats(fig5, "figures/1972_2stage_torque", formats = [".eps"])

elif Case == 'error_plot':

    filename_exp = './experimental_data_kofskey1972_2stage_raw.xlsx'    

    speed_percent =np.flip([110, 100, 90, 70])
    data_sim = pd.read_excel(filename, sheet_name=['overall'])
    data_exp = pd.read_excel(filename_exp, sheet_name = ['scaled'])
    data_sim = data_sim["overall"]
    data_exp = data_exp["scaled"]

    data_mass_flow = data_sim[0:28]
    data_torque = data_sim[28:81]
    data_alpha = data_sim[81:]

    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))        
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    fig3, ax3 = plt.subplots(figsize=(4.8, 4.8))
    fig4, ax4 = plt.subplots(figsize=(4.8, 4.8))

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

        fig4, ax4 = ml.plot_functions.plot_error(
            data_exp[(data_exp["speed_percent"] == speed) & (data_exp["cos_angle"].notna())]['cos_angle'].values,
            data_alpha[(data_alpha["speed_percent"] > speed-1) & (data_alpha["speed_percent"] < speed+1)]["cos_alpha"],
            fig = fig4,
            ax = ax4,
            color = color,
            marker = marker,
            label = str(speed),
        )

    # Add lines to plots
    minima = -200
    maxima = 200
    evenly_distributed_values = np.linspace(minima, maxima, num=5)

    # Add lines to mass flow rate plot
    ax1.plot(evenly_distributed_values, evenly_distributed_values* (1-2.5/100) , 'k--')
    ax1.plot(evenly_distributed_values, evenly_distributed_values* (1+2.5/100), 'k--')
    ax1.plot(evenly_distributed_values, evenly_distributed_values, 'k:')

    # Add lines to torque plot
    ax2.plot(evenly_distributed_values, evenly_distributed_values* (1-2.5/100) , 'k--')
    ax2.plot(evenly_distributed_values, evenly_distributed_values* (1+2.5/100), 'k--')
    ax2.plot(evenly_distributed_values, evenly_distributed_values* (1-10/100) , 'k:')
    ax2.plot(evenly_distributed_values, evenly_distributed_values* (1+10/100), 'k:')

    # Add lines to angle plot
    delta_deg = 5
    ax3.plot([minima, maxima], [minima-delta_deg, maxima-delta_deg], 'k--')
    ax3.plot([minima, maxima], [minima+delta_deg, maxima+delta_deg], 'k--')
    delta_deg = 10
    ax3.plot([minima, maxima], [minima-delta_deg, maxima-delta_deg], 'k:')
    ax3.plot([minima, maxima], [minima+delta_deg, maxima+delta_deg], 'k:')

    # Add lines to angle plot
    ax4.plot(evenly_distributed_values, evenly_distributed_values* (1-2.5/100) , 'k--')
    ax4.plot(evenly_distributed_values, evenly_distributed_values* (1+2.5/100), 'k--')
    ax4.plot(evenly_distributed_values, evenly_distributed_values* (1-5/100) , 'k:')
    ax4.plot(evenly_distributed_values, evenly_distributed_values* (1+5/100), 'k:')

    ax1.legend(loc = 'upper left')
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel('Measured mass flow rate [kg/s]')
    ax1.set_ylabel('Simulated mass flow rate [kg/s]')

    ax3.set_xlabel('Measured rotor exit absolute flow angle [deg]')
    ax3.set_ylabel('Simulated rotor exit absolute flow angle [deg]')

    ax2.set_xlabel('Measured torque [Nm]')
    ax2.set_ylabel('Simulated torque [Nm]')

    ax4.set_xlabel('Cosine of measured rotor exit absolute flow angle [-]')
    ax4.set_ylabel('Cosine of simulated rotor exit absolute flow angle [-]')

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

    if save_figs:
        ml.plot_functions.savefig_in_formats(fig1, "figures/error_1972_2stage_mass_flow_rate", formats = [".eps"])
        ml.plot_functions.savefig_in_formats(fig2, "figures/error_1972_2stage_torque", formats = [".eps"])
        ml.plot_functions.savefig_in_formats(fig3, "figures/error_1972_2stage_absolute_flow_angle", formats = [".eps"])


# Show figures
if show_figures:
    plt.show()  