# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml

RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1974.yaml"

cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 2

if Case == 0:

    filename = 'Performance_data_2023-10-26_11-41-12.xlsx'
    performance_data = ml.plot_functions.load_data(filename)
    
    # Plot mass flow rate at different angular speed
    subset = ["omega"] + list(np.array([0.7, 0.9, 1, 1.1])*design_point["omega"])
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'm', subset, xlabel = "Pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    # Plot total-to-static efficiency at different angular speed
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'eta_ts', subset, xlabel = "Pressure ratio", ylabel = "Total-to-static efficiency [%]", close_fig = False)
    
    # Plot mach at all planes at design angular speed
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'alpha_6', subset, xlabel = "Pressure ratio", ylabel = "Mach", close_fig = False)

    
    # Plot mass flow rate at different pressure ratios
    subset = ["pr_ts", 3.5]
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'omega', 'm', subset, xlabel = "Angular speed [rad/s]", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    
    # # Plot stacked losses at stator on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = ["Y_p_2", "Y_te_2", "Y_inc_2", "Y_s_2"]
    # fig, ax = ml.plot_functions.stack_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Pressure ratio", ylabel = "Losses", title = "Rotor losses")
    
    # # Plot stacked losses at rotor on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = ["Y_p_5", "Y_te_5", "Y_inc_5", "Y_s_5", "Y_cl_5"]
    # fig, ax = ml.plot_functions.stack_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Pressure ratio", ylabel = "Losses", title = "Rotor losses")
    
elif Case == 1:
    
    filename = 'Performance_data_2023-10-26_11-21-57.xlsx'
    performance_data = ml.plot_functions.load_data(filename)
    
    lines = ["m_crit_2","m"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

    lines = ["d_crit_2","d_5"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    lines = ["w_crit_2","w_5"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

    lines = ["Ma_crit_2","Marel_5", "Marel_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

    lines = ["beta_5", "beta_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

elif Case == 2:
    # Get the name of the latest results file
    filename = ml.find_latest_results_file(RESULTS_PATH)
    # filename = "output/performance_analysis_2023-12-06_12-59-53.xlsx" # throat_location  = 0.85
    # filename = "output/performance_analysis_2023-12-06_13-10-58.xlsx" # throat_location  = 1

    # Load performance data
    timestamp = ml.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)

    # Define plot settings
    save_figs = True
    show_figures = True
    color_map = "magma"
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

    # Plot velocities
    # title = "Stator velocity"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["w_1", "w_2"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Velocity [m/s]",
    #     title=title,
    #     filename="stator_velocity"+'_'+timestamp,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # title = "Rotor velocity"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["w_3", "w_4"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Velocity [m/s]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # Plot Mach numbers
    # title = "Stator Mach number"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["Ma_crit_out_1", "Ma_1", "Ma_2"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Mach number",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # title = "Rotor Mach number"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["Ma_crit_out_2", "Ma_rel_3", "Ma_rel_4"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Mach number",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # Plot pressures
    # title = "Stator pressure"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["p_1", "p_2"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Pressure [Pa]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # title = "Rotor pressure"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["p_3", "p_4"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Pressure [Pa]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )
    
    # title = "Rotor density"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["d_crit_2","d_4", "d_5", "d_6"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Density [kg/m^3]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    # Plot flow angles
    # title = "Stator relative flow angles"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["beta_2"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Flow angles [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )

    title = "Rotor relative flow angles"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["beta_4", "beta_subsonic_2", "beta_supersonic_2"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Flow angles [deg]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

    # Group up the losses
    df = data["cascade"]
    losses = [col for col in df.columns if col.startswith("efficiency_drop_")]
    stator_losses = [col for col in losses if col.endswith("_1")]
    rotor_losses = [col for col in losses if col.endswith("_2")]
    data["cascade"]["efficiency_ts_drop_stator"] = df[stator_losses].sum(axis=1)
    data["cascade"]["efficiency_ts_drop_rotor"] = df[rotor_losses].sum(axis=1)

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

