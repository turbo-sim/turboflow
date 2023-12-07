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

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1974_stator.yaml"


cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 1

if Case == 1:
    # Get the name of the latest results file
    filename = ml.find_latest_results_file(RESULTS_PATH)
    # filename = "output/performance_analysis_2023-11-03_16-09-24.xlsx"

    # Load performance data
    timestamp = ml.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)

    # Define plot settings
    save_figs = False
    show_figures = True
    color_map = "magma"
    outdir = "figures"

    # Plot mass flow rate
    title = "Mass flow rate"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["mass_flow_crit_1", "mass_flow_rate"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )

    # Plot velocities
    title = "Stator velocity"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["w_1", "w_2", "w_3"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Velocity [m/s]",
        title=title,
        filename="stator_velocity"+'_'+timestamp,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )



    # Plot Mach numbers
    title = "Stator Mach number"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["Ma_1", "Ma_2", "Ma_3"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Mach number",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )



    # Plot pressures
    title = "Stator pressure"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["p_1", "p_2", "p_3"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Pressure [Pa]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )



    # Plot flow angles
    title = "Stator relative flow angles"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=["beta_2", "beta_3"],
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
    title = "Total-to-static efficiency distribution"
    filename = title.lower().replace(" ", "_") + "_" + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=[
            "efficiency_ts",
            "efficiency_ts_drop_kinetic",
            "efficiency_ts_drop_stator",
            "efficiency_ts_drop_rotor",
        ],
        xlabel="Total-to-static pressure ratio",
        ylabel="Total-to-static efficiency",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
        stack=True,
        legend_loc="lower left",
    )

    # Show figures
    if show_figures:
        plt.show()
