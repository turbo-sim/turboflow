# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
import sys

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml
RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1972_1stage.yaml"

cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 1

if Case == 0:
    folder = "Case 8/"
    filename = "results/" + folder + "Performance_data.xlsx"

    data = ml.plot_functions.load_data(filename)

    # Plot mass flow rate at different angular speed
    subset = ["omega"] + list(
        np.array([0.3, 0.5, 0.7, 0.9, 1, 1.1]) * design_point["omega"]
    )
    fig1, ax1 = ml.plot_functions.plot_subsets(
        data,
        "pr_ts",
        "m",
        subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
        close_fig=False,
    )

    # Plot total-to-static efficiency at different angular speeds
    fig2, ax2 = ml.plot_functions.plot_subsets(
        data,
        "pr_ts",
        "eta_ts",
        subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Total-to-static efficiency [%]",
        close_fig=False,
    )

    # Plot torque as different angular speed
    fig3, ax3 = ml.plot_functions.plot_subsets(
        data,
        "pr_ts",
        "torque",
        subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Torque [Nm]",
        close_fig=False,
    )

    # Plot torque as different angular speed
    fig4, ax4 = ml.plot_functions.plot_subsets(
        data,
        "pr_ts",
        "alpha_6",
        subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Rotor exit absolute flow angle [deg]",
        close_fig=False,
    )

    # Plot torque as different angular speed
    fig5, ax5 = ml.plot_functions.plot_subsets(
        data,
        "pr_ts",
        "beta_6",
        subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Rotor exit relative flow angle [deg]",
        close_fig=False,
    )
    ax5.legend(
        labels=["30", "50", "70", "90", "100", "110"], title="% of design angular speed"
    )

    # Plot mass flow rate at different pressure ratios
    subset = ["pr_ts", 3.5]
    fig, ax = ml.plot_functions.plot_subsets(
        data,
        "omega",
        "m",
        subset,
        xlabel="Angular speed [rad/s]",
        ylabel="Mass flow rate [kg/s]",
        close_fig=False,
    )

    # Plot mach at all planes at design angular speed
    subset = ["omega", design_point["omega"]]
    column_names = ["Marel_1", "Marel_2", "Marel_3", "Marel_4", "Marel_5", "Marel_6"]
    fig, ax = ml.plot_functions.plot_lines_on_subset(
        data,
        "pr_ts",
        column_names,
        subset,
        xlabel="Total-to-static pressure ratio",
        ylabel="Mach",
    )

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
    validation = True
    if validation == True:
        filename = "Full_Dataset_Kofskey1972_1stage.xlsx"
        validation_data = pd.read_excel(
            filename,
            sheet_name=[
                "Mass flow rate",
                "Torque",
                "Total-to-static efficiency",
                "Beta_out",
            ],
        )

        mass_flow_rate = validation_data["Mass flow rate"]
        torque = validation_data["Torque"]
        efficiency_ts = validation_data["Total-to-static efficiency"]
        angle_out = validation_data["Beta_out"]

        omega_list = np.sort(mass_flow_rate["omega"].unique())

        def sigmoid(x, k, x0, k2):
            return 1 / (1 + np.exp(-k * (x - x0))) + k2

        colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(omega_list)))
        for i in range(len(omega_list)):
            # Mass flow rate
            m = mass_flow_rate[mass_flow_rate["omega"] == omega_list[i]]["m"]
            pr_ts = mass_flow_rate[mass_flow_rate["omega"] == omega_list[i]]["PR"]
            # ax1.scatter(pr_ts, m, marker = 'x', color = colors[i])

            # Fit the sigmoid function to the data
            params, covariance = curve_fit(sigmoid, pr_ts, m)

            # Extract the fitted parameters
            k_fit, x0_fit, k2_fit = params

            # Generate the curve using the fitted parameters
            x_fit = np.linspace(min(pr_ts), max(pr_ts), 1000)
            y_fit = sigmoid(x_fit, k_fit, x0_fit, k2_fit)

            ax1.plot(
                x_fit, y_fit, label=str(omega_list[i]), linestyle="--", color=colors[i]
            )
            labels = [str(val) for val in omega_list] * 2
            ax1.legend(
                labels=labels,
                title="Percent of design \n angular speed",
                bbox_to_anchor=(1.32, 0.9),
            )
            ax1.set_ylim([2.55, 2.9])
            ax1.set_xlim([1.3, 4.8])

            # Total-to-static efficiency
            eta_ts = efficiency_ts[efficiency_ts["omega"] == omega_list[i]][
                "Efficiency_ts"
            ]
            pr_ts = efficiency_ts[efficiency_ts["omega"] == omega_list[i]]["PR"]
            ax2.scatter(pr_ts, eta_ts, marker="x", color=colors[i])
            ax2.legend(
                labels=labels,
                title="Percent of design \n angular speed",
                bbox_to_anchor=(1.32, 0.9),
            )
            ax2.set_ylim([20, 90])
            ax2.set_xlim([1.3, 4.8])

            # Torque
            tau = torque[torque["omega"] == omega_list[i]]["Torque"]
            pr_ts = torque[torque["omega"] == omega_list[i]]["PR"]
            ax3.scatter(pr_ts, tau, marker="x", color=colors[i])
            ax3.legend(
                labels=labels,
                title="Percent of design \n angular speed",
                bbox_to_anchor=(1.32, 0.9),
            )
            ax3.set_ylim([40, 160])
            ax3.set_xlim([1.3, 4.8])

            # Exit absolute* flow angle
            alpha = angle_out[angle_out["omega"] == omega_list[i]]["Beta"]
            pr_ts = angle_out[angle_out["omega"] == omega_list[i]]["PR"]
            ax4.scatter(pr_ts, alpha, marker="x", color=colors[i])
            ax4.legend(
                labels=labels,
                title="Percent of design \n angular speed",
                bbox_to_anchor=(1.32, 0.9),
            )
            ax4.set_ylim([-60, 20])
            ax4.set_xlim([1.3, 4.8])

        save_figs = True
        if save_figs == True:
            folder = "results/Case 8/"
            ml.plot_functions.save_figure(fig1, folder + "mass_flow_rate.png")
            ml.plot_functions.save_figure(fig2, folder + "efficiency.png")
            ml.plot_functions.save_figure(fig3, folder + "torque.png")
            ml.plot_functions.save_figure(fig4, folder + "absolute_flow_angle.png")
            ml.plot_functions.save_figure(fig5, folder + "relative_flow_angle.png")

elif Case == 1:
    # Get the name of the latest results file
    filename = ml.utils.find_latest_results_file(RESULTS_PATH)
    # filename = "output/performance_analysis_2023-12-06_12-59-53.xlsx" # throat_location  = 0.85
    # filename = "output/performance_analysis_2023-12-06_16-38-46.xlsx" # throat_location  = 1
    
    # Load performance data
    timestamp = ml.utils.extract_timestamp(filename)
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
        y_keys=["beta_4"],
        xlabel="Total-to-static pressure ratio",
        ylabel="Flow angle [deg]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )
    ax1.plot([1.5, 4], [-61.6, -61.6], 'k--')
    ax1.legend(labels = ['Subsonic angle', 'Metal angle'], loc = 'best')
    
    ml.plot_functions.savefig_in_formats(fig1, 'flow_angle', formats=['.png'])
    
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
