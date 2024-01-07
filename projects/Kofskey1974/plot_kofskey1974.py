# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import meanline_axial as ml

RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1974.yaml"

cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 'performance_map'
# Get the name of the latest results file
filename = ml.utils.find_latest_results_file(RESULTS_PATH)
# filename = 'output\performance_analysis_2024-01-05_15-34-10.xlsx' # Case 1
# filename = 'output\performance_analysis_2024-01-05_15-40-29.xlsx' # Case 2
# filename = 'output\performance_analysis_2024-01-05_15-46-52.xlsx' # Case 3
# filename = 'output\performance_analysis_2024-01-06_15-48-18.xlsx' # Case 4
# filename = 'output\performance_analysis_2024-01-06_15-55-23.xlsx' # Case 5
# filename = 'output\performance_analysis_2024-01-06_16-04-00.xlsx' # Case 6

# filename = 'output\performance_analysis_2024-01-07_11-14-12.xlsx'

save_figs = False
validation = True

if Case == 'pressure_line':

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

elif Case == 'performance_map':
    timestamp = ml.utils.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)
    
    show_figures = True

    # Plot mass flow rate at different angular speed
    subsets = ["omega"] + list(
        np.array([0.7, 0.9, 1, 1.05]) * design_point["omega"]
    )
    fig1, ax1 = ml.plot_functions.plot_subsets(
        data,
        "PR_ts",
        "mass_flow_rate",
        subsets,
        xlabel="Total-to-static pressure ratio",
        ylabel="Mass flow rate [kg/s]",
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
        close_fig=False,
    )

    fig5, ax5 = ml.plot_functions.plot_subsets(
    data,
    "PR_ts",
    "beta_4",
    subsets,
    xlabel="Total-to-static pressure ratio",
    ylabel="Rotor exit relative flow angle [deg]",
    close_fig=False,
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
    if validation:
        filename = "experimental_data_Kofskey1974_interpolated.xlsx"
        validation_data = pd.read_excel(filename)

        mass_flow_rate = validation_data["mass_flow_rate"]
        torque = validation_data["torque"]
        efficiency_ts = validation_data["efficiency_ts"]
        angle_out = validation_data["angle_exit_abs"]
        pressure_ratio_ts = validation_data["pressure_ratio_ts"]
        speed_percent = validation_data["speed_percent"].unique()
        speed_percent = speed_percent[0:4]

        def sigmoid(x, k, x0, k2):
            return 1 / (1 + np.exp(-k * (x - x0))) + k2

        colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
        for i in range(len(speed_percent)):

            # Mass flow rate
            # Fit the sigmoid function to the data
            PR = validation_data[validation_data['speed_percent'] == speed_percent[i]]["pressure_ratio_ts"]
            m = validation_data[validation_data['speed_percent'] == speed_percent[i]]["mass_flow_rate"]
            params, covariance = curve_fit(sigmoid, PR.values, m.values)

            # Extract the fitted parameters
            k_fit, x0_fit, k2_fit = params

            # Generate the curve using the fitted parameters
            x_fit = np.linspace(min(PR), max(PR), 1000)
            y_fit = sigmoid(x_fit, k_fit, x0_fit, k2_fit)

            # ax1.plot(
            #     x_fit, y_fit, label=str(speed_percent[i]), linestyle="--", color=colors[i]
            # )
            ax1.scatter(PR,m,marker = 'x', color = colors[i])
            labels = [str(val) for val in speed_percent] * 2
            ax1.legend(
                labels=labels,
                ncols = 2,
                title="Percent of design \n angular speed",
            )
            # ax1.set_ylim([2.55, 2.9])
            # ax1.set_xlim([1.3, 4.8])

            # Total-to-static efficiency
            eta_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]][
                "efficiency_ts"
            ]
            ax2.scatter(PR, eta_ts, marker="x", color=colors[i])
            ax2.legend(
                labels=labels,
                title="Percent of design \n angular speed",
            )
            ax2.set_ylim([20, 90])
            ax2.set_xlim([1.3, 4.8])



            # Torque
            # tau = torque[torque["omega"] == omega_list[i]]["Torque"]
            # pr_ts = torque[torque["omega"] == omega_list[i]]["PR"]
            # ax3.scatter(pr_ts, tau, marker="x", color=colors[i])
            # ax3.legend(
            #     labels=labels,
            #     title="Percent of design \n angular speed",
            #     bbox_to_anchor=(1.32, 0.9),
            # )
            # ax3.set_ylim([40, 160])
            # ax3.set_xlim([1.3, 4.8])

            # Exit absolute* flow angle
            # alpha = angle_out[angle_out["omega"] == omega_list[i]]["Beta"]
            # pr_ts = angle_out[angle_out["omega"] == omega_list[i]]["PR"]
            # ax4.scatter(pr_ts, alpha, marker="x", color=colors[i])
            # ax4.legend(
            #     labels=labels,
            #     title="Percent of design \n angular speed",
            #     bbox_to_anchor=(1.32, 0.9),
            # )
            # ax4.set_ylim([-60, 20])
            # ax4.set_xlim([1.3, 4.8])


            # Show figures
if show_figures:
    plt.show()

if save_figs == True:
    ml.plot_functions.save_figure(fig1, "figures\Case_4_mass_flow_rate.png")
    ml.plot_functions.save_figure(fig2, "figures\Case_4_efficiency.png")
    # ml.plot_functions.save_figure(fig3, folder + "torque.png")
    # ml.plot_functions.save_figure(fig4, folder + "absolute_flow_angle.png")
    labels = [str(val) for val in speed_percent]
    ax5.legend(labels, title="Percent of design \n angular speed")
    ml.plot_functions.save_figure(fig5, "figures\Case_4_relative_flow_angle.png")

