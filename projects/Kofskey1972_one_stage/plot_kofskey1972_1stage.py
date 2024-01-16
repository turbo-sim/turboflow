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
CONFIG_FILE = "kofskey1972_1stage.yaml"

cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 'performance_map'
# Get the name of the latest results file
filename = ml.utils.find_latest_results_file(RESULTS_PATH)
# filename = "output/performance_analysis_2023-12-30_18-23-16.xlsx" 
save_figs = False
validation = True
show_figures = True

if Case == 'pressure_line':

    # Load performance data
    timestamp = ml.utils.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)
    indices = data['overall'].index[data["overall"]["angular_speed"] == 1627]

    for key in data.keys():
        data[key] = data[key].loc[indices]

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

    # title = "Rotor relative flow angles"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=["beta_4"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Flow angle [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )
    # ax1.plot([1.5, 4], [-61.6, -61.6], 'k--')
    # ax1.legend(labels = ['Subsonic angle', 'Metal angle'], loc = 'best')

    # title = "Losses"
    # filename = title.lower().replace(" ", "_") + '_' + timestamp
    # fig1, ax1 = ml.plot_functions.plot_lines(
    #     data,
    #     x_key="PR_ts",
    #     y_keys=[
    #         "efficiency_drop_profile_1",
    #             # "efficiency_drop_incidence_1",
    #             "efficiency_drop_secondary_1",
    #             # "efficiency_drop_clearance_1",
    #             "efficiency_drop_trailing_1",
    #             "efficiency_drop_profile_2",
    #             # "efficiency_drop_incidence_2",
    #             "efficiency_drop_secondary_2",
    #             "efficiency_drop_clearance_2",
    #             "efficiency_drop_trailing_2",
    #             ],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Flow angle [deg]",
    #     title=title,
    #     filename=filename,
    #     outdir=outdir,
    #     color_map=color_map,
    #     save_figs=save_figs,
    # )




    title = "Losses"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=    ['loss_profile_1', 
                    # 'loss_clearance_1', 
                    'loss_secondary_1', 
                    'loss_trailing_1', 
                    # 'loss_incidence_1',
                    'loss_profile_2', 
                    'loss_clearance_2', 
                    'loss_secondary_2', 
                    'loss_trailing_2', 
                    # 'loss_incidence_2'
                    ],
        xlabel="Total-to-static pressure ratio",
        ylabel="Flow angle [deg]",
        title=title,
        filename=filename,
        outdir=outdir,
        color_map=color_map,
        save_figs=save_figs,
    )
    # ax1.plot([1.5, 4], [-61.6, -61.6], 'k--')
    # ax1.legend(labels = ['Subsonic angle', 'Metal angle'], loc = 'best')
    
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
        "alpha_4",
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
    if validation == True:
        filename = "experimental_data_Kofskey1972_1stage_interpolated.xlsx"
        validation_data = pd.read_excel(filename)

        # Define which angukar speed lines that should be plotted
        speed_percent = validation_data["speed_percent"].unique()
        speed_percent = np.flip(speed_percent[0:4])

        def sigmoid(x, k, x0, k2):
            return 1 / (1 + np.exp(-k * (x - x0))) + k2

        # Define colors and markers
        colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
        markers = ['x', 'o', '^', 's']

        # Define plot options
        legend_title = "Percent of design \n angular speed"

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
            ax1.scatter(PR,m,marker = markers[i], color = colors[i])
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
            ax2.scatter(PR, eta_ts,  marker = markers[i], color=colors[i])
            ax2.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )

            # Exit absolute* flow angle
            alpha = validation_data[validation_data["speed_percent"] == speed_percent[i]]["angle_exit_abs"]
            pr_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]]["pressure_ratio_ts"]
            ax4.scatter(pr_ts, alpha, marker = markers[i], color=colors[i])
            ax4.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )

            # Torque
            tau = validation_data[validation_data["speed_percent"] == speed_percent[i]]["torque"]
            pr_ts = validation_data[validation_data["speed_percent"] == speed_percent[i]]["pressure_ratio_ts"]
            ax5.scatter(pr_ts, tau,  marker = markers[i], color=colors[i])
            ax5.legend(
                labels=labels,
                ncols = 2,
                title=legend_title,
            )

    # Manual settings
    ax1.set_xlim([1.4, 4.7])
    ax1.set_ylim([2.5, 2.9])

    ax2.set_xlim([1.4, 4.7])
    ax2.set_ylim([40,100])

    ax3.set_xlim([1.4, 4.7])
    ax3.legend(labels = [70, 90, 100, 110], title=legend_title)

    ax4.set_xlim([1.4, 4.7])
    ax4.set_ylim([-50, 10])

    ax5.set_xlim([1.4, 4.7])
    ax5.set_ylim([40, 140])

    if save_figs:
        ml.plot_functions.save_figure(fig1, "figures\Case_4_mass_flow_rate.png")
        ml.plot_functions.save_figure(fig2, "figures\Case_4_efficiency.png")
        ml.plot_functions.save_figure(fig3, "figures\Case_4_relative_flow_angle.png")
        ml.plot_functions.save_figure(fig4, "figures\Case_4_absolute_flow_angle.png")
        ml.plot_functions.save_figure(fig5, "figures\Case_4_torque.png")

            # Show figures
elif Case == 'error_plot':

    filename_sim = 'output\performance_analysis_2024-01-08_14-06-10.xlsx'
    filename_exp = './experimental_data_kofskey1972_1stage_raw.xlsx'

    speed_percent =np.flip([110, 100, 90, 70])
    data_sim = pd.read_excel(filename_sim, sheet_name=['overall'])
    data_sim = data_sim["overall"]

    sheets = ['Mass flow rate', 'Torque', 'Total-to-static efficiency', 'alpha_out']
    data_exp = pd.read_excel(filename_exp, sheet_name = sheets)
    for sheet in sheets:
        data_exp[sheet] = data_exp[sheet][data_exp[sheet]["omega"].isin(speed_percent)]
        data_mass_flow = data_sim[0:37]
        data_torque = data_sim[37:85]
        data_eta = data_sim[85:170]
        data_alpha = data_sim[170:]


    fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))        
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
    fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
    fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))

    # Define colors and markers
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(speed_percent)))
    markers = ['x', 'o', '^', 's']
    for speed, color, marker in zip(speed_percent, colors, markers):

        fig1, ax1 = ml.plot_functions.plot_error(
            data_exp["Mass flow rate"][data_exp["Mass flow rate"]["omega"] == speed]["m"],
            data_mass_flow[(data_mass_flow["speed_percent"] > speed-1) & (data_mass_flow["speed_percent"] < speed+1)]["mass_flow_rate"],
            fig = fig1,
            ax = ax1,
            color = color,
            marker = marker,
            label = str(speed),
        )


        fig2, ax2 = ml.plot_functions.plot_error(
            data_exp["Total-to-static efficiency"][data_exp["Total-to-static efficiency"]["omega"] == speed]["Efficiency_ts"],
            data_eta[(data_eta["speed_percent"] > speed-1) & (data_eta["speed_percent"] < speed+1)]["efficiency_ts"],
            fig = fig2,
            ax = ax2,
            color = color,
            marker = marker,
            label = str(speed),
        )


        fig3, ax3 = ml.plot_functions.plot_error(
            data_exp["Torque"][data_exp["Torque"]["omega"] == speed]["Torque"],
            data_torque[(data_torque["speed_percent"] > speed-1) & (data_torque["speed_percent"] < speed+1)]["torque"],
            fig = fig3,
            ax = ax3,
            color = color,
            marker = marker,
            label = str(speed),
        )

        fig4, ax4 = ml.plot_functions.plot_error(
            data_exp["alpha_out"][data_exp["alpha_out"]["omega"] == speed]["alpha_out"],
            data_alpha[(data_alpha["speed_percent"] > speed-1) & (data_alpha["speed_percent"] < speed+1)]["exit_flow_angle"],
            fig = fig4,
            ax = ax4,
            color = color,
            marker = marker,
            label = str(speed),
        )

    figs = [fig1, fig3]
    axs = [ax1, ax3]
    error_band = 5/100
    for fig, ax in zip(figs, axs):
        minima = ax.get_xlim()[0]
        maxima = ax.get_xlim()[-1]
        evenly_distributed_values = np.linspace(minima, maxima, num=5)

        lower_bound = evenly_distributed_values * (1-error_band)
        upper_bound = evenly_distributed_values * (1+error_band)

        ax.plot(evenly_distributed_values, lower_bound, 'k--')
        ax.plot(evenly_distributed_values, upper_bound, 'k--')

    eta_error = 5
    minima = ax2.get_xlim()[0]
    maxima = ax2.get_xlim()[-1]
    ax2.plot([minima, maxima], [minima-eta_error, maxima-eta_error], 'k--')
    ax2.plot([minima, maxima], [minima+eta_error, maxima+eta_error], 'k--')

    delta_deg = 5
    minima = ax4.get_xlim()[0]
    maxima = ax4.get_xlim()[-1]
    ax4.plot([minima, maxima], [minima-delta_deg, maxima-delta_deg], 'k--')
    ax4.plot([minima, maxima], [minima+delta_deg, maxima+delta_deg], 'k--')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    ax1.set_title("Mass flow rate")
    ax2.set_title("Total to static efficiency")
    ax3.set_title("Torque")
    ax4.set_title("Absolute flow angle")

    if save_figs:
        ml.plot_functions.save_figure(fig1, "figures/1972_mass_flow_rate_error.png")
        ml.plot_functions.save_figure(fig2, "figures/1972_efficiency_error.png")
        ml.plot_functions.save_figure(fig3, "figures/1972_torque_error.png")
        ml.plot_functions.save_figure(fig4, "figures/1972_absolute_flow_angle_error.png")

if show_figures:
    plt.show()


