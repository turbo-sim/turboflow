# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import meanline_axial as ml

ml.utilities.graphics.set_plot_options()

RESULTS_PATH = "output"
CONFIG_FILE = "kofskey1974_stator.yaml"

cascades_data = ml.read_configuration_file(CONFIG_FILE)

if isinstance(cascades_data["operation_points"], list):
    design_point = cascades_data["operation_points"][0]
else:
    design_point = cascades_data["operation_points"]

Case = 'pressure_line'

# Get the name of the latest results file
# filename = "output/performance_analysis_2024-01-27_10-40-15.xlsx" # Y = 0.1
# filename = "output/performance_analysis_2024-01-29_14-21-42.xlsx" # Y = 0.0
filename = "output/performance_analysis_2024-01-29_16-22-47.xlsx" # Deviation
# filename = ml.find_latest_results_file(RESULTS_PATH)

save_figs = False
validation = True
show_figures = True

if Case == 'pressure_line':
    

    # Load performance data
    timestamp = ml.extract_timestamp(filename)
    data = ml.plot_functions.load_data(filename)

    # Define plot settings
    color_map = "Reds"
    outdir = "figures"

    # Plot mass flow rate
    title = "Mass flow rate"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    fig1, ax1 = ml.plot_functions.plot_lines(
        data,
        x_key="PR_ts",
        y_keys=[ "mass_flow_rate"],
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
    #     y_keys=["w_1", "w_2", "w_3"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Velocity [m/s]",
    #     title=title,
    #     filename="stator_velocity"+'_'+timestamp,
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
    #     y_keys=["Ma_1", "Ma_2", "Ma_3"],
    #     xlabel="Total-to-static pressure ratio",
    #     ylabel="Mach number",
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

    # Plot flow angles
    title = "Stator relative flow angles"
    filename = title.lower().replace(" ", "_") + '_' + timestamp
    colors = plt.get_cmap(color_map)(np.linspace(0.2, 1, 3))
    fig1, ax1 = ml.plot_functions.plot_line(
        data,
        x_key="Ma_rel_2",
        y_key="beta_2",
        # xlabel="Exit relative mach number [-]",
        # ylabel="Exit relative flow angle [deg]",
        color = 'k',
        close_fig = False, 
    )

    # Adjusting the space between the axis and the labels
    # ax1.set_ylabel("Exit relative flow angle [deg]", labelpad=20)  # Increase the space for X-axis
    # ax1.set_xlabel("Exit relative mach number [-]", labelpad=20)  # Increase the space for X-axis

    A_throat = 0.00961713
    A_out = 0.02595989
    Ma_crit_throat = 0.90505927
    # Ma_crit_out = 0.76012957
    # beta_crit_out = 68.04758561
    beta_g = np.arccos(A_throat/A_out)*180/np.pi
    ax1.set_ylim([67.01, 68.6])
    ax1.set_xlim([0.45, 1.2])
    fig1.tight_layout(pad=1, w_pad=None, h_pad=None)

    # Add line to illustrate ECC
    ax1.plot([Ma_crit_throat, Ma_crit_throat], [67.01, beta_g], 'k--')
    ax1.text(Ma_crit_throat-0.02, 67.01-0.1, "$\mathrm{Ma}^*$", fontsize=13, color='k')
    ax1.text(Ma_crit_throat+0.03, 67.01+0.25, "Cascade chokes", fontsize=13, color='k')
    line_y = [67.01+0.5, 67.01+0.4]
    line_x = [Ma_crit_throat + 0.001, Ma_crit_throat + 0.1]
    arrow1 = patches.FancyArrowPatch((line_x[0], line_y[0]), (line_x[1], line_y[1]), arrowstyle='<-', mutation_scale=15, color='k')
    ax1.add_patch(arrow1)

    # Add line to illustrate ICC
    # ax1.plot([0.8, 0.8], [67.01, beta_g-0.1], 'k--')
    # ax1.text(0.8-0.02, 67.01-0.1, "$\mathrm{Ma_{out}}^*$", fontsize=13, color='k')
    # ax1.text(0.8-0.175, 67.01+0.25, "Throat choked \n (ICC)", fontsize=13, color='k')
    # line_y = [67.01+0.4, 67.01+0.5]
    # line_x = [0.8-0.0925, 0.8-0.001]
    # arrow1 = patches.FancyArrowPatch((line_x[0], line_y[0]), (line_x[1], line_y[1]), arrowstyle='->', mutation_scale=15, color='k')
    # ax1.add_patch(arrow1)

    # Add line to illustrate low mach
    ax1.plot([0.5, 0.5], [67.01, beta_g-0.84], 'k--')
    ax1.text(0.5-0.02, 67.01-0.1, "$\mathrm{Ma}_0$", fontsize=13, color='k')

    # Add line to illustrate low speed deviation
    line_y = [beta_g-0.84, beta_g]
    line_x = [0.5, 0.5]
    arrow1 = patches.FancyArrowPatch((line_x[0], line_y[0]), (line_x[1], line_y[1]), arrowstyle='<-', mutation_scale=15, color='k')
    arrow2 = patches.FancyArrowPatch((line_x[-2], line_y[-2]), (line_x[-1], line_y[-1]), arrowstyle='->', mutation_scale=15, color='k')
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    ax1.text(0.52,beta_g-0.42, '$\delta_0$', fontsize=13, color='k')

    # Add line to illustrate gauging angle
    ax1.plot([0, 1.2], [beta_g, beta_g], color = 'k', linestyle = '--', label = r'$\beta_\mathrm{g}$')
    ax1.text(0.45-0.03, beta_g+0.0, r'$\beta_\mathrm{g}$', fontsize=13, color='k')

    # Add arrow illustrating supersonic devaition
    line_y = [beta_g-0.4, beta_g]
    line_x = [1.05, 1.05]
    arrow1 = patches.FancyArrowPatch((line_x[0], line_y[0]), (line_x[1], line_y[1]), arrowstyle='<-', mutation_scale=15, color='k')
    arrow2 = patches.FancyArrowPatch((line_x[-2], line_y[-2]), (line_x[-1], line_y[-1]), arrowstyle='->', mutation_scale=15, color='k')
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    ax1.text(0.95, beta_g+0.05, 'Supersonic deviation', fontsize=13, color='k')

    # Add arrow illustrating subsonic devaition
    line_y = [beta_g-0.4, beta_g]
    line_x = [0.71, 0.71]
    arrow1 = patches.FancyArrowPatch((line_x[0], line_y[0]), (line_x[1], line_y[1]), arrowstyle='<-', mutation_scale=15, color='k')
    arrow2 = patches.FancyArrowPatch((line_x[-2], line_y[-2]), (line_x[-1], line_y[-1]), arrowstyle='->', mutation_scale=15, color='k')
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    ax1.text(0.6, beta_g+0.05, 'Subsonic deviation', fontsize=13, color='k')

    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    # ax1.spines[["left", "bottom"]].set_position(("data", 0))
    # Hide the top and right spines.
    ax1.spines[["top", "right"]].set_visible(False)

    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax1.plot(1.2, 67.01, ">k", clip_on=False)
    ax1.plot(0.45, 68.6, "^k", clip_on=False)

    # Arrow for zero deviation
    line_y = [beta_g + 0.2, beta_g+0.01]
    line_x = [Ma_crit_throat-0.03, Ma_crit_throat]
    arrow1 = patches.FancyArrowPatch((line_x[-2], line_y[-2]), (line_x[-1], line_y[-1]), arrowstyle='->', mutation_scale=15, color='k')
    ax1.add_patch(arrow1)
    ax1.text(Ma_crit_throat-0.1 ,beta_g + 0.21, 'Zero deviation', fontsize=13, color='k')

    # Add axis labels
    ax1.text(0.45-0.03, 68.6-0.05, r'$\beta$', fontsize=13, color='k')
    ax1.text(1.2-0.03, 67.01-0.1, '$\mathrm{Ma}$', fontsize=13, color='k')

    # Add circular marker for ICC and ECC
    ax1.plot([Ma_crit_throat], [beta_g], marker='o', markerfacecolor='w', linestyle = 'none', color = 'k')

    #Grid
    ax1.grid(False)

    # Ticks
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_xticks([])


    # # Plot the total-to-static efficiency distribution
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

elif Case == 'mass_flux_plots':

    filename_1 = "output/performance_analysis_2024-01-27_10-33-23.xlsx" # Y = 0.00
    filename_2 = "output/performance_analysis_2024-01-27_10-39-05.xlsx" # Y = 0.05
    filename_3 = "output/performance_analysis_2024-01-27_10-40-15.xlsx" # Y = 0.10
    filename_4 = "output/performance_analysis_2024-01-27_10-41-19.xlsx" # Y = 0.15
    filename_5 = "output/performance_analysis_2024-01-27_10-42-14.xlsx" # Y = 0.20
    filename_6 = "output/performance_analysis_2024-01-27_10-43-13.xlsx" # Y = 0.25
    filename_7 = "output/performance_analysis_2024-01-27_10-44-08.xlsx" # Y = 0.30

    filenames = [filename_1, filename_3, filename_5, filename_7]
    linestyles = ['-',':','--', '-.']

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # Define plot settings
    color_map = "Reds"
    colors = plt.get_cmap(color_map)(np.linspace(0.2, 1, len(filenames)))
    outdir = "figures"

    for i in range(len(filenames)):
        filename = filenames[i]
        data = ml.plot_functions.load_data(filename)
        fig, ax = ml.plot_functions.plot_line(
        data,
        x_key="Ma_rel_2",
        y_key="mass_flux_throat_1",
        xlabel="Cascade exit relative mach number [-]",
        ylabel="Throat mass flux [kg/$\mathrm{m}^2\cdot$s]",
        fig = fig,
        ax = ax,
        linestyle = linestyles[i],
        color = colors[i], 
        close_fig = False
    )

    labels = ["Y = 0.0", "Y = 0.1", "Y = 0.2", "Y = 0.3"]   
    ax.legend(labels = labels, title = 'Loss coefficient')
    ax.set_ylim([160, 260])

elif Case == 'unphysical_bump':
    # Y = 0.3
    filename_2 = "output/performance_analysis_2024-01-29_14-31-41.xlsx" # No bump
    filename_3 = "output/performance_analysis_2024-01-29_14-21-42.xlsx" # Bump

    # Y = 0.0
    filename_4 = "output/performance_analysis_2024-02-14_10-21-07.xlsx" #Isentropic

    # Y = 0.1
    filename_5 = "output/performance_analysis_2024-02-14_10-33-21.xlsx" # No bump
    filename_6 = "output/performance_analysis_2024-02-14_10-45-31.xlsx" # bump

    # Y = 0.2
    filename_7 = "output/performance_analysis_2024-02-14_10-34-55.xlsx" # No bump
    filename_8 = "output/performance_analysis_2024-02-14_10-37-22.xlsx" # bump

    # filenames = [filename_4, filename_2, filename_3, filename_5, filename_6, filename_7, filename_8]
    filenames = [filename_4, filename_5, filename_7, filename_2, filename_6, filename_8, filename_3]
    # linestyles = ['-']+['-','-.']*3
    linestyles = ['-']*4 + 3*['--']

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # Define plot settings
    color_map = "Reds"
    colors = plt.get_cmap(color_map)(np.linspace(0.2, 1, 4))
    colors = [colors[0]]+[colors[1], colors[2], colors[3]]*2

    for i in range(len(filenames)):
        filename = filenames[i]
        data = ml.plot_functions.load_data(filename)
        if filename == filename_2:
            data_2 = data
        elif filename == filename_3:
            data_3 = data
        fig, ax = ml.plot_functions.plot_line(
        data,
        x_key="PR_ts",
        y_key="mass_flow_rate",
        xlabel="Total-to-static pressure ratio [$p_{0, \mathrm{in}}/p_\mathrm{out}$]",
        ylabel="Mass flow rate [kg/s]",
        fig = fig,
        ax = ax,
        linestyle = linestyles[i],
        color = colors[i], 
        close_fig = False
    )
        
    # ax.legend(labels = ['Ma$_\mathrm{exit}$ = Ma$^*$','Ma$_\mathrm{exit}$ = 1'], title = 'Critical condition', loc = 'lower right')
    labels = ["Y = 0.0", "Y = 0.1", "Y = 0.2", "Y = 0.3"]
    ax.legend(labels = labels, title = 'Loss coefficient', loc = 'lower right')
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)
    # xlim = ax.get_xlims()
    # ylim = ax.get_ylims()
    # extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    # Make zoomed in picture
    x_zoom_2 = data_2["overall"]["PR_ts"]
    y_zoom_2 = data_2["overall"]["mass_flow_rate"]
    x_zoom_3 = data_3["overall"]["PR_ts"]
    y_zoom_3 = data_3["overall"]["mass_flow_rate"]
    x1, x2, y1, y2 = 1.7, 2.25, 2.05, 2.15  # subregion of the original image 
    axins = ax.inset_axes(
    [0.275, 0.1, 0.4, 0.4],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.plot(x_zoom_2, y_zoom_2, color = colors[-1])
    axins.plot(x_zoom_3, y_zoom_3, color = colors[-1], linestyle = '--')
    ax.indicate_inset_zoom(axins, edgecolor="black")
    # axins.imshow(Z2, extent=extent, origin="lower")

    # ax.indicate_inset_zoom(axins, edgecolor="black")



# Show figures
if show_figures:
    plt.show()


