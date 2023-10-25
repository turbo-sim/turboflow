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

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml

cascades_data = ml.get_cascades_data("kofskey1972_1stage.yaml")
design_point = cascades_data["BC"]

Case = 1

if Case == 0:

    folder = "Case 8/"
    filename = 'results/' + folder + 'Performance_data.xlsx' 
    
    performance_data = ml.plot_functions.load_data(filename)
    
    # Plot mass flow rate at different angular speed
    subset = ["omega"] + list(np.array([0.3, 0.5, 0.7, 0.9, 1, 1.1])*design_point["omega"])
    fig1, ax1 = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'm', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)    
    
    # Plot total-to-static efficiency at different angular speeds
    fig2, ax2 = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'eta_ts', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Total-to-static efficiency [%]", close_fig = False)
    
    # Plot torque as different angular speed
    fig3, ax3 = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'torque', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Torque [Nm]", close_fig = False)
    
    # Plot torque as different angular speed
    fig4, ax4 = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'alpha_6', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Rotor exit absolute flow angle [deg]", close_fig = False)
    
    # Plot torque as different angular speed
    fig5, ax5 = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'beta_6', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Rotor exit relative flow angle [deg]", close_fig = False)
    ax5.legend(labels = ['30', '50', '70', '90', '100', '110'], title = "% of design angular speed")
    
    # Plot mass flow rate at different pressure ratios
    subset = ["pr_ts", 3.5]
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'omega', 'm', subset, xlabel = "Angular speed [rad/s]", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    # Plot mach at all planes at design angular speed
    subset = ["omega", design_point["omega"]]
    column_names = ["Marel_1", "Marel_2", "Marel_3", "Marel_4", "Marel_5", "Marel_6"]
    fig, ax = ml.plot_functions.plot_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Mach")
    
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
        filename = 'Full_Dataset_Kofskey1972_1stage.xlsx'
        validation_data = pd.read_excel(filename, sheet_name = ['Mass flow rate', 'Torque', 'Total-to-static efficiency', 'Beta_out'])
        
        mass_flow_rate = validation_data["Mass flow rate"]
        torque = validation_data["Torque"]
        efficiency_ts = validation_data["Total-to-static efficiency"]
        angle_out = validation_data["Beta_out"]
        
        omega_list = np.sort(mass_flow_rate["omega"].unique())
        
        def sigmoid(x, k, x0, k2):
            return 1/ (1 + np.exp(-k * (x - x0)))+k2
        
        colors = plt.get_cmap('Reds')(np.linspace(0.2, 1, len(omega_list))) 
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
            
            ax1.plot(x_fit, y_fit, label=str(omega_list[i]), linestyle = '--', color = colors[i])
            labels = [str(val) for val in omega_list]*2
            ax1.legend(labels = labels, title = "Percent of design \n angular speed", bbox_to_anchor = (1.32, 0.9))
            ax1.set_ylim([2.55, 2.9])
            ax1.set_xlim([1.3, 4.8])
            
            # Total-to-static efficiency
            eta_ts = efficiency_ts[efficiency_ts["omega"] == omega_list[i]]["Efficiency_ts"]
            pr_ts = efficiency_ts[efficiency_ts["omega"] == omega_list[i]]["PR"]
            ax2.scatter(pr_ts, eta_ts, marker = 'x', color = colors[i])    
            ax2.legend(labels = labels, title = "Percent of design \n angular speed", bbox_to_anchor = (1.32, 0.9))
            ax2.set_ylim([20, 90])
            ax2.set_xlim([1.3, 4.8])

            
            # Torque
            tau = torque[torque["omega"] == omega_list[i]]["Torque"]
            pr_ts = torque[torque["omega"] == omega_list[i]]["PR"]
            ax3.scatter(pr_ts, tau, marker = 'x', color = colors[i]) 
            ax3.legend(labels = labels, title = "Percent of design \n angular speed", bbox_to_anchor = (1.32, 0.9))
            ax3.set_ylim([40, 160])
            ax3.set_xlim([1.3, 4.8])
            
            # Exit absolute* flow angle
            alpha = angle_out[angle_out["omega"] == omega_list[i]]["Beta"]
            pr_ts = angle_out[angle_out["omega"] == omega_list[i]]["PR"]
            ax4.scatter(pr_ts, alpha, marker = 'x', color = colors[i]) 
            ax4.legend(labels = labels, title = "Percent of design \n angular speed", bbox_to_anchor = (1.32, 0.9))
            ax4.set_ylim([-60, 20])
            ax4.set_xlim([1.3, 4.8])

        
        save_figs = True
        if save_figs == True:
            folder = 'results/Case 8/'
            ml.plot_functions.save_figure(fig1, folder + 'mass_flow_rate.png')
            ml.plot_functions.save_figure(fig2, folder + 'efficiency.png')
            ml.plot_functions.save_figure(fig3, folder + 'torque.png')
            ml.plot_functions.save_figure(fig4, folder + 'absolute_flow_angle.png')
            ml.plot_functions.save_figure(fig5, folder + 'relative_flow_angle.png')

        
elif Case == 1:
    
    filename = 'Performance_data_2023-10-25_18-18-45.xlsx'
    performance_data = ml.plot_functions.load_data(filename)
    
    lines = ["m_crit_2","m"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    lines = ["w_5", "w_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Velocity [m/s]", close_fig = True)
    
    # Plot total-to-static efficiency at different angular speeds
    # lines = ['Ma_crit_2', "Marel_5", "Marel_6"]
    # fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mach [-]", close_fig = False)

    lines = ["beta_5", "beta_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Beta [rad]", close_fig = False)

    # lines = ["p_5", "p_6"]
    # fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Pressure [Pa]", close_fig = False)

    # lines = ["h_5", "h_6"]
    # fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Enthalpy [J/kgK]", close_fig = False)
    
    lines = ["alpha_4", "alpha_crit_2"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Alpha [rad]", close_fig = True)

    lines = ["w_crit_2", "w_5", "w_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Velocity [m/s]", close_fig = False)
    # ax1.set_xlim([3,3.5])

    lines = ["d_crit_2", "d_5"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Density [kg/m^3]", close_fig = False)
    # ax1.set_xlim([3,3.5])

    

