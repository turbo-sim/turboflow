# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:33:20 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from . import cascade_series as cs
from .non_linear_problem import CascadesNonlinearSystemProblem
from .solver import NonlinearSystemSolver
from .plot_options import set_plot_options

set_plot_options()
    
class PerformanceMap:
    
    def __init__(self):
        
        self.m = None
        self.eta = None
        
    def get_pr_line(self, cascades_data, omega_limits, N, method, ig):
        
        omega = np.linspace(omega_limits[0], omega_limits[1], N)
        
        m = np.zeros(N)
        max_residual = np.zeros(N)
        d3 = np.zeros(N)
        v3 = np.zeros(N)
        w3 = np.zeros(N)
        
        self.Ma1 = np.zeros(N)
        self.Ma2 = np.zeros(N)
        self.Ma3 = np.zeros(N)
        self.Ma4 = np.zeros(N)
        self.Ma5 = np.zeros(N)
        self.Ma6 = np.zeros(N)
        
        self.b1 = np.zeros(N)
        self.b2 = np.zeros(N)
        self.b3 = np.zeros(N)
        self.b4 = np.zeros(N)
        self.b5 = np.zeros(N)
        self.b6 = np.zeros(N)
        
        self.p1 = np.zeros(N)
        self.p2 = np.zeros(N)
        self.p3 = np.zeros(N)
        self.p4 = np.zeros(N)
        self.p5 = np.zeros(N)
        self.p6 = np.zeros(N)

        self.Ma_crit1 = np.zeros(N)
        self.Ma_crit2 = np.zeros(N)
        
        self.Y_p = np.zeros(N)
        self.Y_s = np.zeros(N)
        self.Y_cl = np.zeros(N)
        self.Y_te = np.zeros(N)
        self.Y_inc = np.zeros(N)

        self.Y_p_exit = np.zeros(N)
        self.Y_s_exit = np.zeros(N)
        self.Y_cl_exit = np.zeros(N)
        self.Y_te_exit = np.zeros(N)
        self.Y_inc_exit = np.zeros(N)
        
        use_previous = True
        x0 = None
        
        for i in range(N):
            
            print("\n")
            print(f"Angular speed: {omega[i]}")
            cascades_data["BC"]["omega"] = omega[i]
            cascades_problem = CascadesNonlinearSystemProblem(cascades_data, x0, ig)
            solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
            solution = solver.solve(method=method, options = {'maxfev' : 50})
            
            # Try different solver if solution fail to converge properly
            if max(solution.fun)>1e-6:
                cascades_problem = CascadesNonlinearSystemProblem(cascades_data, x0, ig)
                solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
                solution = solver.solve(method='lm', options = {'maxiter' : 50})
                
            if use_previous == True:
                x0 = cs.convert_scaled_x0(solution.x, cascades_data)
                
            # Store varibles for plotting
            m[i] = cascades_data["overall"]["m"]
            max_residual[i] = max(solution.fun)
            
            d3[i] = cascades_data["plane"]["d"][3]
            v3[i] = cascades_data["plane"]["v"][3]
            w3[i] = cascades_data["plane"]["w"][4]
            
            self.Ma1[i] = cascades_data["plane"]["Marel"][0]
            self.Ma2[i] = cascades_data["plane"]["Marel"][1]
            self.Ma3[i] = cascades_data["plane"]["Marel"][2]
            self.Ma4[i] = cascades_data["plane"]["Marel"][3]
            self.Ma5[i] = cascades_data["plane"]["Marel"][4]
            self.Ma6[i] = cascades_data["plane"]["Marel"][5]
            
            self.b1[i] = cascades_data["plane"]["beta"][0]
            self.b2[i] = cascades_data["plane"]["beta"][1]
            self.b3[i] = cascades_data["plane"]["beta"][2]
            self.b4[i] = cascades_data["plane"]["beta"][3]
            self.b5[i] = cascades_data["plane"]["beta"][4]
            self.b6[i] = cascades_data["plane"]["beta"][5]
            
            self.p1[i] = cascades_data["plane"]["p"][0]
            self.p2[i] = cascades_data["plane"]["p"][1]
            self.p3[i] = cascades_data["plane"]["p"][2]
            self.p4[i] = cascades_data["plane"]["p"][3]
            self.p5[i] = cascades_data["plane"]["p"][4]
            self.p6[i] = cascades_data["plane"]["p"][5]      
            
            self.Ma_crit1[i] = cascades_data["cascade"]["Ma_crit"][0]
            self.Ma_crit2[i] = cascades_data["cascade"]["Ma_crit"][1]
            
            self.Y_p[i] = cascades_data["plane"]["Y_p"][4]
            self.Y_s[i] = cascades_data["plane"]["Y_s"][4]
            self.Y_cl[i] = cascades_data["plane"]["Y_cl"][4]
            self.Y_te[i] = cascades_data["plane"]["Y_te"][4]
            self.Y_inc[i] = cascades_data["plane"]["Y_inc"][4]
            
            self.Y_p_exit[i] = cascades_data["plane"]["Y_p"][5]
            self.Y_s_exit[i] = cascades_data["plane"]["Y_s"][5]
            self.Y_cl_exit[i] = cascades_data["plane"]["Y_cl"][5]
            self.Y_te_exit[i] = cascades_data["plane"]["Y_te"][5]
            self.Y_inc_exit[i] = cascades_data["plane"]["Y_inc"][5]




            
        self.m = m
        self.omega = omega
        self.residuals = max_residual
        
        self.d3 = d3
        self.v3 = v3
        self.w3 = w3
                    
    
    def get_omega_line(self, cascades_data, pr_limits, N, method, ig):
        
        n_cascades = cascades_data["geometry"]["n_cascades"]
        
        # Define pressure ratio array from given inputs
        pr = np.linspace(pr_limits[0], pr_limits[1],N)
        
        # Define arrays for storing omega_line
        m             = np.zeros(N)
        eta_ts        = np.zeros(N)
        R             = np.zeros(N)   
        self.torque = np.zeros(N)

        
        # Define array for plane specific parameters
        Ma = {str(key) : np.zeros(N) for key in range(3*n_cascades)}

        # Define array for cascade specific parameters
        self.Y_p_stator  =  np.zeros(N)
        self.Y_s_stator  =  np.zeros(N)
        self.Y_cl_stator =  np.zeros(N)
        self.Y_te_stator =  np.zeros(N)
        self.Y_inc_stator = np.zeros(N)
        
        self.Y_p_rotor  =  np.zeros(N)
        self.Y_s_rotor  =  np.zeros(N)
        self.Y_cl_rotor =  np.zeros(N)
        self.Y_te_rotor =  np.zeros(N)
        self.Y_inc_rotor = np.zeros(N)

        # If true, uses solution of previous point as initial guess for current point
        use_previous = True
        x0 = None
        
        for i in range(N):
            
            print("\n")
            print(f"Pressure ratio: {pr[i]}")
            cascades_data["BC"]["p_out"] = cascades_data["BC"]["p0_in"]/pr[i]
            cascades_problem = CascadesNonlinearSystemProblem(cascades_data, x0, ig)
            solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
            solution = solver.solve(method=method)
            
            # Try different solver if solution fail to converge properly
            if max(solution.fun)>1e-6:
                cascades_problem = CascadesNonlinearSystemProblem(cascades_data, x0, ig)
                solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
                solution = solver.solve(method='lm')
            
            if use_previous == True:
                x0 = cs.convert_scaled_x0(solution.x, cascades_data)
                                        
            # Store varibles for plotting
            m[i] = cascades_data["overall"]["m"]
            eta_ts[i] = cascades_data["overall"]["eta_ts"]
            R[i] = cascades_data["stage"]["R"][0]

            # Plane
            for j in range(n_cascades*3):
                Ma[str(j)][i] = cascades_data["plane"]["Marel"].values[j]
                
            self.Y_p_stator[i] = cascades_data["plane"]["Y_p"][2]
            self.Y_s_stator[i] = cascades_data["plane"]["Y_s"][2]
            self.Y_cl_stator[i] = cascades_data["plane"]["Y_cl"][2]
            self.Y_te_stator[i] =  cascades_data["plane"]["Y_te"][2]
            self.Y_inc_stator[i] = cascades_data["plane"]["Y_inc"][2]
            
            self.Y_p_rotor[i] = cascades_data["plane"]["Y_p"][5]
            self.Y_s_rotor[i] = cascades_data["plane"]["Y_s"][5]
            self.Y_cl_rotor[i] = cascades_data["plane"]["Y_cl"][5]
            self.Y_te_rotor[i] =  cascades_data["plane"]["Y_te"][5]
            self.Y_inc_rotor[i] = cascades_data["plane"]["Y_inc"][5]
            
            self.torque[i] = cascades_data["overall"]["torque"]
            
        self.pr = pr
        self.m = m
        self.eta_ts = eta_ts
        
        # Stage specific parametes
        self.R = R
              
        
        # Plane specific parameters
        self.Ma = Ma
        
    
    def get_performance_map(self, cascades_data, omega_list, pr_limits, N, method, ig):
        
        # Ensure there is one value in ig for eahc omega
        for key, value in ig.items():
            if isinstance(value, (int, float)):  # Check if value is a scalar
                ig[key] = np.ones(len(omega_list))*ig[key]
            elif len(value) != len(omega_list):
                ig[key] = np.ones(len(omega_list))*ig[key][0]
        
        # Calculate omega line for each omega in a list 
        for i in range(len(omega_list)):
            cascades_data["BC"]["omega"] = omega_list[i]
            ig_omega = {key: value[i] for key, value in ig.items()}
            self.get_omega_line(cascades_data, pr_limits, N, method, ig_omega)
            
            if i == 0:
                m_array = self.m
                eta_ts_array = self.eta_ts
                R_array = self.R
                torque_array = self.torque
            else:
                m_array = np.vstack((m_array, self.m))
                eta_ts_array = np.vstack((eta_ts_array, self.eta_ts))
                R_array = np.vstack((R_array, self.R))
                torque_array = np.vstack((torque_array, self.torque))
                
        self.m = m_array
        self.eta_ts = eta_ts_array    
        self.R = R_array
        self.torque = torque_array
    
    def plot_omega_line(self, cascades_data, maps, pr_limits = [2, 4], N = 20, method = 'hybr', ig = {'R':0.4,'eta':0.9,'Ma_crit':0.92}): 
                                                                                                          
        n_cascades = cascades_data["geometry"]["n_cascades"]
        self.get_omega_line(cascades_data, pr_limits = pr_limits, N = N, method = method, ig = ig)
        
        if maps == 'all':
            maps = ['m', 'eta_ts', 'R', 'Y', 'Ma', 'Y_stator', 'Y_rotor', "torque"]

        figs = {}
        if any(element == 'm' for element in maps):
            fig1, ax1 = plt.subplots()
            ax1.plot(self.pr, self.m)
            ax1.set_xlabel('Total-to-static pressure ratio')
            ax1.set_ylabel('Mass flow rate [kg/s]')
            figs["m"] = [fig1, ax1]
            # plt.close(fig1)
            
        if any(element == 'eta_ts' for element in maps):
            fig2, ax2 = plt.subplots()
            ax2.plot(self.pr, self.eta_ts)
            ax2.set_xlabel('Total-to-static pressure ratio')
            ax2.set_ylabel('Total to static efficiency [%]')
            figs["eta_ts"] = [fig2, ax2]
            # plt.close(fig2)
            
        if any(element == 'R' for element in maps):
            fig3, ax3 = plt.subplots()
            ax3.plot(self.pr, self.R)
            ax3.set_xlabel('Total-to-static pressure ratio')
            ax3.set_ylabel('Degree of reaction')
            figs["R"] = [fig3, ax3]
            # plt.close(fig3)
            
        if any(element == 'Y_stator' for element in maps):
            fig4, ax4 = plt.subplots()
            labels = ["Profile", "Incidence", "Trailing", "Secondary", "Clearance"]
            Y = np.stack((self.Y_p_stator, self.Y_inc_stator, self.Y_te_stator, self.Y_s_stator, self.Y_cl_stator))
            ax4.stackplot(self.pr, Y)
            ax4.set_xlabel('Total-to-static pressure ratio')
            ax4.set_ylabel('Loss coefficient')
            ax4.set_title("Stator")
            ax4.legend(labels, ncols = 1, bbox_to_anchor = (1.4,0.7))
            figs["Y_stator"] = [fig4, ax4]
            # plt.close(fig4)
            
        if any(element == 'Ma' for element in maps):
            fig5, ax5 = plt.subplots()
            for i in range(3*n_cascades):
                ax5.plot(self.pr, self.Ma[str(i)], label = f'Plane: {i}')
            ax5.set_xlabel('Total-to-static pressure ratio')
            ax5.set_ylabel('Mach number')
            ax5.legend(bbox_to_anchor = (1,0.6))
            figs["Ma"] = [fig5, ax5]
            # plt.close(fig5)
            
        if any(element == 'Y_rotor' for element in maps):
            fig6, ax6 = plt.subplots()
            labels = ["Profile", "Incidence", "Trailing", "Secondary", "Clearance"]
            Y = np.stack((self.Y_p_rotor, self.Y_inc_rotor, self.Y_te_rotor, self.Y_s_rotor, self.Y_cl_rotor))
            ax6.stackplot(self.pr, Y)
            ax6.set_xlabel('Total-to-static pressure ratio')
            ax6.set_ylabel('Loss coefficient')
            ax6.set_title("Rotor")
            ax6.legend(labels, ncols = 1, bbox_to_anchor = (1.4,0.7))
            figs["Y_rotor"] = [fig6, ax6]
            # plt.close(fig6)
        
        if any(element == 'torque' for element in maps):
            fig7, ax7 = plt.subplots()
            ax7.plot(self.pr, self.torque)
            ax6.set_xlabel('Total-to-static pressure ratio')
            ax6.set_ylabel('Torque [Nm]')
            figs["torque"] = [fig7, ax7]
            
            
        return figs
    
    def plot_pr_line(self, cascades_data, maps, omega_limits, N = 20, method = 'hybr', ig = {'R':0.4,'eta':0.9,'Ma_crit':0.92}):
                
        self.get_pr_line(cascades_data, omega_limits = omega_limits, N = N, method = method, ig = ig)
        
        if maps == 'all':
            maps = ['m', 'res']
        
        figs = {}
        if any(element == "m" for element in maps):
            fig1, ax1 = plt.subplots()
            ax1.plot(self.omega, self.m)
            ax1.set_xlabel('Angular speed [rad/s]')
            ax1.set_ylabel('Mass flow rate [kg/s]')
            figs["m"] = [fig1, ax1]
            # plt.close(fig1)
            
        if any(element == "res" for element in maps):
            fig2, ax2 = plt.subplots()
            ax2.plot(self.omega, self.residuals)
            ax2.set_xlabel('Angular speed [rad/s]')
            ax2.set_ylabel('Residual')
            figs["res"] = [fig2, ax2]
            # plt.close(fig2)
            
        fig3, ax3 = plt.subplots()
        ax3.plot(self.omega, self.d3)
        ax3.set_xlabel('Angular speed [rad/s]')
        ax3.set_ylabel('Stator exit density')
        figs["d3"] = [fig3, ax3]
        # plt.close(fig3)
            
        fig4, ax4 = plt.subplots()
        ax4.plot(self.omega, self.v3)
        ax4.set_xlabel('Angular speed [rad/s]')
        ax4.set_ylabel('Stator exit velocity')
        figs["v3"] = [fig4, ax4]
        # plt.close(fig4)
        
        fig5, ax5 = plt.subplots()
        ax5.plot(self.omega, self.w3)
        ax5.set_xlabel('Angular speed [rad/s]')
        ax5.set_ylabel('Stator exit relative velocity')
        figs["w3"] = [fig5, ax5]
        # plt.close(fig5)
        
        fig6, ax6 = plt.subplots()
        ax6.plot(self.omega, self.Ma1, label = '1')
        ax6.plot(self.omega, self.Ma2, label = '2')
        ax6.plot(self.omega, self.Ma3, label = '3')
        ax6.plot(self.omega, self.Ma4, label = '4')
        ax6.plot(self.omega, self.Ma5, label = '5')
        ax6.plot(self.omega, self.Ma6, label = '6')
        ax6.plot(self.omega, self.Ma_crit1, label = '1crit', linestyle="--")
        ax6.plot(self.omega, self.Ma_crit2, label = '2crit', linestyle="--")

        ax6.set_xlabel('Angular speed [rad/s]')
        ax6.set_ylabel('Mach')
        ax6.legend()
        figs["Ma"] = [fig6, ax6]
        # plt.close(fig6)
        
        fig7, ax7 = plt.subplots()
        ax7.plot(self.omega, self.b1*180/np.pi, label = '1')
        ax7.plot(self.omega, self.b2*180/np.pi, label = '2')
        ax7.plot(self.omega, self.b3*180/np.pi, label = '3')
        ax7.plot(self.omega, self.b4*180/np.pi, label = '4')
        ax7.plot(self.omega, self.b5*180/np.pi, label = '5')
        ax7.plot(self.omega, self.b6*180/np.pi, label = '6')
        ax7.set_xlabel('Angular speed [rad/s]')
        ax7.set_ylabel('Beta [deg]')
        ax7.legend()
        figs["Beta"] = [fig7, ax7]
        # plt.close(fig7)
        
        fig8, ax8 = plt.subplots()
        ax8.plot(self.omega, self.p1, label = '1')
        ax8.plot(self.omega, self.p2, label = '2')
        ax8.plot(self.omega, self.p3, label = '3')
        ax8.plot(self.omega, self.p4, label = '4')
        ax8.plot(self.omega, self.p5, label = '5')
        ax8.plot(self.omega, self.p6, label = '6')
        ax8.set_xlabel('Angular speed [rad/s]')
        ax8.set_ylabel('Static pressure [Pa]')
        ax8.legend()
        figs["Pressure"] = [fig8, ax8]
        # plt.close(fig8)
        
        fig10, ax10 = plt.subplots()
        labels = ["p", "s", "cl", "te", "inc"]
        Y = np.stack((self.Y_p, self.Y_s, self.Y_cl, self.Y_te, self.Y_inc))
        ax10.stackplot(self.omega, Y)
        ax10.set_xlabel('Angular speed [rad/s]')
        ax10.set_ylabel('Loss coefficient')
        ax10.legend(labels, ncols = 1, bbox_to_anchor = (1.4,0.7))
        figs["Ystack"] = [fig10, ax10]
        # plt.close(fig10)
        
        fig11, ax11 = plt.subplots()
        labels = ["p", "s", "cl", "te", "inc"]
        Y = np.stack((self.Y_p_exit, self.Y_s_exit, self.Y_cl_exit, self.Y_te_exit, self.Y_inc_exit))
        ax11.stackplot(self.omega, Y)
        ax11.set_xlabel('Angular speed [rad/s]')
        ax11.set_ylabel('Loss coefficient')
        ax11.legend(labels, ncols = 1, bbox_to_anchor = (1.4,0.7))
        figs["Ystack_exit"] = [fig11, ax11]
        # plt.close(fig11)
        
        return figs
    
    def plot_perfomance_map(self, cascades_data, maps, omega_list, pr_limits = [2, 4], N = 40, method = 'hybr', ig = {'R':0.4,'eta':0.9,'Ma_crit':0.92}):
        
        if maps == 'all':
            maps = ["m", "eta_ts", 'R', "torque"]
        
        self.get_performance_map(cascades_data, omega_list, pr_limits, N, method, ig)
        figs = {}
        
        # Plot omega lines for each omega
        if any(element == 'm' for element in maps):
            fig1, ax1 = plt.subplots()
            for i in range(len(omega_list)):
                ax1.plot(self.pr, self.m[i], label = str(int(omega_list[i])))
            ax1.set_xlabel('Total-to-static pressure ratio')
            ax1.set_ylabel('Mass flow rate [kg/s]')
            figs["m"] = [fig1, ax1]
            ax1.legend()
            # plt.close(fig1)

        if any(element == 'eta_ts' for element in maps):
            fig2, ax2 = plt.subplots()
            for i in range(len(omega_list)):
                ax2.plot(self.pr, self.eta_ts[i], label = str(int(omega_list[i])))
            ax2.set_xlabel('Total-to-static pressure ratio')
            ax2.set_ylabel('Total to static efficiency [%]')
            ax2.legend()
            figs["eta_ts"] = [fig2, ax2]
            # plt.close(fig2)
        
        if any(element == 'R' for element in maps):
            fig3, ax3 = plt.subplots()
            for i in range(len(omega_list)):
                ax3.plot(self.pr, self.R[i], label = str(int(omega_list[i])))
            ax3.set_xlabel('Total-to-static pressure ratio')
            ax3.set_ylabel('Degree of reaction')
            ax3.legend()
            figs["R"] = [fig3, ax3]
            # plt.close(fig3)

        if any(element == 'torque' for element in maps):
            fig4, ax4 = plt.subplots()
            for i in range(len(omega_list)):
                ax4.plot(self.pr, self.torque[i], label = str(int(omega_list[i])))
            ax4.set_xlabel('Total-to-static pressure ratio')
            ax4.set_ylabel('Torque [Nm]')
            ax4.legend()
            figs["torque"] = [fig4, ax4]
            # plt.close(fig3)
        
        return figs
        
