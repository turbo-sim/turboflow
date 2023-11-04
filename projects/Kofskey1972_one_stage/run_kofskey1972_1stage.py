# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import copy

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


CONFIG_FILE = "kofskey1972_1stage.yaml"
cascades_data = ml.read_configuration_file(CONFIG_FILE)


Case = 4

if Case == 0:
    # Solve using nonlinear equation solver
    operating_point = cascades_data["operation_points"][0]
    solver = ml.compute_operating_point(operating_point, cascades_data)
    solver.plot_convergence_history()
    # plt.show()

elif Case == 1:
    # Solve using optimization algorithm
    cascade_problem = ml.CascadesOptimizationProblem(cascades_data)
    solver = ml.solver.OptimizationSolver(
        cascade_problem, cascade_problem.x0, display=True, plot=False
    )
    solver = solver.solve(method="trust-constr")

elif Case == 2:
    # Calculate a dataset corresponding to a dataset
    p_min = 1.6
    p_max = 4.5
    speed_min = 0.3
    speed_max = 1.1

    N_pressure = int((p_max - p_min) * 10) + 1
    N_speed = int((speed_max - speed_min) * 10) + 1
    N = N_pressure * N_speed
    pressure_ratio = np.linspace(p_min, p_max, N_pressure)
    speed = np.linspace(speed_min, speed_max, N_speed)
    p_out = cascades_data["BC"]["p0_in"] / pressure_ratio
    p_out = result = np.concatenate(
        [p_out if i % 2 == 0 else np.flip(p_out) for i in range(N_speed)]
    )
    angular_speed = np.sort(np.repeat(speed * cascades_data["BC"]["omega"], N_pressure))
    boundary_conditions = {
        key: val * np.ones(N)
        for key, val in cascades_data["BC"].items()
        if key != "fluid_name"
    }
    boundary_conditions["fluid_name"] = N * [cascades_data["BC"]["fluid_name"]]
    boundary_conditions["p_out"] = p_out
    boundary_conditions["omega"] = angular_speed

    ml.calculate.performance_map(boundary_conditions, cascades_data)

elif Case == 3:
    filename = "Full_Dataset_Kofskey1972_1stage.xlsx"
    exp_data = pd.read_excel(
        filename,
        sheet_name=[
            "Mass flow rate",
            "Torque",
            "Total-to-static efficiency",
            "Beta_out",
        ],
    )

    p0_in = cascades_data["BC"]["p0_in"]
    omega_des = cascades_data["BC"]["omega"]

    p_out = []
    omega = []

    filenames = [
        "mass_flow_rate.xlsx",
        "torque.xlsx",
        "total-to-static_efficiency.xlsx",
        "beta_out.xlsx",
    ]
    cascades_data_org = cascades_data.copy()
    i = 0
    for key in exp_data.keys():
        p_out = p0_in / exp_data[key]["PR"]
        omega = exp_data[key]["omega"] / 100 * omega_des

        if len(p_out) != len(omega):
            raise Exception("PR and omega have different dimensions")

        N = len(p_out)
        boundary_conditions = {
            key: val * np.ones(N)
            for key, val in cascades_data["BC"].items()
            if key != "fluid_name"
        }
        boundary_conditions["fluid_name"] = N * [cascades_data["BC"]["fluid_name"]]
        boundary_conditions["p_out"] = p_out.values
        boundary_conditions["omega"] = omega.values

        ml.calculate.performance_map(
            boundary_conditions, cascades_data, filename=filenames[i]
        )

        cascades_data = cascades_data_org.copy()

        i += 1

elif Case == 4:
    # Compute performance map according to config file
    # TODO add option to give operation points as list of lists to define several speed lines
    # TODO add option to define range of values for all the parameters of the operating point, including T0_in, p0_in and alpha_in
    # TODO all variables should be ranged to create a nested list of lists
    # TODO the variables should work if they are scalars as well
    # TODO update plotting so the different lines are plotted separately
    # TODO update initial guess strategy so each nested list uses the first element of the parent list as initial guess
    # TODO for example, when computing a new speed line, the first initial guess should be the final solution of the first point of the previous speed line
    
    operation_points = ml.generate_operation_points(cascades_data["performance_map"])
    ml.performance_map(operation_points, cascades_data)
