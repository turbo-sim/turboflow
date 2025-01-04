
import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load object
# filename_slsqp = "pickle_file_test_slsqp.pkl"
filename_slsqp = "output/pickle_slsqp_2024-08-19_09-17-24.pkl"
filename_ipopt = "pickle_file_test_ipopt.pkl"
# filename_sga = "pickle_file_test_sga.pkl"
filename_sga = "output/pickle_sga_2024-08-19_09-11-25.pkl"


solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_sga = tf.load_from_pickle(filename_sga)

# plots = ["convergence_process", "velocity_triangle", "design_variables"]
plots = []

# print(solver_sga.problem.optimization_process.root_finder_solutions[0])
print(solver_slsqp.convergence_history["constraint_violation"])

if "convergence_process" in plots:

    convergence_data = {"SLSQP" : {"objective" : solver_slsqp.convergence_history["objective_value"],
                               "constraint_violation" : solver_slsqp.convergence_history["constraint_violation"],
                               "iterations" : solver_slsqp.convergence_history["grad_count"]},
                    "IPOPT" : {"objective" : solver_ipopt.convergence_history["objective_value"],
                               "constraint_violation" : solver_ipopt.convergence_history["constraint_violation"],
                               "iterations" : solver_ipopt.convergence_history["grad_count"]}}
    
    fig, ax = tf.plot_functions.plot_convergence_process(convergence_data, plot_together=False)

if "velocity_triangle" in plots:
    
    fig1, ax1 = tf.plot_functions.plot_velocity_triangle_stage(solver_slsqp.problem.results["plane"], color = "#0072BD")
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_ipopt.problem.results["plane"], fig = fig1, ax = ax1, linestyle = '--', color = "#D95319")
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_sga.problem.results["plane"], fig = fig1, ax = ax1, linestyle = ':', color = "k")

if "design_variables" in plots:

    initial_slsqp = solver_slsqp.convergence_history["x"][0]
    final_slsqp = solver_slsqp.convergence_history["x"][-1][13:]
    final_ipopt = solver_ipopt.convergence_history["x"][-1][13:]
    final_ipopt = solver_ipopt.convergence_history["x"][-1][13:]
    lower_bounds = solver_slsqp.problem.bounds[0]
    upper_bounds = solver_slsqp.problem.bounds[1]
    variable_names = [
        r"$\omega_s$",  # specific speed
        r"$\nu$",  # blade jet ratio
        r"$\frac{r_t}{r_h}_{\text{in}_1}$",  # hub-tip ratio in 1
        r"$\frac{r_t}{r_h}_{\text{in}_2}$",  # hub-tip ratio in 2
        r"$\frac{r_t}{r_h}_{\text{out}_1}$",  # hub-tip ratio out 1
        r"$\frac{r_t}{r_h}_{\text{out}_2}$",  # hub-tip ratio out 2
        r"$AR_1$",  # aspect ratio 1
        r"$AR_2$",  # aspect ratio 2
        r"$\frac{p}{c}_1$",  # pitch-chord ratio 1
        r"$\frac{p}{c}_2$",  # pitch-chord ratio 2
        r"$\frac{t_e}{O}_1$",  # trailing edge thickness opening ratio 1
        r"$\frac{t_e}{O}_2$",  # trailing edge thickness opening ratio 2
        r"$\theta_{\text{LE}_1}$",  # leading edge angle 1
        r"$\theta_{\text{LE}_2}$",  # leading edge angle 2
        r"$\beta_{g_1}$",  # gauging angle 1
        r"$\beta_{g_2}$"  # gauging angle 2
    ]


    # Plot relative change of design variables
    sols = [final_slsqp, final_ipopt]
    markers = ["o", "x"]
    labels = ["SLSQP", "IPOPT"]
    fig, ax = tf.plot_functions.plot_optimization_results(initial_slsqp[13:], lower_bounds[13:], upper_bounds[13:], sols, markers=markers, labels = labels, variable_names=variable_names, fontsize = 15)

# plt.show()

