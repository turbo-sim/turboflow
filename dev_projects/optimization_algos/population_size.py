
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm


analysis = "convergence_analysis" # convergence_analysis, design_variables, velocity_triangles
save_fig = True

filename_sga_10_new = "output/pickle_2stage_BB_sga_new_2024-10-30_15-51-35.pkl"
filename_sga_20_new = "output/pickle_2stage_BB_sga_new_2024-10-31_18-19-51.pkl"
filename_sga_30_new = "output/pickle_2stage_BB_sga_new_2024-11-01_08-35-42.pkl"
filename_pso_10_new = "output/pickle_2stage_BB_pso_new_2024-11-01_17-44-22.pkl"
filename_pso_20_new = "output/pickle_2stage_BB_pso_new_2024-11-05_09-03-00.pkl"
filename_pso_30_new = "output/pickle_2stage_BB_pso_new_2024-11-06_01-08-01.pkl"

solver_sga_10_new = tf.load_from_pickle(filename_sga_10_new)
solver_sga_20_new = tf.load_from_pickle(filename_sga_20_new)
solver_sga_30_new = tf.load_from_pickle(filename_sga_30_new)
solver_pso_10_new = tf.load_from_pickle(filename_pso_10_new)
solver_pso_20_new = tf.load_from_pickle(filename_pso_20_new)
solver_pso_30_new = tf.load_from_pickle(filename_pso_30_new)
plt.close()


def adjust_objective_values(objective_values, constraint_values, cut_off = None):

    # Adjust array such that the objective value is strictly deacreasing
    objective_values = objective_values.copy()
    constraint_values = constraint_values.copy()
    for i in range(1, len(objective_values)):
        if objective_values[i] > objective_values[i-1] or np.isnan(objective_values[i]) :
            objective_values[i] = objective_values[i-1]
            constraint_values[i] = constraint_values[i-1]

    # Cut off objective value
    if isinstance(cut_off, float):
        last_val = objective_values[-1]
        for i in range(1,len(objective_values)):
                if objective_values[i] <= cut_off*last_val:
                        objective_values = objective_values[:i]
                        constraint_values = constraint_values[:i]
                        break

    return objective_values, constraint_values

if analysis == "convergence_analysis":
    

    obj_sga_10, cons_sga_10 = adjust_objective_values(solver_sga_10_new.convergence_history["objective_value"][:-1], solver_sga_10_new.problem.optimization_process.constraint_violation[1:])
    obj_pso_10, cons_pso_10 = adjust_objective_values(solver_pso_10_new.convergence_history["objective_value"][:-1], solver_pso_10_new.problem.optimization_process.constraint_violation[1:])
    obj_sga_20, cons_sga_20 = adjust_objective_values(solver_sga_20_new.convergence_history["objective_value"][:-1], solver_sga_20_new.problem.optimization_process.constraint_violation[1:])
    obj_pso_20, cons_pso_20 = adjust_objective_values(solver_pso_20_new.convergence_history["objective_value"][:-1], solver_pso_20_new.problem.optimization_process.constraint_violation[1:])
    obj_sga_30, cons_sga_30 = adjust_objective_values(solver_sga_30_new.convergence_history["objective_value"][:-1], solver_sga_30_new.problem.optimization_process.constraint_violation[1:])
    obj_pso_30, cons_pso_30 = adjust_objective_values(solver_pso_30_new.convergence_history["objective_value"][:-1], solver_pso_30_new.problem.optimization_process.constraint_violation[1:])


    convergence_data = {"SGA, 10" : {"objective" : obj_sga_10,
                            "constraint_violation" : cons_sga_10,
                            "iterations" : range(len(obj_sga_10))},
                    "SGA, 20" : {"objective" : obj_sga_20,
                            "constraint_violation" : cons_sga_20,
                            "iterations" : range(len(obj_sga_20))},
                    "SGA, 30" : {"objective" : obj_sga_30,
                            "constraint_violation" : cons_sga_30,
                            "iterations" : range(len(obj_sga_30))},
                    "PSO, 10" : {"objective" : obj_pso_10,
                            "constraint_violation" : cons_pso_10,
                            "iterations" : range(len(obj_pso_10))},           
                    "PSO, 20" : {"objective" : obj_pso_20,
                            "constraint_violation" : cons_pso_20,
                            "iterations" : range(len(obj_pso_20))},
                    "PSO, 30" : {"objective" : obj_pso_30,
                            "constraint_violation" : cons_pso_30,
                            "iterations" : range(len(obj_pso_30))}
                            }

    colors = cm.magma(np.linspace(0.85, 0.25, 6)) 
    linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5)),  (0, (3, 1, 1, 1, 1, 1))]
    linewidth = 2.5
    fontsize = 22
    figs, ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=True, fontsize=fontsize, linewidth=linewidth)

    ax[0].set_ylim([-100, -60])
    ax[0].set_xlim([-50, 4000])
    ax[1].set_ylim([-0.02, 0.39])
    ax[1].set_xlim([-50, 4000])
    ax[0].legend(loc = 'upper right', fontsize = 20, handlelength=3)
    ax[0].set_yticks([-100, -90, -80, -70, -60])

    if save_fig:
                tf.savefig_in_formats(figs[0], 'figures/pop_size_analysis_new', formats=[".png", ".eps"])

elif analysis == "design_variables":
     
    filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
    solver_slsqp = tf.load_from_pickle(filename_slsqp)
    plt.close()
    initial_slsqp = solver_slsqp.convergence_history["x"][0]
    final_slsqp = solver_slsqp.convergence_history["x"][-1][13:]
    final_sga_10 = solver_sga_10_new.convergence_history["x"][-1]
    final_pso_10 = solver_pso_10_new.convergence_history["x"][-1]
    final_sga_20 = solver_sga_20_new.convergence_history["x"][-1]
    final_pso_20 = solver_pso_20_new.convergence_history["x"][-1]
    final_sga_30 = solver_sga_30_new.convergence_history["x"][-1]
    final_pso_30 = solver_pso_30_new.convergence_history["x"][-1]

    
    lower_bounds = solver_slsqp.problem.bounds[0]
    upper_bounds = solver_slsqp.problem.bounds[1]
    variable_names = [
        r"$\Omega_s$",  # specific speed
        r"$\lambda$",  # blade jet ratio
        r"$r_{\mathrm{ht, in}_\text{s}}$", # hub-tip ratio in 1
        r"$r_{\mathrm{ht, in}_\text{r}}$",  # hub-tip ratio in 2
        r"$r_{\mathrm{ht, out}_\text{s}}$",  # hub-tip ratio out 1
        r"$r_{\mathrm{ht, out}_\text{r}}$",  # hub-tip ratio out 2
        r"$H/c_\text{s}$",  # aspect ratio 1
        r"$H/c_\text{r}$",  # aspect ratio 2
        r"$s/c_\text{s}$",  # pitch-chord ratio 1
        r"$s/c_\text{r}$",  # pitch-chord ratio 2
        r"$t_\mathrm{te}/o_\text{s}$",  # trailing edge thickness opening ratio 1
        r"$t_\mathrm{te}/o_\text{r}$",  # trailing edge thickness opening ratio 2
        r"$\theta_{\text{in}_\text{s}}$",  # leading edge angle 1
        r"$\theta_{\text{in}_\text{r}}$",  # leading edge angle 2
        r"$\theta_{\text{out}_\text{s}}$",  # gauging angle 1
        r"$\theta_{\text{out}_\text{r}}$"  # gauging angle 2
    ]

    labels = ["SGA, 10", "PSO, 10", "SGA, 20", "PSO, 20", "SGA, 30", "PSO, 30", "SLSQP",]
    labels = ["SGA, 20", "SGA, 30","SGA, 10", "SLSQP",]
    sols = [final_sga_10, final_pso_10, final_sga_20, final_pso_20, final_sga_30, final_pso_30, final_slsqp]
    sols = [final_sga_20, final_sga_30, final_sga_10, final_slsqp]
    colors = cm.magma(np.linspace(0.85, 0.25, 7))
    markers = ['*','o', 's', '^', 'v', 'D', 'x']
    markersizes = np.array([10, 8, 6, 4])
    fig, ax = tf.plot_functions.plot_optimization_results(initial_slsqp[13:], lower_bounds[13:], upper_bounds[13:], sols, colors=colors, markers=markers, markersizes=markersizes, labels = labels, variable_names=variable_names, fontsize = 18)
    ax.legend(ncols = 2, fontsize = 12, loc = 'lower left')
    ax.set_ylim([-1.0, 1.0])

elif analysis == "velocity_triangle":
     
    filename_ig = "output/initial_guess_2024-09-20_16-12-35.pkl"
    initial_guess = tf.load_from_pickle(filename_ig)

    labels = ["SGA, 10", "PSO, 10", "SGA, 20", "PSO, 20", "SGA, 30", "PSO, 30"]
    linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5)),  (0, (3, 1, 1, 1, 1, 1))]
    colors = cm.magma(np.linspace(0.85, 0.25, 6)) 

    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(initial_guess.results["plane"], linestyle = ':', color = "grey", label = "Baseline")
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_sga_10_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[0], color = colors[0], label = labels[0])
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_pso_10_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[1], color = colors[1], label = labels[1])
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_sga_20_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[2], color = colors[2], label = labels[2])
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_pso_20_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[3], color = colors[3], label = labels[3])
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_sga_30_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[4], color = colors[4], label = labels[4])
    fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_pso_30_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[5], color = colors[5], label = labels[5])
    ax.legend(ncol = 4, bbox_to_anchor=(1.0, -0.5))


plt.show()
