
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm
import seaborn as sns

print_evals = False
save_fig = True
axins = True

def adjust_objective_values(objective_values, constraint_values, efficiency_values, cut_off = 1.0):

    # Adjust array such that the objective value is strictly deacreasing
    objective_values = objective_values.copy()
    constraint_values = constraint_values.copy()
    efficiency_values = efficiency_values.copy()
    for i in range(1, len(objective_values)):
        if objective_values[i] > objective_values[i-1] or np.isnan(objective_values[i]) :
            objective_values[i] = objective_values[i-1]
            constraint_values[i] = constraint_values[i-1]
            efficiency_values[i] = efficiency_values[i-1]

    # Cut off objective value
    if isinstance(cut_off, float):
        last_val = objective_values[-1]
        for i in range(1,len(objective_values)):
            if objective_values[i] <= cut_off*last_val:
                objective_values = objective_values[:i]
                constraint_values = constraint_values[:i]
                efficiency_values = efficiency_values[:i]
                break

    return objective_values, constraint_values, efficiency_values

# Load object
filename_slsqp = "output/pickle_2stage_slsqp_2024-09-03_22-57-21.pkl"
# filename_slsqp_BB = "output/pickle_2stage_BB_slsqp_2024-09-04_12-17-17.pkl"
# filename_sga = "output/pickle__2stage_sga_2024-08-31_01-38-24.pkl"
filename_slsqp_BB = "output/pickle_2stage_BB_slsqp_new_2024-11-07_12-34-56.pkl"
filename_sga = "output/pickle_2stage_BB_sga_new_2024-11-09_09-50-03.pkl"
filename_ig = "output/initial_guess_2stage_2024-09-20_16-04-06.pkl"

solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_slsqp_BB = tf.load_from_pickle(filename_slsqp_BB)
solver_sga = tf.load_from_pickle(filename_sga)
initial_guess = tf.load_from_pickle(filename_ig)

# Close open figures
plt.close()

# Plot options
# colors = sns.color_palette("colorblind", n_colors=3)  # Adjust n_colors as needed
colors = cm.magma(np.linspace(0.85, 0.25, 3)) 
linestyles = ['-', '--', '-.']
linewidth = 2.5
fontsize = 22

obj_sga, cons_sga, eff_sga = adjust_objective_values(solver_sga.problem.optimization_process.iterations_objective_function[1:], solver_sga.problem.optimization_process.constraint_violation[1:], solver_sga.problem.optimization_process.iterations_efficiency[1:], cut_off=None)

convergence_data = {"EO, SLSQP" : {"objective" : solver_slsqp.convergence_history["objective_value"], 
                        "constraint_violation" : solver_slsqp.convergence_history["constraint_violation"], 
                        "iterations" : solver_slsqp.convergence_history["grad_count"]},
        "BB, SLSQP" : {"objective" : solver_slsqp_BB.convergence_history["objective_value"],
                "constraint_violation" : solver_slsqp_BB.convergence_history["constraint_violation"],
                "iterations" : solver_slsqp_BB.convergence_history["grad_count"]},               
        "BB, SGA" : {"objective" : obj_sga,
                "constraint_violation" : cons_sga,
                "iterations" : range(len(obj_sga))},
                }

figs,  ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=True, fontsize=fontsize, linewidth=linewidth)
# ax[0].set_xscale("log")
# ax[1].set_xscale("log")
ax[0].set_ylim(-100, -40.0)
ax[1].set_ylim(-0.03, 1.0)
ax[0].set_xlim(-100, 3250)
ax[1].set_xlim(-100, 3250)
ax[0].set_xticks([])
ax[0].legend(loc = 'upper left', fontsize = 20)

if axins: 

    bbox = (0.585, 0.4, 0.4, 0.6)

    # Define the region you want to zoom into (e.g., x = 4 to 6)
    x1, x2, y1, y2 = 0, 40, -100, -75

    # Create an inset plot within the main plot
    axins1 = inset_axes(ax[0], width="100%", height="100%", loc='upper right', bbox_to_anchor = bbox, bbox_transform=ax[0].transAxes)

    # Plot the zoomed-in region in the inset plot
    axins1.plot(solver_slsqp.convergence_history["grad_count"], solver_slsqp.convergence_history["objective_value"], color = colors[0], linestyle = linestyles[0], linewidth=linewidth)
    axins1.plot(solver_slsqp_BB.convergence_history["grad_count"], solver_slsqp_BB.convergence_history["objective_value"], color = colors[1], linestyle = linestyles[1], linewidth=linewidth)
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    axins1.tick_params(labelsize=fontsize)

    # Optional: Add a rectangular box around the zoomed region on the main plot
    ax[0].indicate_inset_zoom(axins1)
    # mark_inset(ax[0], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

    # Define the region you want to zoom into (e.g., x = 4 to 6)
    x1, x2, y1, y2 = 0, 40, -0.03, 0.5 

    # Create an inset plot within the main plot
    axins2 = inset_axes(ax[1], width="100%", height="100%", loc='upper right', bbox_to_anchor = bbox, bbox_transform=ax[1].transAxes)

    # Plot the zoomed-in region in the inset plot
    axins2.plot(solver_slsqp.convergence_history["grad_count"], solver_slsqp.convergence_history["constraint_violation"], color = colors[0], linestyle = linestyles[0], linewidth=linewidth)
    axins2.plot(solver_slsqp_BB.convergence_history["grad_count"], solver_slsqp_BB.convergence_history["constraint_violation"], color = colors[1], linestyle = linestyles[1], linewidth=linewidth)
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.tick_params(labelsize=fontsize)

    # Optional: Add a rectangular box around the zoomed region on the main plot
    ax[1].indicate_inset_zoom(axins2)
    # mark_inset(ax[1], axins2, loc1=2, loc2=4, fc="none", ec="0.5")

if save_fig:
    tf.savefig_in_formats(figs[0], 'figures/convergence_rate_2stage_new', formats=[".png", ".eps"])

if print_evals:

    initial_fitness = -1*initial_guess.results["overall"]["efficiency_ts"].values[0]
    initial_efficiency = initial_guess.results["overall"]["efficiency_ts"].values[0]

    # initial_fitness_slsqp = solver_slsqp.convergence_history["objective_value"][0]
    # initial_efficiency_slsqp = -1*solver_slsqp.convergence_history["objective_value"][0]
    print("\n")
    print("EO, SLSQP")
    print(f"Objective value: {solver_slsqp.convergence_history['objective_value'][-1]}")
    print(f"Efficiency: {solver_slsqp.problem.results['overall']['efficiency_ts']}")
    print(f"Fitness improvment: {(solver_slsqp.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
    print(f"Efficiency improvment: {solver_slsqp.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
    print(f"Constraint violation: {solver_slsqp.convergence_history['constraint_violation'][-1]}")
    print(f"Grad eval: {solver_slsqp.convergence_history['grad_count'][-1]}")
    print(f"Func eval: {solver_slsqp.convergence_history['func_count'][-1]}")
    print(f"Model eval: {solver_slsqp.convergence_history['func_count_total'][-1]}")

    # initial_fitness_slsqp_BB = solver_slsqp_BB.convergence_history["objective_value"][0]
    # initial_efficiency_slsqp_BB = -1*solver_slsqp_BB.convergence_history["objective_value"][0]
    print("\n")
    print("BB, SLSQP")
    print(f"Objective value: {solver_slsqp_BB.convergence_history['objective_value'][-1]}")
    print(f"Efficiency: {solver_slsqp_BB.problem.results['overall']['efficiency_ts']}")
    print(f"Fitness improvment: {(solver_slsqp_BB.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
    print(f"Efficiency improvment: {solver_slsqp_BB.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
    print(f"Constraint violation: {solver_slsqp_BB.problem.optimization_process.constraint_violation[-1]}")
    print(f"Grad eval: {solver_slsqp_BB.convergence_history['grad_count'][-1]}")
    print(f"Func eval: {solver_slsqp_BB.convergence_history['func_count'][-1]}")
    print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_slsqp_BB.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")

    # initial_fitness_sga = solver_sga.problem.optimization_process.iterations_objective_function[0]
    # initial_efficiency_sga = solver_sga.problem.optimization_process.iterations_efficiency[0]
    print("\n")
    print("BB, SGA")
    print(f"Objective value: {obj_sga[-1]}")
    print(f"Efficiency: {eff_sga[-1]}")
    print(f"Fitness improvment: {(obj_sga[-1]-initial_fitness)/initial_fitness*100}")
    print(f"Efficiency improvment: {eff_sga[-1] - initial_efficiency}")
    print(f"Constraint violation: {cons_sga[-1]}")
    print(f"Func eval: {solver_sga.convergence_history['func_count'][-1]}")
    print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_sga.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")


plt.show()