
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm
import seaborn as sns

algorthims = True
save_fig = False

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_SNOPT_tight_2025-02-17_17-11-25.pkl" 
filename_sga_20 = "output/pickle_2stage_BB_sga_new_2024-10-31_18-19-51.pkl"
filename_pso_30 = "output/pickle_2stage_BB_pso_new_2024-11-06_01-08-01.pkl"
filename_slsqp_BB_new = "output/pickle_2stage_BB_slsqp_new_2024-10-30_20-41-31.pkl"
filename_ipopt_BB_new = "output/pickle_2stage_BB_ipopt_new_2024-10-31_03-29-47.pkl"
filename_snopt_BB_new = "output/pickle_2stage_BB_snopt_new_2024-10-31_11-32-52.pkl"

solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)
solver_sga_20 = tf.load_from_pickle(filename_sga_20)
solver_pso_30 = tf.load_from_pickle(filename_pso_30)
solver_slsqp_BB_new = tf.load_from_pickle(filename_slsqp_BB_new)
solver_ipopt_BB_new = tf.load_from_pickle(filename_ipopt_BB_new)
solver_snopt_BB_new = tf.load_from_pickle(filename_snopt_BB_new)

# Close open figures
plt.close()

# Plot options
colors = cm.magma(np.linspace(0.85, 0.25, 8)) 
linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5)),  (0, (3, 1, 1, 1, 1, 1)), '-', '--']
linewidth = 2.5
fontsize = 22

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

# Adjust gradent-free results 
obj_sga_20, cons_sga_20 = adjust_objective_values(solver_sga_20.convergence_history["objective_value"][:-1], solver_sga_20.problem.optimization_process.constraint_violation[1:], cut_off=1.0)
obj_pso_30, cons_pso_30 = adjust_objective_values(solver_pso_30.convergence_history["objective_value"][:-1], solver_pso_30.problem.optimization_process.constraint_violation[1:], cut_off=1.0)

# Construct convergence data dictionary
convergence_data = {"EO, SLSQP" : {"objective" : solver_slsqp.convergence_history["objective_value"], 
                                "constraint_violation" : solver_slsqp.convergence_history["constraint_violation"], 
                                "iterations" : solver_slsqp.convergence_history["grad_count"]},
                "EO, IPOPT" : {"objective" : solver_ipopt.convergence_history["objective_value"],
                                "constraint_violation" : solver_ipopt.convergence_history["constraint_violation"],
                                "iterations" : solver_ipopt.convergence_history["grad_count"]},
                "EO, SNOPT" : {"objective" : solver_snopt.convergence_history["objective_value"],
                                "constraint_violation" : solver_snopt.convergence_history["constraint_violation"],
                                "iterations" : solver_snopt.convergence_history["grad_count"]},
                "BB, SLSQP" : {"objective" : solver_slsqp_BB_new.convergence_history["objective_value"],
                        "constraint_violation" : solver_slsqp_BB_new.convergence_history["constraint_violation"],
                        "iterations" : solver_slsqp_BB_new.convergence_history["grad_count"]},
                "BB, IPOPT" : {"objective" : solver_ipopt_BB_new.convergence_history["objective_value"],
                        "constraint_violation" : solver_ipopt_BB_new.convergence_history["constraint_violation"],
                        "iterations" : solver_ipopt_BB_new.convergence_history["grad_count"]},
                "BB, SNOPT" : {"objective" : solver_snopt_BB_new.convergence_history["objective_value"],
                        "constraint_violation" : solver_snopt_BB_new.convergence_history["constraint_violation"],
                        "iterations" : solver_snopt_BB_new.convergence_history["grad_count"]},
                "BB, SGA" : {"objective" : obj_sga_20,
                        "constraint_violation" : cons_sga_20,
                        "iterations" : range(len(obj_sga_20))},
                "BB, PSO" : {"objective" : obj_pso_30,
                        "constraint_violation" : cons_pso_30,
                        "iterations" : range(len(obj_pso_30))},
                        }


figs, ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=True, fontsize=fontsize, linewidth=linewidth)
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_ylim(-100, -60)
ax[1].set_ylim(-0.05, 0.85)
ax[0].set_xlim(-100, 4000)
ax[1].set_xlim(-100, 4000)
ax[0].set_xticks([])
ax[0].set_yticks([-100, -90, -80, -70, -60])
ax[1].legend(loc = 'upper right', ncols = 2, fontsize = 20, handlelength=2.75)
ax[0].legend().remove()
plt.tight_layout()

if save_fig:
        tf.savefig_in_formats(figs[0], 'figures/convergence_rate_1stage_new', formats=[".png", ".eps"])

       
plt.show()