
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm
import seaborn as sns

algorthims = True
save_fig = True

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_snopt_2024-08-19_20-45-38.pkl"
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
# figs[0].subplots_adjust(hspace=0.3)
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_ylim(-100, -60)
ax[1].set_ylim(-0.05, 0.85)
ax[0].set_xlim(-100, 4000)
ax[1].set_xlim(-100, 4000)
ax[0].set_xticks([])
ax[0].set_yticks([-100, -90, -80, -70, -60])
# ax[0].legend(loc = 'upper left', ncols = 4, bbox_to_anchor=(0.15, 0.8, 0.5, 0.5), fontsize = 16)
ax[1].legend(loc = 'upper right', ncols = 2, fontsize = 20, handlelength=2.75)
ax[0].legend().remove()
plt.tight_layout()

# if axins:
#         # Define the region you want to zoom into (e.g., x = 4 to 6)
#         x1, x2, y1, y2 = -2, 75, -92.5, -87.5 

#         # Create an inset plot within the main plot
#         axins1 = inset_axes(ax[0], width="50%", height="60%", loc = 'upper right')#, bbox_to_anchor=(-0.075, -0.01, 1.0, 1.0), bbox_transform=ax[0].transAxes)

#         # Plot the zoomed-in region in the inset plot
#         axins1.plot(solver_slsqp.convergence_history["grad_count"], solver_slsqp.convergence_history["objective_value"], color = colors[0], linestyle = linestyles[0], linewidth = linewidth)
#         axins1.plot(solver_ipopt.convergence_history["grad_count"], solver_ipopt.convergence_history["objective_value"], color = colors[1], linestyle = linestyles[1], linewidth = linewidth)
#         axins1.plot(solver_snopt.convergence_history["grad_count"], solver_snopt.convergence_history["objective_value"], color = colors[2], linestyle = linestyles[2], linewidth = linewidth)
#         # axins1.plot(solver_slsqp_BB.convergence_history["grad_count"], solver_slsqp_BB.convergence_history["objective_value"], color = colors[3], linestyle = linestyles[3], linewidth = linewidth)
#         # axins1.plot(solver_ipopt_BB.convergence_history["grad_count"], solver_ipopt_BB.convergence_history["objective_value"], color = colors[4], linestyle = linestyles[4], linewidth = linewidth)
#         # axins1.plot(solver_snopt_BB.convergence_history["grad_count"], solver_snopt_BB.convergence_history["objective_value"], color = colors[5], linestyle = linestyles[5], linewidth = linewidth)
#         axins1.plot(solver_slsqp_BB_new.convergence_history["grad_count"], solver_slsqp_BB_new.convergence_history["objective_value"], color = colors[3], linestyle = linestyles[3], linewidth = linewidth)
#         axins1.plot(solver_ipopt_BB_new.convergence_history["grad_count"], solver_ipopt_BB_new.convergence_history["objective_value"], color = colors[4], linestyle = linestyles[4], linewidth = linewidth)
#         axins1.plot(solver_snopt_BB_new.convergence_history["grad_count"], solver_snopt_BB_new.convergence_history["objective_value"], color = colors[5], linestyle = linestyles[5], linewidth = linewidth)
#         axins1.set_xlim(x1, x2)
#         axins1.set_ylim(y1, y2)
#         axins1.tick_params(labelsize=fontsize)

#         # Optional: Add a rectangular box around the zoomed region on the main plot
#         ax[0].indicate_inset_zoom(axins1)
#         # mark_inset(ax[0], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

#         # Define the region you want to zoom into (e.g., x = 4 to 6)
#         x1, x2, y1, y2 = -2, 75, -0.01, 0.075 

#         # Create an inset plot within the main plot
#         axins2 = inset_axes(ax[1], width="50%", height="60%", loc='upper right')#, bbox_to_anchor=(-0.075, -0.01, 1.0, 1.0), bbox_transform=ax[1].transAxes)

#         # Plot the zoomed-in region in the inset plot
#         axins2.plot(solver_slsqp.convergence_history["grad_count"], solver_slsqp.convergence_history["constraint_violation"], color = colors[0], linestyle = linestyles[0], linewidth = linewidth)
#         axins2.plot(solver_ipopt.convergence_history["grad_count"], solver_ipopt.convergence_history["constraint_violation"], color = colors[1], linestyle = linestyles[1], linewidth = linewidth)
#         axins2.plot(solver_snopt.convergence_history["grad_count"], solver_snopt.convergence_history["constraint_violation"], color = colors[2], linestyle = linestyles[2], linewidth = linewidth)
#         # axins2.plot(solver_slsqp_BB.convergence_history["grad_count"], solver_slsqp_BB.convergence_history["constraint_violation"], color = colors[3], linestyle = linestyles[3], linewidth = linewidth)
#         # axins2.plot(solver_ipopt_BB.convergence_history["grad_count"], solver_ipopt_BB.convergence_history["constraint_violation"], color = colors[4], linestyle = linestyles[4], linewidth = linewidth)
#         # axins2.plot(solver_snopt_BB.convergence_history["grad_count"], solver_snopt_BB.convergence_history["constraint_violation"], color = colors[5], linestyle = linestyles[5], linewidth = linewidth)
#         axins2.plot(solver_slsqp_BB_new.convergence_history["grad_count"], solver_slsqp_BB_new.convergence_history["constraint_violation"], color = colors[3], linestyle = linestyles[3], linewidth = linewidth)
#         axins2.plot(solver_ipopt_BB_new.convergence_history["grad_count"], solver_ipopt_BB_new.convergence_history["constraint_violation"], color = colors[4], linestyle = linestyles[4], linewidth = linewidth)
#         axins2.plot(solver_snopt_BB_new.convergence_history["grad_count"], solver_snopt_BB_new.convergence_history["constraint_violation"], color = colors[5], linestyle = linestyles[5], linewidth = linewidth)
#         axins2.set_xlim(x1, x2)
#         axins2.set_ylim(y1, y2)
#         axins2.tick_params(labelsize=fontsize)

#         # Optional: Add a rectangular box around the zoomed region on the main plot
#         ax[1].indicate_inset_zoom(axins2)
#         # mark_inset(ax[1], axins2, loc1=2, loc2=4, fc="none", ec="0.5")

if save_fig:
        tf.savefig_in_formats(figs[0], 'figures/convergence_rate_1stage_new', formats=[".png", ".eps"])

       

plt.show()