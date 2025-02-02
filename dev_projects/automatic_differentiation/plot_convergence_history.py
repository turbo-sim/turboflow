
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm
import seaborn as sns
import os

tf.set_plot_options()

algorthims = False
pop_size = True

save_fig = True

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

# Main directory
output_dir = "output"

# Load object
filename_jax_slsqp = os.path.join(output_dir, './case_46/two_stage_config_2024-12-01_14-31-43.pkl')
# filename_FD_3_slsqp = os.path.join(output_dir, './case_17/one_stage_config_latest.pkl')
# filename_FD_6_slsqp = os.path.join(output_dir, './case_18/one_stage_config_latest.pkl')
# filename_FD_9_slsqp = os.path.join(output_dir, './case_19/one_stage_config_latest.pkl')
# filename_FD_12_slsqp = os.path.join(output_dir, './case_20/one_stage_config_latest.pkl')

filename_jax_ipopt = os.path.join(output_dir, './case_51/two_stage_config_2024-12-01_17-52-40.pkl')
# filename_FD_3_ipopt = os.path.join(output_dir, '1stage_FD_1e-3_ipopt.pkl')
# filename_FD_6_ipopt = os.path.join(output_dir, '1stage_FD_1e-6_ipopt.pkl')
# filename_FD_9_ipopt = os.path.join(output_dir, '1stage_FD_1e-9_ipopt.pkl')

filename_jax_snopt = os.path.join(output_dir, './case_56/two_stage_config_2024-12-01_18-39-40.pkl')
# filename_FD_3_snopt = os.path.join(output_dir, '1stage_FD_1e-3_snopt.pkl')
# filename_FD_6_snopt = os.path.join(output_dir, '1stage_FD_1e-6_snopt.pkl')
# filename_FD_9_snopt = os.path.join(output_dir, '1stage_FD_1e-9_snopt.pkl')


solver_jax_slsqp = tf.load_from_dill(filename_jax_slsqp)
# solver_FD_3_slsqp = tf.load_from_dill(filename_FD_3_slsqp)
# solver_FD_6_slsqp = tf.load_from_dill(filename_FD_6_slsqp)
# solver_FD_9_slsqp = tf.load_from_dill(filename_FD_9_slsqp) 
# solver_FD_12_slsqp = tf.load_from_dill(filename_FD_12_slsqp)

solver_jax_ipopt = tf.load_from_dill(filename_jax_ipopt)
# solver_FD_3_ipopt = tf.load_from_dill(filename_FD_3_ipopt)
# solver_FD_6_ipopt = tf.load_from_dill(filename_FD_6_ipopt) 
# solver_FD_9_ipopt = tf.load_from_dill(filename_FD_9_ipopt)

solver_jax_snopt = tf.load_from_dill(filename_jax_snopt)
# solver_FD_3_snopt = tf.load_from_dill(filename_FD_3_snopt)
# solver_FD_6_snopt = tf.load_from_dill(filename_FD_6_snopt) 
# solver_FD_9_snopt = tf.load_from_dill(filename_FD_9_snopt)

# Close open figures
plt.close()

# Access the tab10 colormap
# cmap = plt.get_cmap('tab10')
# colors = [cmap(i) for i in range(cmap.N)]
colors = plt.get_cmap('plasma_r')(np.linspace(0.3, 1.0, 3))
# colors = sns.color_palette("colorblind", n_colors=3)  # Adjust n_colors as needed

# Linestyles
# linestyles = ['-', '--', ':', '-.', '-', '-', '--', ':']
linestyles = ['-', '--', '-.', ':', '--']

def adjust_objective_values(objective_values, constraint_values):

    objective_values = objective_values.copy()
    constraint_values = constraint_values.copy()
    for i in range(1, len(objective_values)):

        if objective_values[i] > objective_values[i-1] or np.isnan(objective_values[i]) :
            objective_values[i] = objective_values[i-1]
            constraint_values[i] = constraint_values[i-1]

    return objective_values, constraint_values

# obj_sga_20, cons_sga_20 = adjust_objective_values(solver_sga_20.convergence_history["objective_value"][:-1], solver_sga_20.problem.optimization_process.constraint_violation[1:])
# obj_pso_20, cons_pso_20 = adjust_objective_values(solver_pso_20.convergence_history["objective_value"][:-1], solver_pso_20.problem.optimization_process.constraint_violation[1:])

# if algorthims:
     
convergence_data = {
    #  "Automatic differentiation" : {"objective" : solver_jax_slsqp.convergence_history["objective_value"], 
    #                             "constraint_violation" : solver_jax_slsqp.convergence_history["constraint_violation"], 
    #                             "iterations" : solver_jax_slsqp.convergence_history["grad_count"]},
    #                 "FD step size = 1e-3" : {"objective" : solver_FD_3_slsqp.convergence_history["objective_value"], 
    #                             "constraint_violation" : solver_FD_3_slsqp.convergence_history["constraint_violation"], 
    #                             "iterations" : solver_FD_3_slsqp.convergence_history["grad_count"]},
    #                 "FD step size = 1e-6" : {"objective" : solver_FD_6_slsqp.convergence_history["objective_value"], 
    #                             "constraint_violation" : solver_FD_6_slsqp.convergence_history["constraint_violation"], 
    #                             "iterations" : solver_FD_6_slsqp.convergence_history["grad_count"]},
    #                 "FD step size = 1e-9" : {"objective" : solver_FD_9_slsqp.convergence_history["objective_value"], 
    #                             "constraint_violation" : solver_FD_9_slsqp.convergence_history["constraint_violation"], 
    #                             "iterations" : solver_FD_9_slsqp.convergence_history["grad_count"]},
    #                 "FD step size = 1e-12" : {"objective" : solver_FD_12_slsqp.convergence_history["objective_value"], 
    #                             "constraint_violation" : solver_FD_12_slsqp.convergence_history["constraint_violation"], 
    #                             "iterations" : solver_FD_12_slsqp.convergence_history["grad_count"]},
                "SLSQP" : {"objective" : solver_jax_slsqp.convergence_history["objective_value"], 
                                "constraint_violation" : solver_jax_slsqp.convergence_history["constraint_violation"], 
                                "iterations" : solver_jax_slsqp.convergence_history["grad_count"]},
                    
                "IPOPT" : {"objective" : solver_jax_ipopt.convergence_history["objective_value"],
                                "constraint_violation" : solver_jax_ipopt.convergence_history["constraint_violation"],
                                "iterations" : solver_jax_ipopt.convergence_history["grad_count"]},
                "SNOPT" : {"objective" : solver_jax_snopt.convergence_history["objective_value"],
                                "constraint_violation" : solver_jax_snopt.convergence_history["constraint_violation"],
                                "iterations" : solver_jax_snopt.convergence_history["grad_count"]}
                }

# Calculate best objective value
obj_values = np.array([])
for key in convergence_data.keys():
        obj_values = np.append(obj_values, convergence_data[key]["objective"][-1])
obj_best = np.min(obj_values)

fig1, fig2,  ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=True)
ax[0].set_ylim(-105, -65)
ax[0].set_xlim(0, 60)
ax[1].set_xlim(0, 60)
ax[1].set_ylim(1e-12, 1e-0)
ax[1].set_yscale("log")

#     ax[0].plot([-100, 2100], [obj_best, obj_best], color = "grey", linestyle = "--")

# Define the region you want to zoom into (e.g., x = 4 to 6)
x1, x2, y1, y2 = 0, 30, -95, -80 

# Create an inset plot within the main plot
# axins1 = inset_axes(ax[0], width="32%", height="32%", loc='center')

# Plot the zoomed-in region in the inset plot
# axins1.plot(solver_jax_slsqp.convergence_history["grad_count"], solver_jax_slsqp.convergence_history["objective_value"], color = colors[0], linestyle = linestyles[0])
# axins1.plot(solver_jax_ipopt.convergence_history["grad_count"], solver_jax_ipopt.convergence_history["objective_value"], color = colors[1], linestyle = linestyles[1])
# axins1.plot(solver_jax_snopt.convergence_history["grad_count"], solver_jax_snopt.convergence_history["objective_value"], color = colors[2], linestyle = linestyles[2])
# axins1.plot(solver_FD_3_slsqp.convergence_history["grad_count"], solver_FD_3_slsqp.convergence_history["objective_value"], color = colors[1], linestyle = linestyles[1])
# axins1.plot(solver_FD_6_slsqp.convergence_history["grad_count"], solver_FD_6_slsqp.convergence_history["objective_value"], color = colors[2], linestyle = linestyles[2])
# axins1.plot(solver_FD_9_slsqp.convergence_history["grad_count"], solver_FD_9_slsqp.convergence_history["objective_value"], color = colors[3], linestyle = linestyles[3])
# axins1.plot(solver_FD_12_slsqp.convergence_history["grad_count"], solver_FD_12_slsqp.convergence_history["objective_value"], color = colors[4], linestyle = linestyles[4])
# axins1.set_xlim(x1, x2)
# axins1.set_ylim(y1, y2)

# Optional: Add a rectangular box around the zoomed region on the main plot
# ax[0].indicate_inset_zoom(axins1)
# mark_inset(ax[0], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Define the region you want to zoom into (e.g., x = 4 to 6)
x1, x2, y1, y2 = 0, 40, 1e-6, 0.100

# # Create an inset plot within the main plot
# axins2 = inset_axes(ax[1], width="40%", height="40%", loc='lower center')

# Plot the zoomed-in region in the inset plot
# axins2.plot(solver_jax_slsqp.convergence_history["grad_count"], solver_jax_slsqp.convergence_history["constraint_violation"], color = colors[0], linestyle = linestyles[0])
# axins2.plot(solver_jax_ipopt.convergence_history["grad_count"], solver_jax_ipopt.convergence_history["constraint_violation"], color = colors[1], linestyle = linestyles[1])
# axins2.plot(solver_jax_snopt.convergence_history["grad_count"], solver_jax_snopt.convergence_history["constraint_violation"], color = colors[2], linestyle = linestyles[2])
# axins2.plot(solver_FD_3_slsqp.convergence_history["grad_count"], solver_FD_3_slsqp.convergence_history["constraint_violation"], color = colors[1], linestyle = linestyles[1])
# axins2.plot(solver_FD_6_slsqp.convergence_history["grad_count"], solver_FD_6_slsqp.convergence_history["constraint_violation"], color = colors[2], linestyle = linestyles[2])
# axins2.plot(solver_FD_9_slsqp.convergence_history["grad_count"], solver_FD_9_slsqp.convergence_history["constraint_violation"], color = colors[3], linestyle = linestyles[3])
# axins2.plot(solver_FD_12_slsqp.convergence_history["grad_count"], solver_FD_12_slsqp.convergence_history["constraint_violation"], color = colors[4], linestyle = linestyles[4])
# axins2.set_xlim(x1, x2)
# axins2.set_ylim(y1, y2)

# Optional: Add a rectangular box around the zoomed region on the main plot
# ax[1].indicate_inset_zoom(axins2)
# mark_inset(ax[1], axins2, loc1=2, loc2=4, fc="none", ec="0.5")


# if save_fig:
tf.savefig_in_formats(fig1, f'{figures_dir}/2stage_solver_comparison_convergence_history_tol1e-6', formats=[".png", ".eps"])
# tf.savefig_in_formats(fig2, f'{figures_dir}/1stage_solver_comparison_convergence_history_constrain_violation_tol1e-6_updated1', formats=[".png", ".eps"])


# if pop_size:

#         filename_sga_10 = "output/pickle_sga_2024-08-25_19-11-11.pkl"
#         filename_pso_10 = "output/pickle_pso_2024-08-22_15-39-52.pkl"
#         filename_sga_30 = "output/pickle_sga_2024-08-24_21-24-46.pkl"
#         filename_pso_30 = "output/pickle_pso_2024-08-25_13-02-32.pkl"
#         solver_sga_10 = tf.load_from_pickle(filename_sga_10)
#         solver_pso_10 = tf.load_from_pickle(filename_pso_10)
#         solver_sga_30 = tf.load_from_pickle(filename_sga_30)
#         solver_pso_30 = tf.load_from_pickle(filename_pso_30)
#         obj_sga_10, cons_sga_10 = adjust_objective_values(solver_sga_10.convergence_history["objective_value"][:-1], solver_sga_10.problem.optimization_process.constraint_violation[1:])
#         obj_pso_10, cons_pso_10 = adjust_objective_values(solver_pso_10.convergence_history["objective_value"][:-1], solver_pso_10.problem.optimization_process.constraint_violation[1:])
#         obj_sga_30, cons_sga_30 = adjust_objective_values(solver_sga_30.convergence_history["objective_value"][:-1], solver_sga_30.problem.optimization_process.constraint_violation[1:])
#         obj_pso_30, cons_pso_30 = adjust_objective_values(solver_pso_30.convergence_history["objective_value"][:-1], solver_pso_30.problem.optimization_process.constraint_violation[1:])


#         convergence_data = {"SGA, 10" : {"objective" : obj_sga_10,
#                                "constraint_violation" : cons_sga_10,
#                                "iterations" : solver_sga_10.convergence_history["func_count"][:-1]},
#                         "SGA, 20" : {"objective" : obj_sga_20,
#                                 "constraint_violation" : cons_sga_20,
#                                 "iterations" : solver_sga_20.convergence_history["func_count"][:-1]},
#                         "SGA, 30" : {"objective" : obj_sga_30,
#                                 "constraint_violation" : cons_sga_30,
#                                 "iterations" : solver_sga_30.convergence_history["func_count"][:-1]},
#                         "PSO, 10" : {"objective" : obj_pso_10,
#                                 "constraint_violation" : cons_pso_10,
#                                 "iterations" : solver_pso_10.convergence_history["func_count"][:-1]},           
#                         "PSO, 20" : {"objective" : obj_pso_20,
#                                 "constraint_violation" : cons_pso_20,
#                                 "iterations" : solver_pso_20.convergence_history["func_count"][:-1]},
#                         "PSO, 30" : {"objective" : obj_pso_30,
#                                 "constraint_violation" : cons_pso_30,
#                                 "iterations" : solver_pso_30.convergence_history["func_count"][:-1]}
#                                 }

#         colors = sns.color_palette("colorblind", n_colors=6)
#         linestyles = ['-', '--', ':', '-.', '-', '-']
#         fig1, fig2, ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=False)

#         ax[0].set_ylim([-100, -40])
#         ax[0].set_xlim([-50, 4000])

#         ax[1].set_ylim([0, 0.5])
#         ax[1].set_xlim([-50, 4000])

#         if save_fig:
#                 tf.savefig_in_formats(fig1, 'figures/pop_size_objective', formats=[".png", ".eps"])
#                 # tf.savefig_in_formats(fig2, 'figures/pop_size_constrain_violation', formats=[".png", ".eps"])

plt.show()