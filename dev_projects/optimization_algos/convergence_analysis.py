
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm
import seaborn as sns

algorthims = False
pop_size = True
save_fig = True

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_snopt_2024-08-19_20-45-38.pkl"
filename_sga_20 = "output/pickle_sga_2024-08-26_05-00-45.pkl"
filename_pso_20 = "output/pickle_pso_2024-08-23_09-31-51.pkl"
filename_slsqp_BB = "output/pickle_slsqp_BB_2024-08-23_10-49-46.pkl"
filename_snopt_BB = "output/pickle_snopt_BB_2024-08-23_15-33-27.pkl"
filename_ipopt_BB = "output/pickle_ipopt_BB_2024-08-24_01-35-28.pkl"

solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)
solver_sga_20 = tf.load_from_pickle(filename_sga_20)
solver_pso_20 = tf.load_from_pickle(filename_pso_20)
solver_slsqp_BB = tf.load_from_pickle(filename_slsqp_BB)
solver_snopt_BB = tf.load_from_pickle(filename_snopt_BB)
solver_ipopt_BB = tf.load_from_pickle(filename_ipopt_BB)

# Close open figures
plt.close()

# Access the tab10 colormap
# cmap = plt.get_cmap('tab10')
# colors = [cmap(i) for i in range(cmap.N)]
# colors = plt.get_cmap('magma')(np.linspace(0.2, 1.0, 7))
colors = sns.color_palette("colorblind", n_colors=8)  # Adjust n_colors as needed


# Linestyles
linestyles = ['-', '--', ':', '-.', '-', '-', '--', ':']

def adjust_objective_values(objective_values, constraint_values):

    objective_values = objective_values.copy()
    constraint_values = constraint_values.copy()
    for i in range(1, len(objective_values)):

        if objective_values[i] > objective_values[i-1] or np.isnan(objective_values[i]) :
            objective_values[i] = objective_values[i-1]
            constraint_values[i] = constraint_values[i-1]

    return objective_values, constraint_values

obj_sga_20, cons_sga_20 = adjust_objective_values(solver_sga_20.convergence_history["objective_value"][:-1], solver_sga_20.problem.optimization_process.constraint_violation[1:])
obj_pso_20, cons_pso_20 = adjust_objective_values(solver_pso_20.convergence_history["objective_value"][:-1], solver_pso_20.problem.optimization_process.constraint_violation[1:])

if algorthims:
     
        convergence_data = {"EO, SLSQP" : {"objective" : solver_slsqp.convergence_history["objective_value"], 
                                        "constraint_violation" : solver_slsqp.convergence_history["constraint_violation"], 
                                        "iterations" : solver_slsqp.convergence_history["grad_count"]},
                        "EO, IPOPT" : {"objective" : solver_ipopt.convergence_history["objective_value"],
                                        "constraint_violation" : solver_ipopt.convergence_history["constraint_violation"],
                                        "iterations" : solver_ipopt.convergence_history["grad_count"]},
                        "EO, SNOPT" : {"objective" : solver_snopt.convergence_history["objective_value"],
                                        "constraint_violation" : solver_snopt.convergence_history["constraint_violation"],
                                        "iterations" : solver_snopt.convergence_history["grad_count"]},
                        "BB, SLSQP" : {"objective" : solver_slsqp_BB.convergence_history["objective_value"],
                                "constraint_violation" : solver_slsqp_BB.convergence_history["constraint_violation"],
                                "iterations" : solver_slsqp_BB.convergence_history["grad_count"]},
                        "BB, IPOPT" : {"objective" : solver_ipopt_BB.convergence_history["objective_value"],
                                "constraint_violation" : solver_ipopt_BB.convergence_history["constraint_violation"],
                                "iterations" : solver_ipopt_BB.convergence_history["grad_count"]},  
                        "BB, SNOPT" : {"objective" : solver_snopt_BB.convergence_history["objective_value"],
                                "constraint_violation" : solver_snopt_BB.convergence_history["constraint_violation"],
                                "iterations" : solver_snopt_BB.convergence_history["grad_count"]},                 
                        "BB, SGA" : {"objective" : obj_sga_20,
                                "constraint_violation" : cons_sga_20,
                                "iterations" : solver_sga_20.convergence_history["func_count"][:-1]},
                        "BB, PSO" : {"objective" : obj_pso_20,
                                "constraint_violation" : cons_pso_20,
                                "iterations" : solver_pso_20.convergence_history["func_count"][:-1]}}

        # Calculate best objective value
        obj_values = np.array([])
        for key in convergence_data.keys():
                obj_values = np.append(obj_values, convergence_data[key]["objective"][-1])
        obj_best = np.min(obj_values)

        fig1, fig2,  ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=False)
        ax[0].set_ylim(-100, 0)
        ax[0].set_xlim(-50, 2050)
        #     ax[0].plot([-100, 2100], [obj_best, obj_best], color = "grey", linestyle = "--")

        # Define the region you want to zoom into (e.g., x = 4 to 6)
        x1, x2, y1, y2 = 0, 50, -95, -65 

        # Create an inset plot within the main plot
        axins1 = inset_axes(ax[0], width="60%", height="60%", loc='upper center')

        # Plot the zoomed-in region in the inset plot
        axins1.plot(solver_slsqp.convergence_history["grad_count"], solver_slsqp.convergence_history["objective_value"], color = colors[0], linestyle = linestyles[0])
        axins1.plot(solver_ipopt.convergence_history["grad_count"], solver_ipopt.convergence_history["objective_value"], color = colors[1], linestyle = linestyles[1])
        axins1.plot(solver_snopt.convergence_history["grad_count"], solver_snopt.convergence_history["objective_value"], color = colors[2], linestyle = linestyles[2])
        axins1.plot(solver_slsqp_BB.convergence_history["grad_count"], solver_slsqp_BB.convergence_history["objective_value"], color = colors[3], linestyle = linestyles[3])
        axins1.plot(solver_ipopt_BB.convergence_history["grad_count"], solver_ipopt_BB.convergence_history["objective_value"], color = colors[4], linestyle = linestyles[4])
        axins1.plot(solver_snopt_BB.convergence_history["grad_count"], solver_snopt_BB.convergence_history["objective_value"], color = colors[5], linestyle = linestyles[5])
        axins1.set_xlim(x1, x2)
        axins1.set_ylim(y1, y2)

        # Optional: Add a rectangular box around the zoomed region on the main plot
        ax[0].indicate_inset_zoom(axins1)
        mark_inset(ax[0], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

        # Define the region you want to zoom into (e.g., x = 4 to 6)
        x1, x2, y1, y2 = 0, 50, -0.05, 0.5 

        # Create an inset plot within the main plot
        axins2 = inset_axes(ax[1], width="60%", height="60%", loc='upper center')

        # Plot the zoomed-in region in the inset plot
        axins2.plot(solver_slsqp.convergence_history["grad_count"], solver_slsqp.convergence_history["constraint_violation"], color = colors[0], linestyle = linestyles[0])
        axins2.plot(solver_ipopt.convergence_history["grad_count"], solver_ipopt.convergence_history["constraint_violation"], color = colors[1], linestyle = linestyles[1])
        axins2.plot(solver_snopt.convergence_history["grad_count"], solver_snopt.convergence_history["constraint_violation"], color = colors[2], linestyle = linestyles[2])
        axins2.plot(solver_slsqp_BB.convergence_history["grad_count"], solver_slsqp_BB.convergence_history["constraint_violation"], color = colors[3], linestyle = linestyles[3])
        axins2.plot(solver_ipopt_BB.convergence_history["grad_count"], solver_ipopt_BB.convergence_history["constraint_violation"], color = colors[4], linestyle = linestyles[4])
        axins2.plot(solver_snopt_BB.convergence_history["grad_count"], solver_snopt_BB.convergence_history["constraint_violation"], color = colors[5], linestyle = linestyles[5])
        axins2.set_xlim(x1, x2)
        axins2.set_ylim(y1, y2)

        # Optional: Add a rectangular box around the zoomed region on the main plot
        ax[1].indicate_inset_zoom(axins2)
        mark_inset(ax[1], axins2, loc1=2, loc2=4, fc="none", ec="0.5")

        if save_fig:
                tf.savefig_in_formats(fig1, 'figures/convergence_rate_objective', formats=[".png", ".eps"])
                tf.savefig_in_formats(fig2, 'figures/convergence_rate_constrain_violation', formats=[".png", ".eps"])


if pop_size:

        filename_sga_10 = "output/pickle_sga_2024-08-25_19-11-11.pkl"
        filename_pso_10 = "output/pickle_pso_2024-08-22_15-39-52.pkl"
        filename_sga_30 = "output/pickle_sga_2024-08-24_21-24-46.pkl"
        filename_pso_30 = "output/pickle_pso_2024-08-25_13-02-32.pkl"
        solver_sga_10 = tf.load_from_pickle(filename_sga_10)
        solver_pso_10 = tf.load_from_pickle(filename_pso_10)
        solver_sga_30 = tf.load_from_pickle(filename_sga_30)
        solver_pso_30 = tf.load_from_pickle(filename_pso_30)
        obj_sga_10, cons_sga_10 = adjust_objective_values(solver_sga_10.convergence_history["objective_value"][:-1], solver_sga_10.problem.optimization_process.constraint_violation[1:])
        obj_pso_10, cons_pso_10 = adjust_objective_values(solver_pso_10.convergence_history["objective_value"][:-1], solver_pso_10.problem.optimization_process.constraint_violation[1:])
        obj_sga_30, cons_sga_30 = adjust_objective_values(solver_sga_30.convergence_history["objective_value"][:-1], solver_sga_30.problem.optimization_process.constraint_violation[1:])
        obj_pso_30, cons_pso_30 = adjust_objective_values(solver_pso_30.convergence_history["objective_value"][:-1], solver_pso_30.problem.optimization_process.constraint_violation[1:])


        convergence_data = {"SGA, 10" : {"objective" : obj_sga_10,
                               "constraint_violation" : cons_sga_10,
                               "iterations" : solver_sga_10.convergence_history["func_count"][:-1]},
                        "SGA, 20" : {"objective" : obj_sga_20,
                                "constraint_violation" : cons_sga_20,
                                "iterations" : solver_sga_20.convergence_history["func_count"][:-1]},
                        "SGA, 30" : {"objective" : obj_sga_30,
                                "constraint_violation" : cons_sga_30,
                                "iterations" : solver_sga_30.convergence_history["func_count"][:-1]},
                        "PSO, 10" : {"objective" : obj_pso_10,
                                "constraint_violation" : cons_pso_10,
                                "iterations" : solver_pso_10.convergence_history["func_count"][:-1]},           
                        "PSO, 20" : {"objective" : obj_pso_20,
                                "constraint_violation" : cons_pso_20,
                                "iterations" : solver_pso_20.convergence_history["func_count"][:-1]},
                        "PSO, 30" : {"objective" : obj_pso_30,
                                "constraint_violation" : cons_pso_30,
                                "iterations" : solver_pso_30.convergence_history["func_count"][:-1]}
                                }

        colors = sns.color_palette("colorblind", n_colors=6)
        linestyles = ['-', '--', ':', '-.', '-', '-']
        fig1, fig2, ax = tf.plot_functions.plot_convergence_process(convergence_data, colors=colors, linestyles=linestyles, plot_together=False)

        ax[0].set_ylim([-100, -40])
        ax[0].set_xlim([-50, 4000])

        ax[1].set_ylim([0, 0.5])
        ax[1].set_xlim([-50, 4000])

        if save_fig:
                tf.savefig_in_formats(fig1, 'figures/pop_size_objective', formats=[".png", ".eps"])
                # tf.savefig_in_formats(fig2, 'figures/pop_size_constrain_violation', formats=[".png", ".eps"])

plt.show()