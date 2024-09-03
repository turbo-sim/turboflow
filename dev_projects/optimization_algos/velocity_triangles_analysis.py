
import numpy as np
import turboflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

save_fig = True

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_snopt_2024-08-19_20-45-38.pkl"
filename_sga_10 = "output/pickle_sga_2024-08-21_18-46-08.pkl"
filename_pso_10 = "output/pickle_pso_2024-08-22_15-39-52.pkl"
filename_sga_20 = "output/pickle_sga_2024-08-26_05-00-45.pkl"
filename_pso_20 = "output/pickle_pso_2024-08-23_09-31-51.pkl"
filename_slsqp_BB = "output/pickle_slsqp_BB_2024-08-23_10-49-46.pkl"
filename_snopt_BB = "output/pickle_snopt_BB_2024-08-23_15-33-27.pkl"
filename_ipopt_BB = "output/pickle_ipopt_BB_2024-08-24_01-35-28.pkl"

solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)
# solver_sga_10 = tf.load_from_pickle(filename_sga_10)
# solver_pso_10 = tf.load_from_pickle(filename_pso_10)
solver_sga_20 = tf.load_from_pickle(filename_sga_20)
solver_pso_20 = tf.load_from_pickle(filename_pso_20)
solver_slsqp_BB = tf.load_from_pickle(filename_slsqp_BB)
solver_snopt_BB = tf.load_from_pickle(filename_snopt_BB)
solver_ipopt_BB = tf.load_from_pickle(filename_ipopt_BB)

plt.close()

# Access the tab10 colormap
# cmap = plt.get_cmap('tab10')
# colors = [cmap(i) for i in range(cmap.N)]
colors = sns.color_palette("colorblind", n_colors=8) 


# Linestyles
linestyles = ['-', '--', ':', '-.', '-', '-', '--', ':']

# labels 
labels = ["EO, SLSQP", "EO, IPOPT", "EO, SNOPT", "BB, SLSQP", "BB, IPOPT", "BB, SNOPT", "BB, SGA", "BB, PSO"]

fig1, ax1 = tf.plot_functions.plot_velocity_triangle_stage(solver_slsqp.problem.results["plane"], linestyle = linestyles[0], color = colors[0], label = labels[0])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_ipopt.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[1], color = colors[1], label = labels[1])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_snopt.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[2], color = colors[2], label = labels[2])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_slsqp_BB.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[3], color = colors[3], label = labels[3])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_ipopt_BB.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[4], color = colors[4], label = labels[4])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_snopt_BB.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[5], color = colors[5], label = labels[5])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_sga_20.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[6], color = colors[6], label = labels[6])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_pso_20.problem.results["plane"], fig = fig1, ax = ax1, linestyle = linestyles[7], color = colors[7], label = labels[7])
ax.legend(ncol = 2, loc = 'lower left', bbox_to_anchor=(0.2, -1.65))

if save_fig:
    tf.savefig_in_formats(fig, 'figures/velocity_triangles', formats=[".png", ".eps"])

plt.show()