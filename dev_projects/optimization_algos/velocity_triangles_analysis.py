
import numpy as np
import turboflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


save_fig = False

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_SNOPT_tight_2025-02-16_18-12-04.pkl" # New
filename_sga_20_new = "output/pickle_2stage_BB_sga_new_2024-10-31_18-19-51.pkl"
filename_pso_30_new = "output/pickle_2stage_BB_pso_new_2024-11-06_01-08-01.pkl"
filename_slsqp_BB_new = "output/pickle_2stage_BB_slsqp_new_2024-10-30_20-41-31.pkl"
filename_ipopt_BB_new = "output/pickle_2stage_BB_ipopt_new_2024-10-31_03-29-47.pkl"
filename_snopt_BB_new = "output/pickle_2stage_BB_snopt_new_2024-10-31_11-32-52.pkl"
filename_ig = "output/initial_guess_2024-09-20_16-12-35.pkl"


solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)

# Load new files
solver_sga_20_new = tf.load_from_pickle(filename_sga_20_new)
solver_pso_30_new = tf.load_from_pickle(filename_pso_30_new)
solver_slsqp_BB_new = tf.load_from_pickle(filename_slsqp_BB_new)
solver_ipopt_BB_new = tf.load_from_pickle(filename_ipopt_BB_new)
solver_snopt_BB_new = tf.load_from_pickle(filename_snopt_BB_new)

# Load initial guess
initial_guess = tf.load_from_pickle(filename_ig)

# Calculate initial velocity triangle
plt.close()
colors = cm.magma(np.linspace(0.85, 0.25, 8)) 
linestyles = ['-', '--', ':', '-.', '-', '-', '--', ':']

# labels 
labels = ["EO, SLSQP", "EO, IPOPT", "EO, SNOPT", "BB, SLSQP", "BB, IPOPT", "BB, SNOPT", "BB, SGA", "BB, PSO"]

fig, ax = tf.plot_functions.plot_velocity_triangle_stage(initial_guess.results["plane"], linestyle = ':', color = "grey", label = "Baseline")
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_slsqp.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[0], color = colors[0], label = labels[0])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_ipopt.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[1], color = colors[1], label = labels[1])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_snopt.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[2], color = colors[2], label = labels[2])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_slsqp_BB_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[3], color = colors[3], label = labels[3])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_ipopt_BB_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[4], color = colors[4], label = labels[4])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_snopt_BB_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[5], color = colors[5], label = labels[5])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_sga_20_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[6], color = colors[6], label = labels[6])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(solver_pso_30_new.problem.results["plane"], fig = fig, ax = ax, linestyle = linestyles[7], color = colors[7], label = labels[7])

ax.legend(ncol = 4, loc = 'lower left', bbox_to_anchor=(-0.08, -0.5))

if save_fig:
    tf.savefig_in_formats(fig, 'figures/velocity_triangles_new', formats=[".png", ".eps"])

plt.show()