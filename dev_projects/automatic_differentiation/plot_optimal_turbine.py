
import numpy as np
import turboflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

save_fig = True

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

# Main directory
output_dir = "output"

# Load object
filename_jax_slsqp = os.path.join(output_dir, '1stage_jax_slsqp.pkl')
filename_FD_3_slsqp = os.path.join(output_dir, '1stage_FD_1e-3_slsqp.pkl')
filename_FD_6_slsqp = os.path.join(output_dir, '1stage_FD_1e-6_slsqp.pkl')
filename_FD_9_slsqp = os.path.join(output_dir, '1stage_FD_1e-9_slsqp.pkl')

filename_FD_3_ipopt = os.path.join(output_dir, '1stage_FD_1e-3_ipopt.pkl')
filename_FD_6_ipopt = os.path.join(output_dir, '1stage_FD_1e-6_ipopt.pkl')
filename_FD_9_ipopt = os.path.join(output_dir, '1stage_FD_1e-9_ipopt.pkl')

filename_FD_3_snopt = os.path.join(output_dir, '1stage_FD_1e-3_snopt.pkl')
filename_FD_6_snopt = os.path.join(output_dir, '1stage_FD_1e-6_snopt.pkl')
filename_FD_9_snopt = os.path.join(output_dir, '1stage_FD_1e-9_snopt.pkl')


solver_jax_slsqp = tf.load_from_dill(filename_jax_slsqp)
solver_FD_3_slsqp = tf.load_from_dill(filename_FD_3_slsqp)
solver_FD_6_slsqp = tf.load_from_dill(filename_FD_6_slsqp) 
solver_FD_9_slsqp = tf.load_from_dill(filename_FD_9_slsqp)

solver_FD_3_ipopt = tf.load_from_dill(filename_FD_3_ipopt)
solver_FD_6_ipopt = tf.load_from_dill(filename_FD_6_ipopt) 
solver_FD_9_ipopt = tf.load_from_dill(filename_FD_9_ipopt)

solver_FD_3_snopt = tf.load_from_dill(filename_FD_3_snopt)
solver_FD_6_snopt = tf.load_from_dill(filename_FD_6_snopt) 
solver_FD_9_snopt = tf.load_from_dill(filename_FD_9_snopt)


plt.close()

# Access the tab10 colormap
# cmap = plt.get_cmap('tab10')
# colors = [cmap(i) for i in range(cmap.N)]
colors = sns.color_palette("colorblind", n_colors=10) 


# Linestyles
linestyles = ['-', '--', ':', '-.', '-', '-', '--', ':', '-', '-']

# labels 
labels = ["jax_SLSQP", "FD_3_SLSQP", "FD_6_SLSQP", "FD_9_SLSQP", "FD_3_IPOPT", "FD_6_IPOPT", "FD_9_IPOPT", "FD_3_SNOPT", "FD_6_SNOPT", "FD_9_SNOPT"]

fig1, ax1 = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_jax_slsqp.problem.results["planes"]), linestyle = linestyles[0], color = colors[0])


fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_3_slsqp.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[1], color = colors[1])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_6_slsqp.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[2], color = colors[2])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_9_slsqp.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[3], color = colors[3])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_3_ipopt.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[4], color = colors[4])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_6_ipopt.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[5], color = colors[5])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_9_ipopt.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[6], color = colors[6])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_3_snopt.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[7], color = colors[7])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_6_snopt.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[8], color = colors[8])
fig, ax = tf.plot_functions.plot_velocity_triangle_stage(pd.DataFrame(solver_FD_9_snopt.problem.results["planes"]), fig = fig1, ax = ax1, linestyle = linestyles[9], color = colors[9])

ax.legend(ncol = 2, loc = 'lower left', bbox_to_anchor=(0.2, -1.65))

if save_fig:
    tf.savefig_in_formats(fig, f'{figures_dir}/velocity_triangles', formats=[".png", ".eps"])

plt.show()