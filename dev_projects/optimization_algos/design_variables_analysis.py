
import turboflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cm as cm
import seaborn as sns

save_fig = True

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_snopt_2024-08-19_20-45-38.pkl"
filename_sga_20_new = "output/pickle_2stage_BB_sga_new_2024-10-31_18-19-51.pkl"
filename_pso_30_new = "output/pickle_2stage_BB_pso_new_2024-11-06_01-08-01.pkl"
filename_slsqp_BB_new = "output/pickle_2stage_BB_slsqp_new_2024-10-30_20-41-31.pkl"
filename_ipopt_BB_new = "output/pickle_2stage_BB_ipopt_new_2024-10-31_03-29-47.pkl"
filename_snopt_BB_new = "output/pickle_2stage_BB_snopt_new_2024-10-31_11-32-52.pkl"

solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)
solver_sga_20_new = tf.load_from_pickle(filename_sga_20_new)
solver_pso_30_new = tf.load_from_pickle(filename_pso_30_new)
solver_slsqp_BB_new = tf.load_from_pickle(filename_slsqp_BB_new)
solver_ipopt_BB_new = tf.load_from_pickle(filename_ipopt_BB_new)
solver_snopt_BB_new = tf.load_from_pickle(filename_snopt_BB_new)

plt.close()

# Access the tab10 colormap
# colors = sns.color_palette("colorblind", n_colors=8) 
colors = cm.magma(np.linspace(0.85, 0.25, 8)) 

# Markers 
markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']

initial_slsqp = solver_slsqp.convergence_history["x"][0]
final_slsqp = solver_slsqp.convergence_history["x"][-1][13:]
final_ipopt = solver_ipopt.convergence_history["x"][-1][13:]
final_snopt = solver_snopt.convergence_history["x"][-1][13:]
final_sga_new = solver_sga_20_new.convergence_history["x"][-1]
final_pso_new = solver_pso_30_new.convergence_history["x"][-1]
final_slsqp_BB_new = solver_slsqp_BB_new.convergence_history["x"][-1]
final_ipopt_BB_new = solver_ipopt_BB_new.convergence_history["x"][-1]
final_snopt_BB_new = solver_snopt_BB_new.convergence_history["x"][-1]

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


# Plot relative change of design variables
labels = ["EO, SLSQP", "EO, IPOPT", "EO, SNOPT", "BB, SLSQP", "BB, IPOPT", "BB, SNOPT", "BB, SGA", "BB, PSO"]
sols = [final_slsqp, final_ipopt, final_snopt, final_slsqp_BB_new, final_ipopt_BB_new, final_snopt_BB_new, final_sga_new, final_pso_new]
fig, ax = tf.plot_functions.plot_optimization_results(initial_slsqp[13:], lower_bounds[13:], upper_bounds[13:], sols, colors=colors, markers=markers, labels = labels, variable_names=variable_names, fontsize = 18)
ax.legend(ncols = 2, fontsize = 12, loc = 'lower left')
ax.set_ylim([-1.0, 1.0])

if save_fig:
    tf.savefig_in_formats(fig, 'figures/change_design_variables_new', formats=[".png", ".eps"])

plt.show()
