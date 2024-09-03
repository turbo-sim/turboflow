
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
filename_sga_10 = "output/pickle_sga_2024-08-21_18-46-08.pkl"
filename_pso_10 = "output/pickle_pso_2024-08-22_15-39-52.pkl"
filename_sga_20 = "output/pickle_sga_2024-08-22_01-46-23.pkl"
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

plt.close()

# Access the tab10 colormap
# cmap = plt.get_cmap('tab10')
# colors = [cmap(i) for i in range(cmap.N)]
colors = sns.color_palette("colorblind", n_colors=8) 

# Linestyles
linestyles = ['-', '--', ':', '-.', (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (5, 1)), (0, (3, 1, 1, 1))]

# Markers 
markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']

initial_slsqp = solver_slsqp.convergence_history["x"][0]
final_slsqp = solver_slsqp.convergence_history["x"][-1][13:]
final_ipopt = solver_ipopt.convergence_history["x"][-1][13:]
final_snopt = solver_snopt.convergence_history["x"][-1][13:]
final_sga = solver_sga_20.convergence_history["x"][-1]
final_pso = solver_pso_20.convergence_history["x"][-1]
final_slsqp_BB = solver_slsqp_BB.convergence_history["x"][-1]
final_ipopt_BB = solver_ipopt_BB.convergence_history["x"][-1]
final_snopt_BB = solver_snopt_BB.convergence_history["x"][-1]
lower_bounds = solver_slsqp.problem.bounds[0]
upper_bounds = solver_slsqp.problem.bounds[1]
variable_names = [
    r"$\omega_s$",  # specific speed
    r"$\nu$",  # blade jet ratio
    r"$\frac{r_h}{r_t}_{\text{in}_1}$",  # hub-tip ratio in 1
    r"$\frac{r_h}{r_t}_{\text{in}_2}$",  # hub-tip ratio in 2
    r"$\frac{r_h}{r_t}_{\text{out}_1}$",  # hub-tip ratio out 1
    r"$\frac{r_h}{r_t}_{\text{out}_2}$",  # hub-tip ratio out 2
    r"$AR_1$",  # aspect ratio 1
    r"$AR_2$",  # aspect ratio 2
    r"$\frac{p}{c}_1$",  # pitch-chord ratio 1
    r"$\frac{p}{c}_2$",  # pitch-chord ratio 2
    r"$\frac{t_e}{O}_1$",  # trailing edge thickness opening ratio 1
    r"$\frac{t_e}{O}_2$",  # trailing edge thickness opening ratio 2
    r"$\theta_{\text{LE}_1}$",  # leading edge angle 1
    r"$\theta_{\text{LE}_2}$",  # leading edge angle 2
    r"$\beta_{g_1}$",  # gauging angle 1
    r"$\beta_{g_2}$"  # gauging angle 2
]


# Plot relative change of design variables
sols = [final_slsqp, final_ipopt, final_snopt, final_slsqp_BB, final_ipopt_BB, final_snopt_BB, final_sga, final_pso]
labels = ["EO, SLSQP", "EO, IPOPT", "EO, SNOPT", "BB, SLSQP", "BB, IPOPT", "BB, SNOPT", "BB, SGA", "BB, PSO"]
fig, ax = tf.plot_functions.plot_optimization_results(initial_slsqp[13:], lower_bounds[13:], upper_bounds[13:], sols, markers=markers, labels = labels, variable_names=variable_names, fontsize = 15)

if save_fig:
    tf.savefig_in_formats(fig, 'figures/change_design_variables', formats=[".png", ".eps"])

plt.show()
