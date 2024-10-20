
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

save_figs = True

filename = "output/pickle_multistart_slsqp_2024-09-01_02-03-20.pkl"
solver_container = tf.load_from_pickle(filename)

solvers = solver_container.solver_container

initial_guess = solvers[0].convergence_history["x"][0]
lower_bounds = solvers[0].problem.bounds[0]
upper_bounds = solvers[0].problem.bounds[1]

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

objective_functions = np.array([])
successes = 0
sols = []
func_count_total = 0
for solver in solvers:
    func_count_total += solver.convergence_history["func_count_total"][-1]
    if solver.success:
        sols.append(solver.convergence_history["x"][-1][13:])
        objective_functions = np.append(objective_functions, solver.convergence_history["objective_value"][-1])


colors = sns.color_palette("colorblind", n_colors=8)
markers = ['o', 'o', 'x', 'o', 'o', 's'] + ['o']*17 + ['*', 'o', '^'] + ['o']*50
colors_fixed = [colors[0], colors[0], colors[1], colors[0], colors[0], colors[6]] + [colors[0]]*17 + [colors[3], colors[0], colors[4]] + [colors[0]]*50
fig, ax = tf.plot_functions.plot_optimization_results(initial_guess[13:], lower_bounds[13:], upper_bounds[13:], sols, variable_names=variable_names, markers = markers, colors=colors_fixed, fontsize = 15)
ax.legend().set_visible(False)
# ax.legend(ncols = 5)

# Step 1: Combine the lists using zip
combined = list(zip(objective_functions, colors_fixed, markers))

# Step 2: Sort the combined list by the array elements, in descending order
combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)

# Step 3: Unzip the sorted list into separate lists
sorted_array, sorted_colors, sorted_markers = zip(*combined_sorted)

fig1, ax1 = plt.subplots()
sorted_values = np.sort(objective_functions)
for i in range(len(sorted_array)):
    ax1.plot([i], [sorted_array[i]], color = sorted_colors[i], marker = sorted_markers[i], linestyle = 'none', markersize = 6)

ax1.set_xlabel('Optimizations')
ax1.set_ylabel('Objective Value')
ax1.set_ylim([-91.0, -88.3])

if save_figs:

    tf.savefig_in_formats(fig, 'figures/multistart_design_variables', formats=[".png",".eps"])
    tf.savefig_in_formats(fig1, 'figures/multistart_objective_values', formats=[".png",".eps"])


print(f"Min: {sorted_values[-1]}")
print(f"Second min: {sorted_values[-2]}")
print(f"Max: {sorted_values[0]}")

print('\n')
print(f"Total model evaluations: {func_count_total}")

plt.show()

