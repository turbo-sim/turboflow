
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

save_figs = True
mark_all = True


filename = "output/pickle_multistart_slsqp_2024-09-01_02-03-20.pkl"
solver_container = tf.load_from_pickle(filename)

solvers = solver_container.solver_container

initial_guess = solvers[0].convergence_history["x"][0]
initial_efficiency = solvers[0].convergence_history["objective_value"][0]
lower_bounds = solvers[0].problem.bounds[0]
upper_bounds = solvers[0].problem.bounds[1]

# variable_names = [
#     r"$\Omega_s$",  # specific speed
#     r"$\nu$",  # blade jet ratio
#     r"$r_{\mathrm{ht, in}_s}$", # hub-tip ratio in 1
#     r"$r_{\mathrm{ht, in}_r}$",  # hub-tip ratio in 2
#     r"$r_{\mathrm{ht, out}_s}$",  # hub-tip ratio out 1
#     r"$r_{\mathrm{ht, out}_r}$",  # hub-tip ratio out 2
#     r"$\frac{H}{c}_s$",  # aspect ratio 1
#     r"$\frac{H}{c}_r$",  # aspect ratio 2
#     r"$\frac{s}{c}_s$",  # pitch-chord ratio 1
#     r"$\frac{s}{c}_r$",  # pitch-chord ratio 2
#     r"$\frac{t_\mathrm{te}}{o}_s$",  # trailing edge thickness opening ratio 1
#     r"$\frac{t_\mathrm{te}}{o}_r$",  # trailing edge thickness opening ratio 2
#     r"$\theta_{\text{le}_s}$",  # leading edge angle 1
#     r"$\theta_{\text{le}_r}$",  # leading edge angle 2
#     r"$\beta_{g_s}$",  # gauging angle 1
#     r"$\beta_{g_r}$"  # gauging angle 2
# ]

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

fontsize = 18

def get_array(index, solver_container):

    """ 
    return array of specified design variable from solver_container
    - i: index correspnodong to the design variable wanted
    """

    arr = np.array([])
    for solver in solver_container:

        final_solution = solver.convergence_history["x"][-1][13:]
        arr = np.append(arr, final_solution[index])

    return arr

def find_clusters(arr, err = 0.01):

    """
    Returns array of indices with cluster
    """

    clusters = {"cluster_1" : {"values" : [arr[0]], "indices" : [0,]}}

    for i, val in zip(range(len(arr[1:])), arr[1:]):
        success = False
        for key in clusters.keys():
            if (clusters[key]["values"][0]*(1-err) < val) and (clusters[key]["values"][0]*(1+err) > val):
                clusters[key]["values"].append(val)
                clusters[key]["indices"].append(i+1)
                
                success = True
                break
        if not success:   
            clusters[f"cluster_{len(clusters)+1}"] = {"values" : [val], "indices" : [i+1]}

    return clusters

objective_functions = np.array([])
successes = 0
sols = []
solver_container = []
func_count_total = 0
for solver in solvers:
    func_count_total += solver.convergence_history["func_count_total"][-1]
    if solver.success:
        sols.append(solver.convergence_history["x"][-1][13:])
        solver_container.append(solver)
        objective_functions = np.append(objective_functions, solver.convergence_history["objective_value"][-1])

print(solver_container[36].convergence_history["objective_value"][-1])
print(solver_container[36].convergence_history["x"][0])
print(solver_container[36].problem.design_variables_keys)
dv = dict(zip(solver_container[36].problem.design_variables_keys, solver_container[36].convergence_history["x"][0]))
for key, val in dv.items():
    print(f"{key}: {val}")

kill_solutions = [2]
for i in kill_solutions:
    del solver_container[i]
    del sols[i]
    objective_functions = np.delete(objective_functions, i)

variable = 13
arr = get_array(variable, solver_container)
clusters = find_clusters(arr, err = 0.05)

# Set colors
colors_base = sns.color_palette("colorblind", n_colors=8)
markers_base =["o", "x", ">", "<", "^", "v", "s", "*"]
colors = [colors_base[0]]*len(solver_container)
markers = [markers_base[0]]*len(solver_container)
indices = [clusters[key]["indices"] for key in clusters.keys()]

for index, i in zip(indices, range(len(indices))):
    for j in index:
        colors[j] = colors_base[i+1]
        markers[j] = markers_base[i+1]

if mark_all:
    # markers = ["o", "x", ">", "<", "^", "v", "s", "*"]*8
    # colors = plt.get_cmap('magma')(np.linspace(0.1, 0.9, len(solver_container)))
    # fig, ax = tf.plot_functions.plot_optimization_results(initial_guess[13:], lower_bounds[13:], upper_bounds[13:], sols, variable_names=variable_names, markers = markers, colors=colors, fontsize = 15)
    # ax.set_ylim([-1.0, 1.0])
    # # ax.legend(ncols = 5, bbox_to_anchor = (0.23, 0.6, 0.5, 0.5))
    # ax.legend().set_visible(False)
    # # plt.tight_layout()

    # # Step 1: Combine the lists using zip
    # combined = list(zip(objective_functions, colors, markers))

    # # Step 2: Sort the combined list by the array elements, in descending order
    # combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)

    # # Step 3: Unzip the sorted list into separate lists
    # sorted_array, sorted_colors, sorted_markers = zip(*combined_sorted)

    # fig1, ax1 = plt.subplots()
    # sorted_values = np.sort(objective_functions)
    
    # for i in range(len(sorted_array)):
    #     ax1.plot([i], [sorted_array[i]], color = sorted_colors[i], marker = sorted_markers[i], linestyle = 'none', markersize = 6)

    # ax1.set_xlabel('Optimizations')
    # ax1.set_ylabel('Objective Value')
    # ax1.set_ylim([-91.0, -90.7])
    # ax1.set_title(f"Clusters based on: {variable_names[variable]}")

    # for solver in solver_container:
    #     # fig, ax = tf.plot_axial_radial_plane(solver.problem.geometry)
    #     print(solver.problem.geometry["flaring_angle"])

    markers = ["o", "x", ">", "<", "^", "v", "s", "*"]*8
    colors = plt.get_cmap('magma')(np.linspace(0.1, 0.9, len(solver_container)))
    # marker_sizes = np.linspace(12,5,len(solver_container))
    marker_sizes = [7]*len(solver_container)

    # Step 1: Combine the lists using zip
    combined = list(zip(objective_functions, sols))

    # Step 2: Sort the combined list by the array elements, in descending order
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)

    # Step 3: Unzip the sorted list into separate lists
    sorted_array, sorted_sols = zip(*combined_sorted)
    fig1, ax1 = plt.subplots()
    for i in range(len(sorted_array)):
        ax1.plot([i], [(sorted_array[i]-initial_efficiency)*-1], color = colors[i], marker = markers[i], linestyle = 'none', markersize = marker_sizes[i])
    # for i, (x_val, y_val) in enumerate(zip(range(len(sorted_array)), sorted_array), 1):
    #     ax1.text(x_val, y_val, str(i), fontsize=12, color=colors[i-1], ha='center', va='center')


    fig, ax = tf.plot_functions.plot_optimization_results(initial_guess[13:], lower_bounds[13:], upper_bounds[13:], sorted_sols, variable_names=variable_names, markers = markers, colors=colors, markersizes=marker_sizes, fontsize = fontsize)
    ax.set_ylim([-1.0, 1.0])
    ax.tick_params(labelsize=fontsize)
    # ax.legend(ncols = 5, bbox_to_anchor = (0.23, 0.6, 0.5, 0.5))
    ax.legend().set_visible(False)

    ax1.set_xlabel('Optimizations', fontsize = fontsize)
    ax1.set_ylabel('Efficiency improvement [pp]', fontsize = fontsize)
    # ax1.set_ylim([-91.0, -90.7])
    # ax1.set_title(f"Clusters based on: {variable_names[variable]}")
    plt.tight_layout()

else:
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

    print(f"Min: {sorted_values[-1]}")
    print(f"Second min: {sorted_values[-2]}")
    print(f"Max: {sorted_values[0]}")

    print('\n')
    print(f"Total model evaluations: {func_count_total}")

if save_figs:

    tf.savefig_in_formats(fig, 'figures/multistart_design_variables', formats=[".png",".eps"])
    # tf.savefig_in_formats(fig1, 'figures/multistart_objective_values', formats=[".png",".eps"])

plt.show()

