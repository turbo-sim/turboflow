
import turboflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition

save_figs = False

# Load files 
filename_80 = "output/real_case_80p_2025-02-22_09-00-48.pkl" # 80 % centrifugal stress limit
filename_85 = "output/real_case_85p_2025-02-22_08-59-17.pkl" # 85 % centrifugal stress limit
filename_90 = "output/real_case_90p_2025-02-22_08-57-52.pkl" # 90 % centrifugal stress limit
filename_95 = "output/real_case_95p_2025-02-22_08-56-22.pkl" # 95 % centrifugal stress limit
filename_100 = "output/real_case_100p_2025-02-22_08-54-50.pkl" # 100 % centrifugal stress limit
filename_ig = "output/initial_guess_2024-09-20_16-12-35.pkl" # Initial guess
solver_80 = tf.load_from_pickle(filename_80)
solver_85 = tf.load_from_pickle(filename_85)
solver_90 = tf.load_from_pickle(filename_90)
solver_95 = tf.load_from_pickle(filename_95)
solver_100 = tf.load_from_pickle(filename_100)
initial_guess = tf.load_from_pickle(filename_ig)

# Get initial guess geometry
ig_geometry = initial_guess.geometry.to_dict(orient = "list")
ig_geometry["number_of_cascades"] = 2

# Plot options
colors = plt.get_cmap('magma')(np.linspace(0.1, 0.9, 2))

# Plot axial-radial plane
fig, ax = tf.plot_axial_radial_plane(ig_geometry, label = "Baseline", blade_color = "grey", plot_casing=False)
fig, ax = tf.plot_axial_radial_plane(solver_90.problem.geometry, fig = fig, ax = ax, label = "Optimized", blade_color=colors[0], linestyle = "--", plot_rotation_ax= False, plot_casing=False)
ylims = ax.get_ylim()
ax.legend(handletextpad=0.5, frameon=False, loc = "lower left")
ax.set_ylim([ylims[0], ylims[1]*1.15])

# Plade profile as inset axis
axins = inset_axes(ax, width="100%", height="100%", loc = "lower left")
ip = InsetPosition(ax, [1.2, -0.1, 1.0, 1.0])  
axins.set_axes_locator(ip)
axins.set_aspect('equal', 'box')
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
axins.spines['left'].set_visible(False)
axins.spines['bottom'].set_visible(False)
axins.yaxis.set_ticks([])
axins.xaxis.set_ticks([])
axins.grid(False)
axins.tick_params(axis='x', which='both', top=False, labeltop=False)
axins.tick_params(axis='y', which='both', right=False, labelright=False)
axins.set_xlabel("Axial direction", labelpad=5)
axins.set_ylabel("Tangential direction", labelpad=5)
repeats = 3
fig2, ax2 = tf.plot_view_axial_tangential(initial_guess.results, ig_geometry, fig = fig, ax = axins, repeats= repeats, color = "grey")
fig2, ax2 = tf.plot_view_axial_tangential(solver_90.problem.results, solver_90.problem.geometry, fig = fig2, ax = ax2, repeats= repeats, color = colors[0])
xlims = axins.get_xlim()
ylims = axins.get_ylim()
plt.annotate(
        "",
        xy=(xlims[-1], ylims[0]),
        xytext=(xlims[0], ylims[0]),
        arrowprops=dict(facecolor="black", arrowstyle="-|>"),
    )
plt.annotate(
        "",
        xy=(xlims[0], ylims[1]),
        xytext=(xlims[0], ylims[0]),
        arrowprops=dict(facecolor="black", arrowstyle="-|>"),
    )
def rearrange_arrays(*arrays):
    """
    Rearranges multiple arrays so that all first elements are grouped first,
    followed by all second elements.
    
    :param arrays: Multiple arrays (lists or tuples) with the same length.
    :return: A new flattened list with reordered elements.
    """
    if not arrays:
        return []
    
    # Ensure all arrays have the same length
    length = len(arrays[0])
    if not all(len(arr) == length for arr in arrays):
        raise ValueError("All arrays must have the same length")

    # Rearrange the elements
    result = [arrays[j][i] for i in range(length) for j in range(len(arrays))]
    
    return result

# Print perfromance parameters
results = [initial_guess.results, solver_90.problem.results]
for i, result in zip(range(len(results)), results):
    r = result["stage"]["reaction"] # reaction
    psi = (result["plane"]["v_t"].values[2] - result["plane"]["v_t"].values[3])/result["plane"]["blade_speed"].values[-1]
    phi = result["plane"]["v_m"].values[3]/result["plane"]["blade_speed"].values[-1]
    print("\n")
    print(f"Results {i}")
    print(f"Degree of reaction: {r}")
    print(f"Stage loading coefficient: {psi}")
    print(f"Flow coefficient: {phi}")

initial_fitness = -1*initial_guess.results["overall"]["efficiency_ts"].values[0]
print((solver_90.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100)
print(f"Grad eval: {solver_90.convergence_history['grad_count'][-1]}")
print(f"Model eval: {solver_90.convergence_history['func_count_total'][-1]}")


if save_figs:

    tf.savefig_in_formats(fig, 'figures/geometry_case_study', formats=[".png",".eps"])

plt.show()