
import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition

save_figs = False
calculate_performance_parameters = True

# Load solver containes
filename = "output/pickle_multistart_slsqp_2024-09-01_02-03-20.pkl"
solver_container = tf.load_from_pickle(filename)
solvers = solver_container.solver_container

# Get initial guess and bounds
initial_guess = solvers[0].convergence_history["x"][0]
initial_efficiency = solvers[0].convergence_history["objective_value"][0]
lower_bounds = solvers[0].problem.bounds[0]
upper_bounds = solvers[0].problem.bounds[1]

# Define variable names
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

# Filter the succesful solvers
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

# Kill horrible solutions
kill_solutions = [2]
for i in kill_solutions:
    del solver_container[i]
    del sols[i]
    objective_functions = np.delete(objective_functions, i)

# Define plot options
markers = ["o", "x", ">", "<", "^", "v", "s", "*"]*8
colors = plt.get_cmap('magma')(np.linspace(0.1, 0.9, len(solver_container)))
marker_sizes = [7]*len(solver_container)
fontsize = 18

# Sort solutions from least to most efficiency upgrade
combined = list(zip(objective_functions, sols))
combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
sorted_array, sorted_sols = zip(*combined_sorted)

# Plot efficiency upgrade for each solution 
fig1, ax1 = plt.subplots()
for i in range(len(sorted_array)):
    ax1.plot([i], [(sorted_array[i]-initial_efficiency)*-1], color = colors[i], marker = markers[i], linestyle = 'none', markersize = marker_sizes[i])

# PLot change in design variables
fig, ax = tf.plot_functions.plot_optimization_results(initial_guess[13:], lower_bounds[13:], upper_bounds[13:], sorted_sols, variable_names=variable_names, markers = markers, colors=colors, markersizes=marker_sizes, fontsize = fontsize)
ax.set_ylim([-1.0, 1.0])
ax.tick_params(labelsize=fontsize)
ax.legend().set_visible(False)
ax1.set_xlabel('Optimizations', fontsize = fontsize)
ax1.set_ylabel('Efficiency improvement [pp]', fontsize = fontsize)
plt.tight_layout()

# Calcaulate geometry for three selected solutions (unfortunately, the problem object for each sovler object was updated during the mutli-start optimization, so every problem is the same)
selected_solutions = [0, 25, -1] # Selected solutions
geoms = [] # Geometry list
solv_alt = [initial_guess[13:], sols[selected_solutions[0]], sols[selected_solutions[1]], sols[selected_solutions[2]]] # Final x for selected solutions
effs = [(sorted_array[selected_solutions[0]]-initial_efficiency)*-1, (sorted_array[selected_solutions[1]]-initial_efficiency)*-1, (sorted_array[selected_solutions[2]]-initial_efficiency)*-1] # Efficiency upgrade for selected solutions
h0_in = solvers[0].problem.boundary_conditions["h0_in"] # Load boundary conditions
mass_flow = solvers[0].problem.reference_values["mass_flow_ref"] # Load boundary conditions
h_is = solvers[0].problem.reference_values["h_out_s"] # Load boundary conditions
d_is = solvers[0].problem.reference_values["d_out_s"] # Load boundary conditions
v0 = solvers[0].problem.reference_values["v0"] # Load boundary conditions
angle_range = solvers[0].problem.reference_values["angle_range"] # Load boundary conditions
angle_min = solvers[0].problem.reference_values["angle_min"] # Load boundary conditions
radius_type = "constant_mean"
for dv in solv_alt:
    geom = {
        "hub_tip_ratio_in" : np.array([dv[2], dv[3]]),
        "hub_tip_ratio_out" : np.array([dv[4], dv[5]]),
        "aspect_ratio" : np.array([dv[6], dv[7]]),
        "pitch_chord_ratio" : np.array([dv[8], dv[9]]),
        "trailing_edge_thickness_opening_ratio" : np.array([dv[10], dv[11]]),
        "leading_edge_angle" : np.array([dv[12], dv[13]])* angle_range + angle_min,
        "gauging_angle" : np.array([dv[14], dv[15]])* angle_range + angle_min,
        "throat_location_fraction" : np.array([5/6, 5/6]),
        "leading_edge_diameter" : np.array([2*0.127e-2, 2*0.081e-2]), 
        "leading_edge_wedge_angle" : np.array([50.0, 50.0]),
        "tip_clearance" : np.array([0.00, 0.030e-2]), 
        "cascade_type" : ["stator", "rotor"],
    }
    omega = solvers[0].problem.get_omega(dv[0], mass_flow, h0_in, d_is, h_is) # Calculate angular speed
    blade_speed = dv[1] * v0 # Calculate blade speed
    geom["radius"] = blade_speed / omega # Calculate radius
    geom = tf.axial_turbine.geometry_model.prepare_geometry(geom, radius_type) # Calculate geometry
    geom = tf.axial_turbine.geometry_model.calculate_full_geometry(geom) # Calculate full geometry
    geoms.append(geom)

# Plot axial-radial plane
fig2, ax2 = tf.plot_axial_radial_plane(geoms[0], label = "Baseline", blade_color = "grey", plot_casing=False)
fig2, ax2 = tf.plot_axial_radial_plane(geoms[1], fig = fig2, ax = ax2, label = f"$\Delta \eta = {effs[0]:.2f}$", blade_color = colors[selected_solutions[0]], plot_casing=False, plot_rotation_ax=False)
fig2, ax2 = tf.plot_axial_radial_plane(geoms[2], fig = fig2, ax = ax2, blade_color = colors[selected_solutions[1]], label = f"$\Delta \eta = {effs[1]:.2f}$", plot_casing=False, plot_rotation_ax=False)
fig2, ax2 = tf.plot_axial_radial_plane(geoms[3], fig = fig2, ax = ax2, blade_color = colors[selected_solutions[2]], label = f"$\Delta \eta = {effs[2]:.2f}$", plot_casing=False, plot_rotation_ax=False)
ylims = ax2.get_ylim()
ax2.legend(handletextpad=0.5, frameon=False, loc = "lower left")
ax2.set_ylim([ylims[0], ylims[1]*1.15])
ax2.set_ylabel("Radius", labelpad=7)

# Plot blade profile as inset axes
axins1 = inset_axes(ax2, width="100%", height="100%", loc = "lower left")#, bbox_to_anchor=(0.0, 0.0, 1.0, 1.0)) #, bbox_transform=ax[0].transAxes)
ip = InsetPosition(ax2, [1.2, -0.1, 1.0, 1.0])  
axins1.set_axes_locator(ip)
axins1.set_aspect('equal', 'box')
axins1.spines['top'].set_visible(False)
axins1.spines['right'].set_visible(False)
axins1.spines['left'].set_visible(False)
axins1.spines['bottom'].set_visible(False)
axins1.yaxis.set_ticks([])
axins1.xaxis.set_ticks([])
axins1.grid(False)
axins1.tick_params(axis='x', which='both', top=False, labeltop=False)
axins1.tick_params(axis='y', which='both', right=False, labelright=False)
axins1.set_xlabel("Axial direction", labelpad=5)
axins1.set_ylabel("Tangential direction", labelpad=5)
turbine_data = {"stage" : {"reaction" : pd.Series([0.8])}}
repeats = 3
fig2, ax2 = tf.plot_view_axial_tangential(turbine_data, geoms[0], fig = fig2, ax = axins1, repeats= repeats, color = "grey")
fig2, ax2 = tf.plot_view_axial_tangential(turbine_data, geoms[1], fig = fig2, ax = ax2, repeats= repeats, color = colors[selected_solutions[0]])
fig2, ax2 = tf.plot_view_axial_tangential(turbine_data, geoms[2], fig = fig2, ax = ax2, repeats= repeats, color = colors[selected_solutions[1]])
fig2, ax2 = tf.plot_view_axial_tangential(turbine_data, geoms[3], fig = fig2, ax = ax2, repeats= repeats, color = colors[selected_solutions[2]])
xlims = axins1.get_xlim()
ylims = axins1.get_ylim()
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
    ) # Arrow

# Calculate selected 
if calculate_performance_parameters:
    for j,i in zip(range(len(selected_solutions)), selected_solutions):
        solver_container[i].problem.fitness_gradient_based(solver_container[i].convergence_history["x"][-1])
        results = solver_container[i].problem.results
        r = results["stage"]["reaction"].values[0] # reaction
        psi = (results["plane"]["v_t"].values[2] - results["plane"]["v_t"].values[3])/results["plane"]["blade_speed"].values[-1]
        phi = results["plane"]["v_m"].values[3]/results["plane"]["blade_speed"].values[-1]
        stress = results["cascade"]["centrifugal_stress"].values[1]

        print("\n")
        print(f"Efficiency upgrade: {effs[j]}")
        print(f"Degree of reaction: {r}")
        print(f"Stage loading coefficient: {psi}")
        print(f"Flow coefficient: {phi}")
        print(f"Centrifugal stress: {stress}")

    # Print initial guess
    solver_container[0].problem.fitness_gradient_based(initial_guess)
    results = solver_container[0].problem.results
    r = results["stage"]["reaction"].values[0] # reaction
    psi = (results["plane"]["v_t"].values[2] - results["plane"]["v_t"].values[3])/results["plane"]["blade_speed"].values[-1]
    phi = results["plane"]["v_m"].values[3]/results["plane"]["blade_speed"].values[-1]
    stress = results["cascade"]["centrifugal_stress"].values[1]
    print("\n")
    print("Initial guess")
    print(f"Degree of reaction: {r}")
    print(f"Stage loading coefficient: {psi}")
    print(f"Flow coefficient: {phi}")
    print(f"Centrifugal stress: {stress}")

if save_figs:

    # tf.savefig_in_formats(fig, 'figures/multistart_design_variables', formats=[".png",".eps"])
    # tf.savefig_in_formats(fig1, 'figures/multistart_objective_values', formats=[".png",".eps"])
    tf.savefig_in_formats(fig2, 'figures/geometry', formats=[".png",".eps"])

plt.show()
