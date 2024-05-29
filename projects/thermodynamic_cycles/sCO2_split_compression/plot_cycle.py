import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import meanline_axial as ml

import barotropy as bpy

from datetime import datetime



# TODO
# Add capability to export configuration file
# Improve capability to export initial file (now it does not export solver options)
# Plotting is messy, it would be nice to have a function to plot the processes and points only to be used independently of the heavy features for autoplotting and updatting
# Do not plot the phase diagram when plotting points and processes
# Latest file function to code stack, imporve the meanline analysis output structure to folders instead of single files

# Add colorbar
# Add greyscale
# Add white patch to the diagram



# Define configuration filename
# CONFIG_FILE = "case_sCO2_recompression.yaml"

# Find results file


ml.set_plot_options(grid=False)


# Initialize Brayton cycle problem
file_name = "optimal_cycle.yaml"
config = ml.utilities.read_configuration_file(file_name)
thermoCycle = ml.cycles.ThermodynamicCycleProblem(config)
thermoCycle.fitness(thermoCycle.x0)

fig_x = 7
fig_y = 4.0
fig_x = 6
fig_y = 4.0
# fig_y = 12
fig, ax = plt.subplots(figsize=(fig_x, fig_y))
ax.set_xlabel("Entropy (kJ/kg/K)")
ax.set_ylabel("Temperature ($^\circ$ C)")

# thermoCycle.fluid.plot_phase_diagram(axes=ax, plot_saturation_line=True, plot_critical_point=True)







# # ---------------------------------------------------------------------------- #
# # Plot inlet states on T-s diagram
# # ---------------------------------------------------------------------------- #
# # Create figure
# fig, ax = plt.subplots(figsize=(6.8, 4.8))
# ax.set_xlabel("Entropy [kJ/kg/K]")
# ax.set_ylabel("Temperature [$^\circ$C]")
# range_x = np.linspace(1000, 2000, 60)
# range_y = np.linspace(5, 55, 40) + 273.15
# range_z = np.asarray([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]) * 1e5
# ax.set_xlim(np.asarray([range_x[0], range_x[-1]]) / 1e3)
# ax.set_ylim(np.asarray([range_y[0], range_y[-1]]) - 273.15)

# Plot diagram


# Define nice colormap
# colormap = mcolors.LinearSegmentedColormap.from_list(
#     "truncated_blues", plt.cm.Blues(np.linspace(0.4, 1.0, 256))
# )
# colormap = mcolors.LinearSegmentedColormap.from_list(
#     "truncated_blues", plt.cm.Blues(np.linspace(0.0, 1, 256)[::-1])
# )

# Create a grayscale colormap
colormap = mcolors.LinearSegmentedColormap.from_list(
    "custom_gray", [(i, i, i) for i in np.linspace(0.5, 1, 256)]
)


# Create fluid object
fluid_name = "CO2"
fluid = bpy.Fluid(name=fluid_name, backend="HEOS", exceptions=True)


Nx = 50
Ny = 50
rho_1 = fluid.triple_point_vapor.d/20
rho_2 = fluid.critical_point.d/2
rho_3 = fluid.set_state(ml.PT_INPUTS, fluid.p_max, fluid.T_min).d
T_1 = fluid.triple_point_liquid.T
T_2 = fluid.critical_point.T
T_3 = fluid.T_max
range_x1 = np.logspace(np.log10(rho_1), np.log10(rho_2), Nx)
range_x2 = np.linspace(rho_2, rho_3, Nx)
range_x = np.hstack([range_x1, range_x2])
range_y1 = np.linspace(T_1, T_2, Ny)
range_y2 = np.linspace(T_2, T_3, Ny)
range_y = np.hstack([range_y1, range_y2])

# Plot pressure isobars
prop_dict = bpy.compute_properties_meshgrid(fluid, ml.DmassT_INPUTS, range_x, range_y, generalize_quality=False)

contour = ax.contourf(
    prop_dict["s"],
    prop_dict["T"],
    prop_dict["Z"],
    levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    cmap=colormap,
    zorder=1,
)


# Add a colorbar with a label
colorbar = fig.colorbar(contour, ax=ax, label='$Z$ — Compressibility factor')

contour = ax.contour(
    prop_dict["s"],
    prop_dict["T"],
    prop_dict["Z"],
    levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    linewidths=0.25,
    colors="black",
    zorder=1,
)


# plot_config = {"x_variable": "s",
#                "y_variable": "T",
#                "x_scale": "linear",
#                "y_scale": "linear"}


fluid.plot_phase_diagram(x_variable="s", y_variable="T", axes=ax, plot_critical_point=False)

x = fluid.sat_liq["s"] + fluid.sat_vap["s"]
y = fluid.sat_liq["T"] + fluid.sat_vap["T"]
# Fill between plot on top of everything
ax.fill_between(x, y, fluid.sat_liq["T"][0], alpha=1, color="white")

# Plot thermodynamic processes
for name, component in thermoCycle.cycle_data["components"].items():

    # Identify if the component is a heat exchanger and has the desired fluid
    is_heat_exchanger = component["type"] == "heat_exchanger"
    relevant_sides = ["hot_side", "cold_side"] if is_heat_exchanger else [None]

    for side in relevant_sides:
        # Fetch the data depending on whether it's a heat exchanger or not
        if is_heat_exchanger:
            fluid_data = component[side]
        else:
            fluid_data = component

        # Check if the fluid matches and then plot
        color = "#6F1317"
        color = "#203864"
        print(fluid_data["fluid_name"])
        if fluid_data["fluid_name"] == fluid_name:
            x = fluid_data["states"]["s"]
            y = fluid_data["states"]["T"]
            ax.plot(x, y, color=color)
            ax.plot(x[[0, -1]], y[[0, -1]], marker="o", linestyle='None', color=color, markersize=3.5, zorder=4)



ax.set_ylim([-50, 650])

# # Adjust plot limits if updating
# ax.relim(visible_only=True)
# ax.autoscale_view()
# # % Filled countour plot
# # contourf(fluid.s_grid/1e3, fluid.T_grid-273.15, fluid.Z_grid, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 0.8, 0.9, 1.0, 1.1, 1.2], LineStyle='-', LineWidth=0.25, EdgeColor=[0.5, 0.5, 0.5], HandleVisibility='off');

# Set plot scale
bpy.scale_graphics_x(fig, 1/1e3, mode="multiply")
bpy.scale_graphics_y(fig, -273.15, mode="add")

# Adjust plot limits if updating
ax.relim(visible_only=True)
ax.autoscale_view()

fig.tight_layout(pad=1)


ml.savefig_in_formats(fig, f"figures/Ts_diagram_{fig_x:0.1f}_{fig_y:0.1f}")
# Keep figures open
plt.show()






# prop_pair = 



# # Colorbar settings
# sm = plt.cm.ScalarMappable(
#     cmap=colormap,
#     norm=plt.Normalize(vmin=min(range_z), vmax=max(range_z)),
# )
# sm.set_array([])  # This line is necessary for ScalarMappable
# cbar = plt.colorbar(sm, ax=ax)
# cbar.ax.set_ylabel("Pressure [bar]", rotation=90, labelpad=20)
# cbar.set_ticks(range_z)  # Set tick positions
# cbar.set_ticklabels([f"{level/1e5:.0f}" for level in range_z])



















# def get_latest_file(directory, filename):
#     # Ensure the base directory exists
#     if not os.path.exists(directory):
#         return f"Directory {directory} does not exist."
    
#     # Initialize variables to track the latest folder and its timestamp
#     latest_time = None
#     latest_folder = None
    
#     # Iterate through all items in the specified directory
#     for item in os.listdir(directory):
#         # Construct the full path
#         full_path = os.path.join(directory, item)
#         # Check if the path is a directory
#         if os.path.isdir(full_path):
#             # Parse the timestamp from the directory name
#             try:
#                 # Extract the timestamp from the directory name (after "case_")
#                 time_str = item.split("case_")[-1]
#                 # Parse the datetime
#                 folder_time = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
#                 # Update the latest folder if this folder's time is newer
#                 if latest_time is None or folder_time > latest_time:
#                     latest_time = folder_time
#                     latest_folder = full_path
#             except ValueError:
#                 # If the date format is incorrect, ignore this folder
#                 continue
    
#     # If a latest folder was found
#     if latest_folder:
#         # Construct the path to the specified file within the latest folder
#         file_path = os.path.join(latest_folder, filename)
#         # Check if the file exists
#         if os.path.exists(file_path):
#             return file_path
#         else:
#             return f"File {filename} does not exist in the latest directory."
    
#     return "No valid directories found."

