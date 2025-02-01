import os
import turboflow as tf

# Define mode 
MODE = "performance_map"

# Load configuration file
CONFIG_FILE = os.path.abspath("R125_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if MODE == "performance_analysis":
    
    # Compute performance at operation point(s) according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        export_results=False,
        stop_on_failure=True,
    )

    print(solvers[0].problem.results["overall"]["efficiency_ts"])
    
    
elif MODE == "performance_map":

    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, export_results=True)

elif MODE == "design_optimzation":

    # Compute optimal turbine
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=False)
    fig, ax = tf.plot_functions.plot_axial_radial_plane(solver.problem.geometry)
    fig, ax = tf.plot_functions.plot_velocity_triangles_planes(solver.problem.results["plane"])