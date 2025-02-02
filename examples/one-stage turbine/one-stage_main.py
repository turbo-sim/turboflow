import os
import turboflow as tf

# Define mode 
MODE = "design_optimzation"

# Load configuration file
CONFIG_FILE = os.path.abspath("one-stage_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

# Create logger
logger = tf.create_logger(name="one-stage", path=f"output/logs", use_datetime=True, to_console=True)

if MODE == "performance_analysis":
    
    # Compute performance at operation point(s) according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        export_results=False,
        stop_on_failure=True,
        logger = logger,
    )

    # Print turbine efficiency
    print(solvers[0].problem.results["overall"]["efficiency_ts"])

    # Plot turbine geometry and velocity triangles
    fig, ax = tf.plot_functions.plot_axial_radial_plane(solvers[0].problem.geometry)
    fig, ax = tf.plot_functions.plot_velocity_triangles_planes(solvers[0].problem.results["plane"])

elif MODE == "performance_map":

    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, export_results=True, logger = logger)

elif MODE == "design_optimzation":

    # Compute optimal turbine
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=True, logger = logger)
    fig, ax = tf.plot_functions.plot_axial_radial_plane(solver.problem.geometry)
    fig, ax = tf.plot_functions.plot_velocity_triangles_planes(solver.problem.results["plane"])
