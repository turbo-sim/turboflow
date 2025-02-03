.. _logging:

Export logs and pickling
=====================================================
 
Beside the export function implemented for both :ref:`performance_prediction` and :ref:`design_optimization`, TurboFlow supports
logging the solution process, as well as pickling the output object. This page gives a tutorial of these features.

.. _logging:

Logging
--------

Logging in turboflow tracks key metrics for the solution process for both performance analysis and design optimization. 
Below, we outline what should be logged in each case and provide corresponding code examples.

Logging for Performance Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When analyzing performance, the following key metrics should be logged:

- **Number of function evaluations**: Tracks the number of times the objective function is evaluated.
- **Number of gradient evaluations**: Tracks the number of gradient computations.
- **Norm of residual**: Measures how close the current solution is to satisfying residual equations.
- **Norm of step**: Captures the size of the step taken in an iterative solver.

Logging for performance analysis can be achieved using the following approach:

.. code-block:: python

    import os
    import some_thermal_framework as tf  # Placeholder for actual framework

    # Load configuration file
    CONFIG_FILE = os.path.abspath("one-stage_config.yaml")
    config = tf.load_config(CONFIG_FILE, print_summary=False)

    # Create logger
    logger = tf.create_logger(name="one-stage", path=f"output/logs", use_datetime=True, to_console=True)

    # Compute performance at operation point(s) according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        export_results=False,
        stop_on_failure=True,
        logger=logger,
    )

Logging for Design Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When performing design optimization, the following key metrics should be logged:

- **Number of gradient evaluations**: Tracks the number of gradient computations during optimization.
- **Number of function evaluations**: Counts the number of times the objective function is evaluated.
- **Objective value**: Records the current value of the optimization objective.
- **Constraint violation**: Measures the extent to which constraints are not satisfied.
- **Norm of step**: Captures the magnitude of the step taken in the optimization process.

Logging for design optimization can be achieved using the following approach:

.. code-block:: python

    import os
    import some_thermal_framework as tf  # Placeholder for actual framework

    # Load configuration file
    CONFIG_FILE = os.path.abspath("one-stage_config.yaml")
    config = tf.load_config(CONFIG_FILE, print_summary=False)

    # Create logger
    logger = tf.create_logger(name="one-stage", path=f"output/logs", use_datetime=True, to_console=True)

    # Compute optimal turbine
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=True, logger=logger)
    fig, ax = tf.plot_functions.plot_axial_radial_plane(solver.problem.geometry)
    fig, ax = tf.plot_functions.plot_velocity_triangles_planes(solver.problem.results["plane"])


.. _pickling:

Pickling
---------

Pickling is a method for serializing and saving objects in Python. This allows for storing computation results and reloading them later without recomputation. Below, we outline how to pickle an object and how to load it.

How to Pickle an Object
^^^^^^^^^^^^^^^^^^^^^^^^

To save an object to a pickle file, use the following approach:

.. code-block:: python

    # Save to pickle
    tf.save_to_pickle(solver, filename="pickle_file", path="output/pickle", use_datetime=True)

How to Load a Pickled Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a pickled object, follow this approach:

.. code-block:: python

    # Load pickled file
    filename = "output/pickle/pickle_file.pkl"  # Define filename
    solver = tf.load_from_pickle(filename)  # Load file
    print(solver.problem.results["overall"]["efficiency_ts"])  # Print efficiency from loaded solver object