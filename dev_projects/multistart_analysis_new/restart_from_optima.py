import os
import turboflow as tf

# Define output directory
tf.print_banner()
OUT_DIR = "results"
IN_FILE = "design_optimization"
OUT_FILE = "design_optimization_restart"

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)
config["design_optimization"]["multistarts"] = 0

# Walk through all subdirectories
for root, dirs, files in os.walk(OUT_DIR):
    for file in files:
        if file == f"{IN_FILE}.pkl":

            # Load solver object
            solver = tf.load_from_pickle(os.path.join(root, file))

            # Extract final solution
            x_final = solver.x_final

            # Rerun optimization from x_final
            solver = tf.compute_optimal_turbine(
                config,
                out_dir=OUT_DIR,
                out_filename=OUT_FILE,
                export_results=True,
                x0=x_final
            )

