import os
import turboflow as tf

# Define output directory
tf.print_banner()
OUT_DIR = "results"
OUT_FILE = "design_optimization"

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

# Solve optimization problem
solver = tf.compute_optimal_turbine(
    config,
    out_dir=OUT_DIR,
    out_filename=OUT_FILE,
    export_results=True,
)

tf.save_to_pickle(solver, filename="multistart_container", out_dir=OUT_DIR)