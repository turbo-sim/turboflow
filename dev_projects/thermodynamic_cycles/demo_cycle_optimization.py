import turboflow as tf
import matplotlib.pyplot as plt

# Print package info
tf.print_package_info()

# Define configuration filename
CONFIG_FILE = "case_sCO2_recompression.yaml"

# Initialize Brayton cycle problem
cycle = tf.cycles.ThermodynamicCycleOptimization(CONFIG_FILE)

# Perform cycle opetimization
cycle.run_optimization()
cycle.save_results()

# Keep plots open
plt.show()

# optimizer.generate_output_files()
# optimizer.create_animation()
