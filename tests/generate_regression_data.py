import sys
import os
import turboflow as tf
import datetime
import platform

# Check OS type
SUPPORTED_OS = ["Windows", "Linux"]
os_name = platform.system()
if os_name not in SUPPORTED_OS:
    raise Exception("OS not supported")
os_name = os_name.lower()

# Define configuration files for generation of regression data
CONFIG_FILES_PERFORMANCE_ANALYSIS = [
    "performance_analysis_evaluate_cascade_throat.yaml",
    "performance_analysis_evaluate_cascade_critical.yaml",
    "performance_analysis_evaluate_cascade_isentropic_throat.yaml",
    "performance_analysis_kacker_okapuu.yaml",
    "performance_analysis_moustapha.yaml",
    "performance_analysis_zero_deviation.yaml",
    "performance_analysis_ainley_mathieson.yaml",
]
CONFIG_FILES_DESIGN_OPTIMIZATION = ["design_optimization.yaml"]

TEST_FOLDER = "tests"
DATA_DIR = f"regression_data_{os_name.lower()}"
CONFIG_DIR = "config_files"
DATA_DIR = os.path.join(TEST_FOLDER, DATA_DIR)
CONFIG_DIR = os.path.join(TEST_FOLDER, CONFIG_DIR)

new_regression_PA = True
new_regression_DO = True

def create_regression_data_PA(config_file, data_dir, config_dir, os_type):
    """Save performance analysis data to Excel files for regression tests"""

    # Run simulations
    filepath = os.path.join(config_dir, config_file)
    config = tf.load_config(filepath)

    # Get configuration file name without extension
    config_name, _ = os.path.splitext(config_file)
    base_filename = f"{config_name}"
    filename = f"{base_filename}_{os_type}"
    # If file exists, append a timestamp to the new file name
    if os.path.exists(f"{os.path.join(data_dir, filename)}.xlsx"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{base_filename}_{timestamp}"

    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, out_dir = data_dir, out_filename= filename, export_results = True)

    print(f"Regression data set saved: {filename}")
    
def create_regression_data_DO(config_file, data_dir, config_dir, os_type):
    """Save performance analysis data to Excel files for regression tests"""

    # Run simulations
    filepath = os.path.join(config_dir, config_file)
    config = tf.load_config(filepath)

    # Get configuration file name without extension
    config_name, _ = os.path.splitext(config_file)
    base_filename = f"{config_name}"
    filename = f"{base_filename}_{os_type}"
    
    # If file exists, append a timestamp to the new file name
    if os.path.exists(f"{os.path.join(data_dir, filename)}.xlsx"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{base_filename}_{timestamp}"

    solvers = tf.compute_optimal_turbine(config, out_dir = data_dir, out_filename= filename, export_results = True)

    print(f"Regression data set saved: {filename}")

# Create a directory to save regression test data
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
if new_regression_PA:
    for config in CONFIG_FILES_PERFORMANCE_ANALYSIS:
        create_regression_data_PA(config, DATA_DIR, CONFIG_DIR, os_name)
if new_regression_DO:
    for config in CONFIG_FILES_DESIGN_OPTIMIZATION:
        create_regression_data_DO(config, DATA_DIR, CONFIG_DIR, os_name)