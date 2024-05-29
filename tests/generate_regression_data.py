import sys
import os
import turboflow as tf
import datetime
import platform

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
DATA_DIR = "regression_data"
CONFIG_DIR = "config_files"

new_regression_PA = True
new_regression_DO = False

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
    
def get_os_type():
    os_name = platform.system()
    if os_name == 'Windows':
        return 'win'
    elif os_name == 'Linux':
        return 'linux'
    else:
        return 'unknown'
    
# Get OS type
OS_TYPE = get_os_type()

# Create a directory to save regression test data
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
if new_regression_PA:
    for config in CONFIG_FILES_PERFORMANCE_ANALYSIS:
        create_regression_data_PA(config, DATA_DIR, CONFIG_DIR, OS_TYPE)
if new_regression_DO:
    for config in CONFIG_FILES_DESIGN_OPTIMIZATION:
        create_regression_data_DO(config, DATA_DIR, CONFIG_DIR, OS_TYPE)