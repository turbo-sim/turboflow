import sys
import os
import turboflow as tf
import datetime
import platform
from tests_manager import get_regression_config_files, REGRESSION_CONFIG_DIR

# Check OS type
SUPPORTED_OS = ["Windows", "Linux"]
os_name = platform.system()
if os_name not in SUPPORTED_OS:
    raise Exception("OS not supported")
os_name = os_name.lower()

def create_regression_data_PA(config_file, data_dir, os_type):
    """Save performance analysis data to Excel files for regression tests"""

    # Run simulations
    config = tf.load_config(config_file)

    # Get configuration file name without extension
    config_name, _ = os.path.splitext(os.path.basename(config_file))
    base_filename = f"{config_name}"
    filename = f"{base_filename}_{os_type}"
    # If file exists, append a timestamp to the new file name
    if os.path.exists(f"{os.path.join(data_dir, filename)}.xlsx"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{base_filename}_{timestamp}"

    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, out_dir = data_dir, out_filename= filename, export_results = True)

    print(f"Regression data set saved: {filename}")
    
def create_regression_data_DO(config_file, data_dir, os_type):
    """Save performance analysis data to Excel files for regression tests"""

    # Run simulations
    config = tf.load_config(config_file)

    # Get configuration file name without extension
    config_name, _ = os.path.splitext(os.path.basename(config_file))
    base_filename = f"{config_name}"
    filename = f"{base_filename}_{os_type}"
    
    # If file exists, append a timestamp to the new file name
    if os.path.exists(f"{os.path.join(data_dir, filename)}.xlsx"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{base_filename}_{timestamp}"

    solvers = tf.compute_optimal_turbine(config, out_dir = data_dir, out_filename= filename, export_results = True)

    print(f"Regression data set saved: {filename}")
    
if __name__ == "__main__":

    # Create a directory to save regression test data
    if not os.path.exists(REGRESSION_CONFIG_DIR):
        os.makedirs(REGRESSION_CONFIG_DIR)

    # Get file paths to configuration files for new test data
    config_performance_analysis = get_regression_config_files('performance_analysis')
    config_design_optimization = get_regression_config_files('design_optimization')

    if not config_performance_analysis == None:
        for config in config_performance_analysis:
            create_regression_data_PA(config, REGRESSION_CONFIG_DIR, os_name)
    else:
        print("No configuration files selected to generate regression data for performance analysis tests")
    
    if not config_design_optimization == None:
        for config in config_design_optimization:
            create_regression_data_DO(config, REGRESSION_CONFIG_DIR, os_name)
    else:
        print("No configuration files selected to generate regression data for design optimization tests")