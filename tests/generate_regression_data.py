import sys
import os
import turboflow as tf
import datetime
import platform

config_files_performance_analysis = [
    "performance_analysis_evaluate_cascade_throat.yaml",
    "performance_analysis_evaluate_cascade_critical.yaml",
    "performance_analysis_evaluate_cascade_isentropic_throat.yaml",
    "performance_analysis_kacker_okapuu.yaml",
    "performance_analysis_moustapha.yaml",
    "performance_analysis_zero_deviation.yaml",
    "performance_analysis_ainley_mathieson.yaml",
]

config_files_design_optimization = ["design_optimization.yaml"]
data_dir = "regression_data"
config_dir = "config_files"

new_regression_PA = False
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
    
def get_os_type():
    os_name = platform.system()
    if os_name == 'Windows':
        return 'win'
    elif os_name == 'Linux':
        return 'linux'
    else:
        return 'unknown'
    
os_type = get_os_type()
    
if new_regression_PA:
    for config in config_files_performance_analysis:
        create_regression_data_PA(config, data_dir, config_dir, os_type)
if new_regression_DO:
    for config in config_files_design_optimization:
        create_regression_data_DO(config, data_dir, config_dir, os_type)