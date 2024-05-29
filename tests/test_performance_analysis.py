import os
import sys
import copy
import datetime
import numpy as np
import pandas as pd
import pytest
import turboflow as tf
import platform


# Define regression settings
TEST_FOLDER = "tests"
DATA_DIR = os.path.join(TEST_FOLDER, "regression_data")
CONFIG_DIR = os.path.join(TEST_FOLDER,"config_files")
CONFIG_FILES = [
    "performance_analysis_evaluate_cascade_throat.yaml",
    "performance_analysis_evaluate_cascade_critical.yaml",
    "performance_analysis_evaluate_cascade_isentropic_throat.yaml",
    "performance_analysis_kacker_okapuu.yaml",
    "performance_analysis_moustapha.yaml",
    "performance_analysis_zero_deviation.yaml",
    "performance_analysis_ainley_mathieson.yaml",
]

# Define test settings
DIGIT_TOLERANCE = 10  

# Check OS type
os_name = platform.system()
if os_name == 'Windows':
    OS_TYPE = 'win'
elif os_name == 'Linux':
    OS_TYPE = 'linux'
else:
    OS_TYPE = 'unknown'

# Helper function to check values
def check_values(old_values, new_values, column, config_name, discrepancies):
    """
    Compare and check if the values from the saved dataset and current calculations match.
    Any discrepancies found are added to the discrepancies list with detailed messages.

    Parameters
    ----------
    old_values : numpy.array
        Array of saved values from the dataset.
    new_values : numpy.array
        Array of current values from the solver.
    column : str
        The name of the data column being compared.
    config_name : str
        The name of the configuration file.
    discrepancies : list
        A list to record discrepancies found during comparison.    
    """
    for i, (saved, current) in enumerate(zip(old_values, new_values)):
        if isinstance(saved, str) and saved != current:
            discrepancies.append(
                f"Mismatch in '{config_name}' at operation point {i+1}, of '{column}': expected '{saved}', got '{current}'"
            )
        elif not isinstance(saved, str) and not tf.isclose_significant_digits(
            saved, current, DIGIT_TOLERANCE
        ):
            discrepancies.append(
                f"Mismatch in '{config_name}' at operation point {i+1}, of '{column}': expected {saved:.5e}, got {current:.5e}"
            )


# Fixture to compute solver performance
@pytest.fixture(scope="session", params=CONFIG_FILES)
def compute_performance_from_config(request):
    """
    Compute solver performance for cases specified in the configuration file.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The request object for the fixture, providing access to the invoking test context.

    Yields
    ------
    tuple
        A tuple (solvers, config_file) where:
        - solvers: A list of solver instances computed for the operation points.
        - config_file: The name of the configuration file used.

    Notes
    -----
    This fixture is parameterized over CONFIG_FILES and runs once for each configuration file.
    """
    print()
    config_file = request.param
    config_path = os.path.join(CONFIG_DIR, config_file)
    config = tf.load_config(config_path, )
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, export_results = False)
    return solvers, config_file 


# Test solver convergence using fixture
def test_performance_analysis(compute_performance_from_config):
    """
    Test performance analysis functionality. The function compares the calculated values against saved data on convergence
    and overall performance. 

    Parameters
    ----------
    compute_performance_from_config : fixture
        The pytest fixture that computes solver performance from configuration files.

    Asserts
    -------
    Asserts that there are no discrepancies between saved convergence data and computed data.
    """
    
    # Get the solvers list and configuration filename
    solvers, config_file = compute_performance_from_config
    config_name, _ = os.path.splitext(config_file)

    # Load saved convergence history from Excel file
    filename = os.path.join(DATA_DIR, f"{config_name}_{OS_TYPE}.xlsx")
    saved_df = pd.read_excel(filename, sheet_name=["solver", "overall"])

    # Loop over all operating points
    mismatch = []  # List to store discrepancies
    for key in solvers[0].convergence_history.keys():
        value_old = saved_df["solver"][key].to_numpy()
        value_new = np.array([solvers[i].convergence_history[key][-1] for i in range(len(solvers))])
        check_values(value_old, value_new, key, config_name, mismatch)

    for key in solvers[0].problem.results["overall"].keys():
        value_old = saved_df["overall"][key].to_numpy()
        value_new = np.array([solvers[i].problem.results["overall"][key][0] for i in range(len(solvers))])
        check_values(value_old, value_new, key, config_name, mismatch)

    # Assert no discrepancies
    test_passed = not mismatch
    assert test_passed, "Discrepancies found:\n" + "\n".join(mismatch)



