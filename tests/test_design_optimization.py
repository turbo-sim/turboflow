import os
import sys
import copy
import datetime
import numpy as np
import pandas as pd
import pytest
import turboflow as tf
import platform
from . import get_config_files, REGRESSION_CONFIG_DIR

# Check OS type
os_name = platform.system()
SUPPORTED_OS = ["Windows", "Linux"]
if os_name not in SUPPORTED_OS:
    raise Exception("OS not supported")
os_name = os_name.lower()

# Define test settings
DIGIT_TOLERANCE = 10

# Helper function to check values
def check_value(old_value, new_value, column, config_name, discrepancies):
    """
    Compare and check if the value from the saved dataset and current calculatio match.
    Any discrepancies found are added to the discrepancies list with detailed messages.

    Parameters
    ----------
    old_value : float
        Saved value from the dataset.
    new_value : float
        Current value from the solver.
    column : str
        The name of the data column being compared.
    config_name : str
        The name of the configuration file.
    discrepancies : list
        A list to record discrepancies found during comparison.
    """
    if isinstance(old_value, str) and old_value != new_value:
        discrepancies.append(
            f"Mismatch of '{column}' in '{config_name}' : expected '{old_value}', got '{new_value}'"
        )
    elif not isinstance(old_value, str) and not tf.isclose_significant_digits(
        old_value, new_value, DIGIT_TOLERANCE
    ):
        discrepancies.append(
            f"Mismatch of '{column}' in '{config_name}': expected {old_value:.5e}, got {new_value:.5e}"
        )


# Fixture to compute solver performance
@pytest.fixture(scope="session", params=get_config_files("test_design_optimization"))
def compute_optimal_turbine_from_config(request):
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
    config_path = request.param
    config_file = os.path.basename(config_path)
    config = tf.load_config(config_path)
    solvers = tf.compute_optimal_turbine(config, export_results=False)
    return solvers, config_file


# Test solver convergence using fixture
def test_design_optimization(compute_optimal_turbine_from_config):
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
    solver, config_file = compute_optimal_turbine_from_config
    config_name, _ = os.path.splitext(config_file)

    # Load saved convergence history from Excel file
    filename = os.path.join(REGRESSION_CONFIG_DIR, f"{config_name}_{os_name}.xlsx")
    saved_df = pd.read_excel(filename, sheet_name=["solver", "overall"])

    # Loop over all operating points
    mismatch = []  # List to store discrepancies
    for key in solver.convergence_history.keys():
        value_old = saved_df["solver"][key].to_numpy()[-1]
        value_new = solver.convergence_history[key][-1]
        check_value(value_old, value_new, key, config_name, mismatch)

    for key in solver.problem.results["overall"].keys():
        value_old = saved_df["overall"][key].to_numpy()[0]
        value_new = solver.problem.results["overall"][key][0]
        check_value(value_old, value_new, key, config_name, mismatch)

    # Assert no discrepancies
    test_passed = not mismatch
    assert test_passed, "Discrepancies found:\n" + "\n".join(mismatch)