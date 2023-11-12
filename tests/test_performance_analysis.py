import os
import sys
import copy
import numpy as np
import pandas as pd
import pytest

desired_path = os.path.abspath("..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


# Define run settings
SOLVER_ITER = 8  # Number of values to compare
DIGIT_TOLERANCE = 4  # 0.01%
DATA_DIR = "regression_data"
CONFIG_DIR = "config_files"
CONFIG_FILES = [
    "kofskey1972a_isentropic.yaml",
    "kofskey1972a_kacker_okapuu.yaml",
    "kofskey1972a_benner.yaml",
    "kofskey1972a_moustapha.yaml",
]


def isclose_significant_digits(a, b, sig_digits):
    """
    Check if two floating-point numbers are close based on a specified number of significant digits.

    Parameters
    ----------
    a : float
        The first number to compare.
    b : float
        The second number to compare.
    sig_digits : int
        The number of significant digits to use for the comparison.

    Returns
    -------
    bool
        True if numbers are close up to the specified significant digits, False otherwise.
    """
    format_spec = f".{sig_digits - 1}e"
    return format(a, format_spec) == format(b, format_spec)


# Helper function to check values
def check_values(old_values, new_values, column, config_name, idx, discrepancies):
    """
    Compare and check if the values from the saved dataset and current calculations match.
    Any discrepancies found are added to the discrepancies list with detailed messages.

    Parameters
    ----------
    saved_values : numpy.array
        Array of saved values from the dataset.
    current_values : numpy.array
        Array of current values from the solver.
    column : str
        The name of the data column being compared.
    config_name : str
        The name of the configuration file.
    idx : int
        Index of the current operation point.
    discrepancies : list
        A list to record discrepancies found during comparison.    
    """
    for i, (saved, current) in enumerate(zip(old_values, new_values)):
        if isinstance(saved, str) and saved != current:
            discrepancies.append(
                f"Mismatch in '{config_name}' at operation point {idx+1}, index {i+1} of '{column}': expected '{saved}', got '{current}'"
            )
        elif not isinstance(saved, str) and not isclose_significant_digits(
            saved, current, DIGIT_TOLERANCE
        ):
            discrepancies.append(
                f"Mismatch in '{config_name}' at operation point {idx+1}, index {i+1} of '{column}': expected {saved:.5e}, got {current:.5e}"
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
    config = ml.read_configuration_file(config_path)
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config)
    return solvers, config_file


# Test solver convergence using fixture
def test_solver_convergence(compute_performance_from_config):
    """
    Test solver convergence against saved data.

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

    # Loop over all operating points
    mismatch = []  # List to store discrepancies
    for idx, current_solver in enumerate(solvers):
        # Load saved convergence history from Excel file
        filename = os.path.join(DATA_DIR, f"{config_name}_{idx+1}.xlsx")
        saved_df = pd.read_excel(filename, sheet_name="convergence")

        # Compare the values for the first SOLVER_ITER values
        # The reason is that close to the solution round-off errors from finite differences
        # can affect the residuals and step size significantly
        for column in saved_df.columns:
            values_old = saved_df[column].head(SOLVER_ITER).to_numpy()
            values_new = np.array(current_solver.convergence_history[column][:SOLVER_ITER])
            check_values(values_old, values_new, column, config_name, idx, mismatch)

    # Assert no discrepancies
    test_passed = not mismatch
    assert test_passed, "Discrepancies found:\n" + "\n".join(mismatch)


def test_overall_performance(compute_performance_from_config):
    """
    Test overall solver performance against saved data.

    Parameters
    ----------
    compute_performance_from_config : fixture
        The pytest fixture that computes solver performance from configuration files.

    Asserts
    -------
    Asserts that there are no discrepancies between saved overall performance data and computed data.
    """
    # Get the solvers list and configuration filename
    solvers, config_file = compute_performance_from_config
    config_name, _ = os.path.splitext(config_file)

    # Loop over all operating points
    mismatch = []  # List to store discrepancies
    for idx, current_solver in enumerate(solvers):
        # Load saved overall performance data from Excel file
        config_name, _ = os.path.splitext(config_file)
        filename = os.path.join(DATA_DIR, f"{config_name}_{idx+1}.xlsx")
        saved_df = pd.read_excel(filename, sheet_name="overall")

        # If the 'Unnamed: 0' column exists, drop it
        if 'Unnamed: 0' in saved_df.columns:
            saved_df = saved_df.drop(columns=['Unnamed: 0'])

        # Compare all values
        overall_performance = current_solver.problem.results["overall"]
        for column in saved_df.columns:
            values_old = saved_df[column].to_numpy()
            values_new = np.array(overall_performance[column])
            check_values(values_old, values_new, column, config_name, idx, mismatch)

    # Assert no discrepancies
    test_passed = not mismatch
    assert test_passed, "Discrepancies found:\n" + "\n".join(mismatch)



# Run the tests
if __name__ == "__main__":
    pytest.main()

