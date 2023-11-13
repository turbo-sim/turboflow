import os
import sys
import copy
import pandas as pd
import datetime

desired_path = os.path.abspath("..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml

# Define run settings
DATA_DIR = "regression_data"
CONFIG_DIR = "config_files"
CONFIG_FILES = [
    "kofskey1972a_isentropic.yaml",
    "kofskey1972a_kacker_okapuu.yaml",
    "kofskey1972a_benner.yaml",
    "kofskey1972a_moustapha.yaml",
]

# Create a directory to save regression test data
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def save_regression_data(solvers, config_name, outdir):
    """Save performance analysis data to Excel files for regression tests"""

    # Get configuration file name without extension
    config_name, _ = os.path.splitext(config_name)

    # Loop over computed operation points
    for idx, solver in enumerate(solvers):
        base_filename = f"{config_name}_{idx+1}"
        filename = os.path.join(outdir, f"{base_filename}.xlsx")

        # If file exists, append a timestamp to the new file name
        if os.path.exists(filename):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(outdir, f"{base_filename}_{timestamp}.xlsx")

        # Write dataframes to excel
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            # Export simulation results
            for key, df in solver.problem.results.items():
                if key != "residuals":
                    df = copy.deepcopy(df)
                    df.index += 1
                    df.to_excel(writer, sheet_name=key, index=True)

            # Export convergence history
            df = pd.DataFrame(solver.convergence_history)
            df.to_excel(writer, sheet_name="convergence", index=False)

        print(f"Regression data set saved: {filename}")


if __name__ == "__main__":
    for config_file in CONFIG_FILES:
        # Defile configuration file path
        path = os.path.join(CONFIG_DIR, config_file)

        # Load configuration file
        config = ml.read_configuration_file(path)
        operation_points = config["operation_points"]
        solvers = ml.compute_performance(operation_points, config)

        # Call the function to save regression data
        save_regression_data(solvers, config_file, DATA_DIR)
