import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turboflow as tf

tf.set_plot_options()

# Main directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read case summary
DATAFILE = "./cases_summary.xlsx"
case_data = pd.read_excel(DATAFILE)

# # Run cases based on list of tags
# filter = ["ipopt"]
# case_data = case_data[case_data["method"].isin(filter)]

# Run cases based on case number
case_data = case_data[case_data["case"].isin([100, 101])]

# Loop over cases
results_list = []
for i, row in case_data.iterrows():

    # Load excel file with results
    casename = f"case_{row['case']}"
    excel_filename = os.path.splitext(row['config_file'])[0] # No .yaml extension
    excel_filepath = os.path.join(OUTPUT_DIR, casename, f"{excel_filename}_latest.xlsx")
    df = pd.read_excel(excel_filepath, sheet_name="solver")

    # Define case results dictionary
    results = {
        "stages": row['stages'],
        "method": row['method'],
        "derivative_method": row['derivative_method'],
        "derivative_abs_step": row['derivative_abs_step'],
        "tolerance": row['tolerance'],
        "objective_value": df['objective_value'].iloc[-1],
        "constraint_violation": df['constraint_violation'].iloc[-1],
        "grad_count": df['grad_count'].iloc[-1],
        "func_count": df['func_count'].iloc[-1],
        "func_count_total": df['func_count_total'].iloc[-1],
        "elapsed_time": df['elapsed_time'].iloc[0],
        "success": df['success'].iloc[0],
        "message": df['message'].iloc[0],
    }

    # Append results to the list
    results_list.append(results)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Optionally save the DataFrame to a CSV or Excel file
results_df.to_excel(os.path.join(OUTPUT_DIR, "consolidated_results.xlsx"), index=False)


