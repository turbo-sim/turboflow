
import turboflow as tf
import numpy as np
import CoolProp as cp
import os

stages = ["stage1.yaml",
          "stage1.yaml",
          "stage1.yaml",
          "stage1.yaml",]

p0_in = 101325
T0_in = 288.15 + 120
omega = 14000*2*np.pi/60
mass_flow_rate = 3.5
fluid = "water"
intercooling = 0

operating_point = {
    "T0_in" : T0_in,
    "p0_in" : p0_in,
    "alpha_in" : 0,
    "omega" : omega,
    "mass_flow_rate" : mass_flow_rate,
    "fluid_name" : fluid,
}

pressure_ratio = []
temperature = [T0_in]
efficiency = []

correct_solutions = []
choked = []
mass_flow_residual = []

for stage in stages:

    CONFIG_FILE = os.path.abspath(stage)
    config = tf.load_config(CONFIG_FILE)

    solvers = tf.centrifugal_compressor.compute_performance(
        config,
        operating_point,
        export_results=False,
        stop_on_failure=False,
    )    

    pressure_ratio.append(solvers[0].problem.results["overall"]["PR_tt"])
    temperature.append(solvers[0].problem.results["planes"]["T0"].values[-1])
    efficiency.append(solvers[0].problem.results["overall"]["efficiency_tt"])
    correct_solutions.append(solvers[0].problem.results["impeller"]["throat_plane"]["correct_solution"])
    choked.append(solvers[0].problem.results["impeller"]["throat_plane"]["choked"])
    mass_flow_residual.append(solvers[0].problem.results["impeller"]["throat_plane"]["mass_flow_residual"])

    operating_point["p0_in"] = solvers[0].problem.results["planes"]["p0"].values[-1]
    operating_point["T0_in"] = solvers[0].problem.results["planes"]["T0"].values[-1] - intercooling
    

p0_out = solvers[0].problem.results["planes"]["p0"].values[-1]

print(f"Overall pressure ratio: {p0_out/p0_in}")
print(f"Pressure ratio for each stage: {pressure_ratio}")
print(f"Temperature through compressor: {np.array(temperature)-273.15} C")
print(f"Efficiency for each stage: {efficiency}")

# Checks
print("\nChecks:")
print(f"Choked: {choked}")
print(f"Mass flow residual: {mass_flow_residual}")
print(f"Correct solution: {correct_solutions}")


