import os
import sys
import numpy as np
import matplotlib.pyplot as plt    
import meanline_axial as ml

# Define configuration filename
CONFIG_FILE = "case_sCO2_recompression.yaml"


# TODO For IPOPT is very important to provide the hessian or to increase the number of previous gradients used to update the bfgs
# How many previous updates do SLSQP and SNOPT keep in memory




# Analyze sensitivity to algorithm
fig, ax_1 = plt.subplots()
ax_1.set_xlabel("Number of iterations")
ax_1.set_ylabel("Objective function values")
# ax_2 = ax_1.twinx()
# ax_2.set_ylabel("Constraint violation")

# ax.set_yscale("log")
# methods = ['scipy:slsqp', 'snopt', 'ipopt']
methods = ['ipopt']
cmap = "magma"
colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.8, len(methods)))
for method, color in zip(methods, colors):

    # Initialize Brayton cycle problem
    config = ml.utilities.read_configuration_file(CONFIG_FILE)
    thermoCycle = ml.cycles.ThermodynamicCycleProblem(config["problem_formulation"])
    fit = thermoCycle.fitness(thermoCycle.x0)

    # Optimize the thermodynamic cycle
    config["solver_options"]["method"] = method
    solver = ml.OptimizationSolver(
        thermoCycle,
        thermoCycle.x0,
        **config["solver_options"],
        # callback_func=thermoCycle.plot_cycle_callback,
    )
    solver.solve()

    # Print final solution values
    print()
    print("Optimal set of design variables (normalized)")
    for key, value in solver.problem.vars_normalized.items():
        print(f"{key:40}: {value:0.3f}")

    # Plot convergence history
    x = solver.convergence_history["grad_count"]
    y1 = -100 * np.asarray(solver.convergence_history["objective_value"])
    y2 = solver.convergence_history["constraint_violation"]
    ax_1.plot(
        x,
        y1,
        label=method,
        marker="o",
        markersize=3.0,
        color=color,
    )
    # ax_2.plot(
    #     x,
    #     y2,
    #     marker="o",
    #     linestyle="--",
    #     markersize=3.0,
    #     color=color,
    # )

ax_1.legend(loc="lower right", fontsize=11)
fig.tight_layout(pad=1)
ml.savefig_in_formats(fig, "figures/optimization_solvers")

# Keep plots open
plt.show()


