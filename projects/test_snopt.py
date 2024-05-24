# Import packages
import numpy as np
import matplotlib.pyplot as plt
import meanline_axial as pv
 
# Set options for publication-quality figures
pv.set_plot_options(grid=False)
 
# Set up logger with unique date-time name
logger = pv.create_logger("convergence_history", use_datetime=True)
 
# Solve Rosenbrock problem
x0 = 1.5*np.asarray([1, 1, 1, 1])
problem = pv.RosenbrockProblemConstrained(len(x0))
solver = pv.OptimizationSolver(
    problem,
    library="scipy",
    method="slsqp",
    max_iterations=100,
    tolerance=1e-4,
    print_convergence=True,
    plot_convergence=True,
    logger=logger,
    update_on="function",
    plot_scale_objective="log",
    plot_scale_constraints="log",
)
 
# Solve the problem
solver.solve(x0)
 
# Keep plots open
plt.show()
 
# # Save the figure
# solver.plot_convergence_history(savefile=True)