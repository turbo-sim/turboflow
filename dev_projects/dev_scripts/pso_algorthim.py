
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turboflow as tf
import pygmo as pg

# Define the problem dimension
dim = 2

# Set seed
pg.set_global_rng_seed(1)

# Create an instance of the Rastrigin problem
problem = pg.problem(tf.RosenbrockProblem(dim))

# Create a particle swarm algorithm instance
algo = pg.algorithm(pg.sga(gen=100))  # 'gen' is the number of generations
algo.set_verbosity(3)
pop = pg.population(problem, 20)
pop = algo.evolve(pop)

# Get the best solution found
champion_x = pop.champion_x
champion_f = pop.champion_f

# print("Best solution found:", champion_x)
# print("Objective value of the best solution:", champion_f)


# dim = 2
# problem = tf.RosenbrockProblem(dim)
# x0 = 5*np.ones(dim)
# solver_options = {
#     "library": "pygmo",
#     "method": "pso",
#     "tolerance": 1e-3, 
#     "max_iterations": 100, 
#     "derivative_method": "2-point", 
#     "derivative_abs_step": 1e-6,
#     "print_convergence": True,
#     "plot_convergence": True,
#     "update_on": "function"}

# # Number of generations
# # Population size 
# # Weights on the constraints

# # Initialize solver object using keyword-argument dictionary unpacking
# solver = tf.OptimizationSolver(problem, **solver_options)

# # Solve optimization problem for initial guess x0
# solution = solver.solve(x0)
# print(solution)