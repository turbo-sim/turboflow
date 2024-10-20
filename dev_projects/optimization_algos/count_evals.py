import turboflow as tf
import numpy as np

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_snopt_2024-08-19_20-45-38.pkl"
filename_sga_10 = "output/pickle_sga_2024-08-21_18-46-08.pkl"
filename_pso_10 = "output/pickle_pso_2024-08-22_15-39-52.pkl"
filename_sga_20 = "output/pickle_sga_2024-08-26_05-00-45.pkl"
filename_pso_20 = "output/pickle_pso_2024-08-23_09-31-51.pkl"
filename_slsqp_BB = "output/pickle_slsqp_BB_2024-08-23_10-49-46.pkl"
filename_snopt_BB = "output/pickle_snopt_BB_2024-08-23_15-33-27.pkl"
filename_ipopt_BB = "output/pickle_ipopt_BB_2024-08-24_01-35-28.pkl"

solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)
# solver_sga_10 = tf.load_from_pickle(filename_sga_10)
# solver_pso_10 = tf.load_from_pickle(filename_pso_10)
solver_sga_20 = tf.load_from_pickle(filename_sga_20)
solver_pso_20 = tf.load_from_pickle(filename_pso_20)
solver_slsqp_BB = tf.load_from_pickle(filename_slsqp_BB)
solver_snopt_BB = tf.load_from_pickle(filename_snopt_BB)
solver_ipopt_BB = tf.load_from_pickle(filename_ipopt_BB)

print("\n")
print("EO, SLSQP")
print(f"Grad eval: {solver_slsqp.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_slsqp.convergence_history['func_count'][-1]}")
print(f"Model eval: {solver_slsqp.convergence_history['func_count_total'][-1]}")

print("\n")
print("EO, IPOPT")
print(f"Grad eval: {solver_ipopt.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_ipopt.convergence_history['func_count'][-1]}")
print(f"Model eval: {solver_ipopt.convergence_history['func_count_total'][-1]}")

print("\n")
print("EO, SNOPT")
print(f"Grad eval: {solver_snopt.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_snopt.convergence_history['func_count'][-1]}")
print(f"Model eval: {solver_snopt.convergence_history['func_count_total'][-1]}")

print("\n")
print("BB, SLSQP")
print(f"Grad eval: {solver_slsqp_BB.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_slsqp_BB.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_slsqp_BB.problem.optimization_process.root_finder_solvers[1:]])}")

print("\n")
print("BB, IPOPT")
print(f"Grad eval: {solver_ipopt_BB.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_ipopt_BB.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_ipopt_BB.problem.optimization_process.root_finder_solvers[1:]])}")

print("\n")
print("BB, SNOPT")
print(f"Grad eval: {solver_snopt_BB.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_snopt_BB.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_snopt_BB.problem.optimization_process.root_finder_solvers[1:]])}")

print("\n")
print("BB, SGA")
print(f"Func eval: {solver_sga_20.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_sga_20.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")

print("\n")
print("BB, PSO")
print(f"Func eval: {solver_pso_20.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_pso_20.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
