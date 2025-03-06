import turboflow as tf
import numpy as np
import pandas as pd

def adjust_objective_values(objective_values, constraint_values, efficiency_values, cut_off = 1.0):

    objective_values = objective_values.copy()
    constraint_values = constraint_values.copy()
    efficiency_values = efficiency_values.copy()
    for i in range(1, len(objective_values)):
        if objective_values[i] > objective_values[i-1] or np.isnan(objective_values[i]) :
            objective_values[i] = objective_values[i-1]
            constraint_values[i] = constraint_values[i-1]
            efficiency_values[i] = efficiency_values[i-1]

    # Cut off objective value
    last_val = objective_values[-1]
    for i in range(1,len(objective_values)):
        if objective_values[i] <= cut_off*last_val:
            objective_values = objective_values[:i]
            constraint_values = constraint_values[:i]
            efficiency_values = efficiency_values[:i]
            break

    return objective_values, constraint_values, efficiency_values

# Load object
filename_slsqp = "output/pickle_slsqp_2024-08-19_14-30-05.pkl"
filename_ipopt = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"
filename_snopt = "output/pickle_SNOPT_tight_2025-02-17_17-11-25.pkl" # New V2
filename_sga_20_new = "output/pickle_2stage_BB_sga_new_2024-10-31_18-19-51.pkl"
filename_pso_30_new = "output/pickle_2stage_BB_pso_new_2024-11-06_01-08-01.pkl"
filename_slsqp_BB_new = "output/pickle_2stage_BB_slsqp_new_2024-10-30_20-41-31.pkl"
filename_ipopt_BB_new = "output/pickle_2stage_BB_ipopt_new_2024-10-31_03-29-47.pkl"
filename_snopt_BB_new = "output/pickle_2stage_BB_snopt_new_2024-10-31_11-32-52.pkl"
filename_ig = "output/initial_guess_2024-09-20_16-12-35.pkl"
solver_slsqp = tf.load_from_pickle(filename_slsqp)
solver_ipopt = tf.load_from_pickle(filename_ipopt)
solver_snopt = tf.load_from_pickle(filename_snopt)
solver_sga_20_new = tf.load_from_pickle(filename_sga_20_new)
solver_pso_30_new = tf.load_from_pickle(filename_pso_30_new)
solver_slsqp_BB_new = tf.load_from_pickle(filename_slsqp_BB_new)
solver_ipopt_BB_new = tf.load_from_pickle(filename_ipopt_BB_new)
solver_snopt_BB_new = tf.load_from_pickle(filename_snopt_BB_new)
initial_guess = tf.load_from_pickle(filename_ig)

# Get objective value and constraints in decreaseing order for gradient free convergence
obj_sga, cons_sga, eff_sga = adjust_objective_values(solver_sga_20_new.problem.optimization_process.iterations_objective_function[1:], solver_sga_20_new.problem.optimization_process.constraint_violation[1:], solver_sga_20_new.problem.optimization_process.iterations_efficiency[1:])
obj_pso, cons_pso, eff_pso = adjust_objective_values(solver_pso_30_new.problem.optimization_process.iterations_objective_function[1:], solver_pso_30_new.problem.optimization_process.constraint_violation[1:], solver_pso_30_new.problem.optimization_process.iterations_efficiency[1:])

# Get initial guess
initial_fitness = -1*initial_guess.results["overall"]["efficiency_ts"].values[0]
initial_efficiency = initial_guess.results["overall"]["efficiency_ts"].values[0]


print("\n")
print("EO, SLSQP")
print(f"Objective value: {solver_slsqp.convergence_history['objective_value'][-1]}")
print(f"Efficiency: {solver_slsqp.problem.results['overall']['efficiency_ts']}")
print(f"Fitness improvment: {(solver_slsqp.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {solver_slsqp.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
print(f"Constraint violation: {solver_slsqp.convergence_history['constraint_violation'][-1]}")
print(f"Grad eval: {solver_slsqp.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_slsqp.convergence_history['func_count'][-1]}")
print(f"Model eval: {solver_slsqp.convergence_history['func_count_total'][-1]}")
print(f"Model eval for grad: {solver_slsqp.convergence_history['grad_count'][-1]*27}")
print(f"Fraction of model eval for grad: {solver_slsqp.convergence_history['grad_count'][-1]*27/solver_slsqp.convergence_history['func_count_total'][-1]*100}")

print("\n")
print("EO, IPOPT")
print(f"Objective value: {solver_ipopt.convergence_history['objective_value'][-1]}")
print(f"Efficiency: {solver_ipopt.problem.results['overall']['efficiency_ts']}")
print(f"Fitness improvment: {(solver_ipopt.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {solver_ipopt.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
print(f"Constraint violation: {solver_ipopt.convergence_history['constraint_violation'][-1]}")
print(f"Grad eval: {solver_ipopt.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_ipopt.convergence_history['func_count'][-1]}")
print(f"Model eval: {solver_ipopt.convergence_history['func_count_total'][-1]}")
print(f"Model eval for grad: {solver_ipopt.convergence_history['grad_count'][-1]*27}")
print(f"Fraction of model eval for grad: {solver_ipopt.convergence_history['grad_count'][-1]*27/solver_ipopt.convergence_history['func_count_total'][-1]*100}")


print("\n")
print("EO, SNOPT")
print(f"Objective value: {solver_snopt.convergence_history['objective_value'][-1]}")
print(f"Efficiency: {solver_snopt.problem.results['overall']['efficiency_ts']}")
print(f"Fitness improvment: {(solver_snopt.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {solver_snopt.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
print(f"Constraint violation: {solver_snopt.convergence_history['constraint_violation'][-1]}")
print(f"Grad eval: {solver_snopt.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_snopt.convergence_history['func_count'][-1]}")
print(f"Model eval: {solver_snopt.convergence_history['func_count_total'][-1]}")
print(f"Model eval for grad: {solver_snopt.convergence_history['grad_count'][-1]*27}")
print(f"Fraction of model eval for grad: {solver_snopt.convergence_history['grad_count'][-1]*27/solver_snopt.convergence_history['func_count_total'][-1]*100}")


print("\n")
print("BB, SLSQP")
print(f"Objective value: {solver_slsqp_BB_new.convergence_history['objective_value'][-1]}")
print(f"Efficiency: {solver_slsqp_BB_new.problem.results['overall']['efficiency_ts']}")
print(f"Fitness improvment: {(solver_slsqp_BB_new.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {solver_slsqp_BB_new.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
print(f"Constraint violation: {solver_slsqp_BB_new.problem.optimization_process.constraint_violation[-1]}")
print(f"Grad eval: {solver_slsqp_BB_new.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_slsqp_BB_new.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_slsqp_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
print(f"Model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_slsqp_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
print(f"Fraction of model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_slsqp_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])/np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_slsqp_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])*100}")
grad_count1 = pd.Series([solver.convergence_history['grad_count'][-1]*11 + solver.convergence_history['func_count'][-1] for solver in solver_slsqp_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])
total_count = np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_slsqp_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])
FD_calc = grad_count1.mean()*solver_slsqp_BB_new.convergence_history["grad_count"][-1]*16 
print(f"Fraction of model eval for grad: {FD_calc/total_count*100}")

print("\n")
print("BB, IPOPT")
print(f"Objective value: {solver_ipopt_BB_new.convergence_history['objective_value'][-1]}")
print(f"Efficiency: {solver_ipopt_BB_new.problem.results['overall']['efficiency_ts']}")
print(f"Fitness improvment: {(solver_ipopt_BB_new.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {solver_ipopt_BB_new.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
print(f"Constraint violation: {solver_ipopt_BB_new.problem.optimization_process.constraint_violation[-1]}")
print(f"Grad eval: {solver_ipopt_BB_new.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_ipopt_BB_new.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_ipopt_BB_new.problem.optimization_process.root_finder_solvers[1:]])}")
print(f"Model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_ipopt_BB_new.problem.optimization_process.root_finder_solvers[1:]])}")
print(f"Fraction of model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_ipopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])/np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_ipopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])*100}")
grad_count1 = pd.Series([solver.convergence_history['grad_count'][-1]*11 + solver.convergence_history['func_count'][-1] for solver in solver_ipopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])
total_count = np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_ipopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])
FD_calc = grad_count1.mean()*solver_ipopt_BB_new.convergence_history["grad_count"][-1]*16
print(f"Fraction of model eval for grad: {FD_calc/total_count*100}")

print("\n")
print("BB, SNOPT")
print(f"Objective value: {solver_snopt_BB_new.convergence_history['objective_value'][-1]}")
print(f"Efficiency: {solver_snopt_BB_new.problem.results['overall']['efficiency_ts']}")
print(f"Fitness improvment: {(solver_snopt_BB_new.convergence_history['objective_value'][-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {solver_snopt_BB_new.problem.results['overall']['efficiency_ts'] - initial_efficiency}")
print(f"Constraint violation: {solver_snopt_BB_new.problem.optimization_process.constraint_violation[-1]}")
print(f"Grad eval: {solver_snopt_BB_new.convergence_history['grad_count'][-1]}")
print(f"Func eval: {solver_snopt_BB_new.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_snopt_BB_new.problem.optimization_process.root_finder_solvers[1:]])}")
print(f"Model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_snopt_BB_new.problem.optimization_process.root_finder_solvers[1:]])}")
print(f"Fraction of model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_snopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])/np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_snopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])*100}")
grad_count1 = pd.Series([solver.convergence_history['grad_count'][-1]*11 + solver.convergence_history['func_count'][-1] for solver in solver_snopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])
total_count = np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_snopt_BB_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])
FD_calc = grad_count1.mean()*solver_snopt_BB_new.convergence_history["grad_count"][-1]*16 
print(f"Fraction of model eval for grad: {FD_calc/total_count*100}")

print("\n")
print("BB, SGA")
print(f"Objective value: {obj_sga[-1]}")
print(f"Efficiency: {eff_sga[-1]}")
print(f"Fitness improvment: {(obj_sga[-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {eff_sga[-1] - initial_efficiency}")
print(f"Constraint violation: {cons_sga[-1]}")
print(f"Func eval: {solver_sga_20_new.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_sga_20_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
print(f"Model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_sga_20_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
print(f"Fraction of model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_sga_20_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])/np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_sga_20_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])*100}")

print("\n")
print("BB, PSO")
print(f"Objective value: {obj_pso[-1]}")
print(f"Efficiency: {eff_pso[-1]}")
print(f"Fitness improvment: {(obj_pso[-1]-initial_fitness)/initial_fitness*100}")
print(f"Efficiency improvment: {eff_pso[-1] - initial_efficiency}")
print(f"Constraint violation: {cons_pso[-1]}")
print(f"Func eval: {solver_pso_30_new.convergence_history['func_count'][-1]}")
print(f"Model eval: {np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_pso_30_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
print(f"Model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_pso_30_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])}")
print(f"Fraction of model eval for grad: {np.sum([solver.convergence_history['grad_count'][-1]*11 for solver in solver_pso_30_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])/np.sum([solver.convergence_history['func_count_total'][-1] for solver in solver_pso_30_new.problem.optimization_process.root_finder_solvers[1:] if not isinstance(solver, float)])*100}")
