
import turboflow as tf
import numpy as np

# filename = "output/pickle_2stage_BB_pso_new_2024-11-05_09-03-00.pkl"
filename = "output/pickle_ipopt_2024-08-21_13-42-48.pkl"

solver = tf.load_from_pickle(filename)

def adjust_objective_values(objective_values, constraint_values, cut_off = None):

    # Adjust array such that the objective value is strictly deacreasing
    objective_values = objective_values.copy()
    constraint_values = constraint_values.copy()
    for i in range(1, len(objective_values)):
        if objective_values[i] > objective_values[i-1] or np.isnan(objective_values[i]) :
            objective_values[i] = objective_values[i-1]
            constraint_values[i] = constraint_values[i-1]

    # Cut off objective value
    if isinstance(cut_off, float):
        last_val = objective_values[-1]
        for i in range(1,len(objective_values)):
                if objective_values[i] <= cut_off*last_val:
                        objective_values = objective_values[:i]
                        constraint_values = constraint_values[:i]
                        break

    return objective_values, constraint_values
# obj_pso_20, cons_pso_20 = adjust_objective_values(solver.convergence_history["objective_value"][:-1], solver.problem.optimization_process.constraint_violation[1:], cut_off=1.0)
# print(np.where(solver.convergence_history["objective_value"][:-1] == min(solver.convergence_history["objective_value"][:-1])))
# print(solver.convergence_history["objective_value"][3500:])
# print(obj_pso_20[-1])

final = solver.convergence_history["x"][-1][13:]
print(final[13]*180/np.pi)





