import numpy as np
from . import numeric as num
from . import configuration as cfg


def evaluate_constraints(data, constraints):
    """
    Evaluates the constraints based on the provided data and constraint definitions.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the constraints will be evaluated.
    constraints : list of dicts
        A list of constraint definitions, where each constraint is defined as a dictionary.
        Each dictionary must have 'variable' (str), 'type' (str, one of '=', '>', '<'), and 'value' (numeric).

    Returns
    -------
    tuple of numpy.ndarray
        Returns two numpy arrays: the first is an array of equality constraints, and the second is an array of 
        inequality constraints. These arrays are flattened and concatenated from the evaluated constraint values.

    Raises
    ------
    ValueError
        If an unknown constraint type is specified in the constraints list.
    """
    # Initialize constraint lists
    c_eq = []    # Equality constraints
    c_ineq = []  # Inequality constraints

    # Loop over all constraint from configuration file
    for constraint in constraints:
        name = constraint['variable']
        type = constraint['type']
        target = constraint['value']
        normalize = constraint.get("normalize", False)

        # Get the performance value for the given variable name
        current = cfg.render_and_evaluate(name, data)

        # Evaluate constraint
        mismatch = current - target

        # Normalize constraint according to specifications
        normalize_factor = normalize if num.is_numeric(normalize) else target
        if normalize is not False:
            if normalize_factor == 0:
                raise ValueError(f"Cannot normalize constraint '{name} {type} {target}' because the normalization factor is '{normalize_factor}' (division by zero).")
            mismatch /= normalize_factor

        # Add constraints to lists
        if type == '=':
            c_eq.append(mismatch)
        elif type == '>':
            c_ineq.append(mismatch)
        elif type == '<':
            # Change sign because optimizer handles c_ineq > 0
            c_ineq.append(-mismatch)
        else:
            raise ValueError(f"Unknown constraint type: {type}")

    # Flatten and concatenate constraints
    c_eq = np.hstack([np.atleast_1d(item) for item in c_eq]) if c_eq else np.array([])
    c_ineq = np.hstack([np.atleast_1d(item) for item in c_ineq]) if c_ineq else np.array([])

    return c_eq, c_ineq


def evaluate_objective_function(data, objective_function):
    """
    Evaluates the objective function based on the provided data and configuration.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the objective function will be evaluated.
    objective_function : dict
        A dictionary defining the objective function. It must have 'variable' (str) 
        and 'type' (str, either 'minimize' or 'maximize').

    Returns
    -------
    float
        The value of the objective function, adjusted for optimization. Positive for minimization and 
        negative for maximization.

    Raises
    ------
    ValueError
        If an unknown objective function type is specified in the configuration.
    """

    # Get the performance value for the given variable name
    name = objective_function['variable']
    type = objective_function['type']
    value = cfg.render_and_evaluate(name, data)

    if not np.isscalar(value):
        raise ValueError(f"The objective function '{name}' must be an scalar, but the value is: {value}")

    if type == 'minimize':
        return value
    elif type == 'maximize':
        return -value
    else:
        raise ValueError(f"Unknown objective function type: {type}")

