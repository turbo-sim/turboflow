import os
import numpy as np
from scipy.optimize import minimize

# Attempt to import pygmo and pygmo_plugins_nonfree
try:
    import pygmo as pg
    import pygmo_plugins_nonfree as ppnf

    PYGMO_AVAILABLE = True

except ImportError:
    PYGMO_AVAILABLE = False

SCIPY_SOLVERS = [
    "nelder-mead",
    "powell",
    "cg",
    "bfgs",
    "newton-cg",
    "l-bfgs-b",
    "tnc",
    "cobyla",
    "slsqp",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
]
"""List of valid Scipy solvers"""


PYGMO_SOLVERS = [
    "ipopt",
    "snopt",
    # "worhp",
    "de",
    "sga",
    "pso",
]
"""List of valid Pygmo solvers"""

GENETIC_SOLVERS = [
    "de",
    "sga",
    "pso",
]

GRADIENT_SOLVERS = list(set(PYGMO_SOLVERS + SCIPY_SOLVERS) - set(GENETIC_SOLVERS))

NLOPT_SOLVERS = [
    "auglag",
    "auglag_eq",
    "bobyqa",
    "ccsaq",
    "cobyla",
    "lbfgs",
    "mma",
    "neldermead",
    "newuoa",
    "newuoa_bound",
    "praxis",
    "sbplx",
    "slsqp",
    "tnewton",
    "tnewton_precond",
    "tnewton_precond_restart",
    "tnewton_restart",
    "var1",
    "var2",
]
"""List of valid NLOpt solvers (interfaced through Pygmo)"""


def minimize_scipy(problem, x0, method, options):
    """
    Optimize a given problem using the specified SciPy method.

    Parameters
    ----------
    problem : OptimizationProblem
        An object representing the problem.
    x0 : array_like
        Initial guess for the decision variables.
    method : str
        The name of the Scipy solver to use. Must be one of the solvers defined in the SCIPY_SOLVERS list.
    options : dict
        Configuration options for the solver. Options vary depending on the solver used.

    Returns
    -------
    tuple
        A tuple (success, message) where 'success' is a boolean indicating if the optimization was successful, and 'message' is a string describing the outcome.

    Raises
    ------
    ValueError
        If an invalid solver is specified.
    """

    # Check if method is valid
    if method not in SCIPY_SOLVERS:
        error_message = (
            f"Invalid solver: '{method}'. \nAvailable options:\n   - "
            + "\n   - ".join(SCIPY_SOLVERS)
            + "."
        )
        raise ValueError(error_message)

    # Define the options dictionary
    tol = 1e-6 # Default tolerance
    max_iter = 100 # Default iterations
    default_options = {
        "bfgs": {"gtol": tol, "maxiter": max_iter},
        "l-bfgs-b": {"ftol": tol, "gtol": tol, "maxiter": max_iter},
        "slsqp": {"ftol": tol, "maxiter": max_iter},
        "trust-constr": {"gtol": tol, "xtol": tol, "maxiter": max_iter},
    }
    default_options = (
        default_options[method] if method in default_options.keys() else {}
    )
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options  # Merge defaults with input

    # Convert bounds from Pygmo to Scipy convention
    bounds = convert_pygmo_to_scipy_bounds(problem.get_bounds)

    # Define list of constraint dictionaries
    constr = []
    if problem.get_nec() > 0:
        constr.append({"type": "eq", "fun": problem.c_eq, "jac": problem.c_eq_jac})
    if problem.get_nic() > 0:
        # Convert inequality constraint convention from Pygmo (c<0) to SciPy (c>0)
        c_ineq = lambda x: -problem.c_ineq(x)
        c_ineq_jac = lambda x: -problem.c_ineq_jac(x)
        constr.append({"type": "ineq", "fun": c_ineq, "jac": c_ineq_jac})

    # Solve the optimization problem
    solution = minimize(
        problem.f,
        x0,
        tol=tol,
        jac=problem.f_jac,
        constraints=constr,
        bounds=bounds(),
        method=method,
        options=combined_options,
    )

    # Optimization output
    success = solution.success
    message = solution.message

    return success, message


def minimize_pygmo(problem, x0, method, options):
    """
    Optimize a given problem using the specified Pygmo optimization method.

    Parameters
    ----------
    problem : OptimizationProblem
        An object representing the problem.
    x0 : array_like
        Initial guess for the decision variables.
    method : str
        The name of the Pygmo solver to use. Must be one of the solvers defined in the PYGMO_SOLVERS list.
    options : dict
        Configuration options for the solver. Options vary depending on the solver used.

    Returns
    -------
    tuple
        A tuple (success, message) where 'success' is a boolean indicating if the optimization was successful, and 'message' is a string describing the outcome.

    Raises
    ------
    ValueError
        If an invalid solver name is specified.
    """

    # Check if pygmo is available
    if not PYGMO_AVAILABLE:
        raise ImportError(
            f"To use the selected optimizer ({method}), you need to install 'pygmo' via Conda.\n\n"
            "   1. Activate the Conda virtual environment where 'turboflow' is installed. For example:\n\n"
            "           conda activate turboflow_env\n\n"
            "   2. Install 'pygmo' and 'pygmo_plugins_nonfree' by running the following command:\n\n"
            "           conda install --channel conda-forge pygmo pygmo_plugins_nonfree\n"
        )

    if method == "snopt":
        return _minimize_pygmo_snopt(problem, x0, options)
    elif method == "ipopt":
        return _minimize_pygmo_ipopt(problem, x0, options)
    # elif method == "worhp":
    #     return _minimize_pygmo_worhp(problem, x0, options)
    elif method == "de":
        return _minimize_pygmo_de(problem, x0, options)
    elif method == "sga":
        return _minimize_pygmo_sga(problem, x0, options)
    elif method == "pso":
        return _minimize_pygmo_pso(problem, x0, options)
    else:
        error_message = (
            f"Invalid solver: '{method}'. \nAvailable options:\n   - "
            + "\n   - ".join(PYGMO_SOLVERS)
            + "."
        )
        raise ValueError(error_message)

def _minimize_pygmo_de(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to de"""

    # Define solver options
    default_options = {
        "gen": 10, 
        "pop_size": 10,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options
    
    # Solve the problem
    algorithm = pg.algorithm(pg.de(gen = int(combined_options["gen"])))
    algorithm.set_verbosity(50)
    problem = pg.problem(problem)
    population = pg.population(problem, int(combined_options["pop_size"]))
    population = algorithm.evolve(population)

    # print solution
    print("Best solution found:", population.champion_x)
    print("Objective value of the best solution:", population.champion_f)

    # Optimization output
    success = ""
    message = ""

    return success, message

def _minimize_pygmo_sga(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to sga"""

    # Define solver options
    default_options = {
        "gen": 10, 
        "pop_size": 10,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options

    # Set seed 
    if "seed" in list(combined_options.keys()):
        pg.set_global_rng_seed(combined_options["seed"])
    
    # Solve the problem
    algorithm = pg.algorithm(pg.sga(gen = int(combined_options["gen"])))
    algorithm.set_verbosity(0)
    problem = pg.problem(problem)
    population = pg.population(problem, int(combined_options["pop_size"]))
    population = algorithm.evolve(population)

    # print solution
    print("Best solution found:", population.champion_x)
    print("Objective value of the best solution:", population.champion_f)

    # Evaluate final solution 
    final = problem.fitness(population.champion_x)

    # Optimization output
    success = ""
    message = ""

    return success, message

def _minimize_pygmo_pso(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to PSO"""

    # Define solver options
    default_options = {
        "gen": 10, 
        "pop_size": 10,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options

    # Set seed 
    if "seed" in list(combined_options.keys()):
        pg.set_global_rng_seed(combined_options["seed"])

    # Solve the problem
    algorithm = pg.algorithm(pg.pso(gen = int(combined_options["gen"])))
    algorithm.set_verbosity(0)
    problem = pg.problem(problem)
    population = pg.population(problem, int(combined_options["pop_size"]))
    population = algorithm.evolve(population)

    # print solution
    print("Best solution found:", population.champion_x)
    print("Objective value of the best solution:", population.champion_f)

    # Evaluate final solution 
    final = problem.fitness(population.champion_x)

    # Optimization output
    success = ""
    message = ""

    return success, message


def _minimize_pygmo_ipopt(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to IPOPT"""

    # Define mapping from exit flag to status
    exitflag_mapping = {
        0: "Solve Succeeded",
        1: "Solved To Acceptable Level",
        2: "Infeasible Problem Detected",
        3: "Search Direction Becomes Too Small",
        4: "Diverging Iterates",
        5: "User Requested Stop",
        6: "Feasible Point Found",
        -1: "Maximum Iterations Exceeded",
        -2: "Restoration Failed",
        -3: "Error In Step Computation",
        -4: "Maximum CpuTime Exceeded",
        -5: "Maximum WallTime Exceeded",
        -10: "Not Enough Degrees Of Freedom",
        -11: "Invalid Problem Definition",
        -12: "Invalid Option",
        -13: "Invalid Number Detected",
        -100: "Unrecoverable Exception",
        -101: "NonIpopt Exception Thrown",
        -102: "Insufficient Memory",
        -199: "Internal Error",
    }
    
    # Define solver options
    default_options = {
        "print_level": 0,  # No output to the screen
        "print_info_string" : "yes",
        "sb": "yes",  # Suppress banner
        "hessian_approximation": "limited-memory",  # exact, limited-memory
        "limited_memory_update_type": "bfgs",  # bfgs, sr1
        "line_search_method": "cg-penalty",  # filter, cg-penalty, penalty
        "limited_memory_max_history": 30,  # bfgs
        "max_iter": 100,
        "tol": 1e-3,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options
    combined_options["max_iter"] = int(combined_options["max_iter"])
    combined_options["print_level"] = int(combined_options["print_level"])
    combined_options["acceptable_iter"] = int(combined_options["acceptable_iter"])
    if combined_options["tol"] < 1e-3:
        print(
            f"IPOPT struggles to converge to tight tolerances when exact Hessians are not provided.\nConsider relaxing the termination tolerance above 1e-3. Current tolerance: {combined_options['tol']:0.2e}."
        )

    # Solve the problem
    problem = pg.problem(problem)
    algorithm = pg.algorithm(pg.ipopt())
    algorithm_handle = algorithm.extract(pg.ipopt)
    _set_pygmo_options(algorithm_handle, combined_options)
    population = pg.population(problem, size=1)
    population.set_x(0, x0)
    population = algorithm.evolve(population)

    # Optimization output
    exitflag = algorithm_handle.get_last_opt_result()
    message = exitflag_mapping.get(exitflag, "Unknown Status")
    success = exitflag in (0, 1)

    return success, message


def _minimize_pygmo_snopt(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to SNOPT"""

    # Define solver options
    default_options = {
        "Major feasibility tolerance": 1e-3,
        "Major optimality tolerance": 1e-3,
        "Minor feasibility tolerance": 1e-3,
        "Iterations limit": 100,
        "Major iterations limit": 1e6,
        "Minor iterations limit": 1e6,
        "Hessian updates": 30,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options
    combined_options["Iterations limit"] = int(combined_options["Iterations limit"])

    # Define SNOPT path
    lib = os.getenv("SNOPT_LIB")
    if not lib:
        raise ValueError("SNOPT library path not set in environment variables.")

    # Solve the problem
    problem = pg.problem(problem)
    algorithm = pg.algorithm(ppnf.snopt7(library=lib, minor_version=7))
    algorithm_handle = algorithm.extract(ppnf.snopt7)
    _set_pygmo_options(algorithm_handle, combined_options)
    population = pg.population(problem, size=1)
    population.set_x(0, x0)
    population = algorithm.evolve(population)

    # Optimization output

    message = _extract_snopt_message(algorithm.get_extra_info())
    success = (
        True
        if message == "Finished successfully - optimality conditions satisfied"
        else False
    )

    return success, message


def _minimize_pygmo_worhp(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to WORHP"""

    # Define solver options
    default_options = {
        "Algorithm": 1,  # Set: 1 (SQP) or 2 (IP)
        "BFGSmethod": 0,  # Set 0 to use the classic dense BFGS method
        "TolOpti": 1e-3,
        "TolFeas": 1e-3,
        "MaxIter": 100,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options

    # Define WORHP path
    lib = os.getenv("WORHP_LIB")
    if not lib:
        raise ValueError("WORHP library path not set in environment variables.")

    # Solve the problem
    problem = pg.problem(problem)
    algorithm = pg.algorithm(ppnf.worhp(screen_output=False, library=lib))
    algorithm_handle = algorithm.extract(ppnf.worhp)
    _set_pygmo_options(algorithm_handle, combined_options)
    population = pg.population(problem, size=1)
    population.set_x(0, x0)
    population = algorithm.evolve(population)

    # Optimization output
    success = ""
    message = ""

    return success, message


def minimize_nlopt(problem, x0, method, options):
    """
    Optimize a given problem using the specified NLOpt optimization method.

    Parameters
    ----------
    problem : OptimizationProblem
        An object representing the problem.
    x0 : array_like
        Initial guess for the decision variables.
    method : str
        The name of the Pygmo solver to use. Must be one of the solvers defined in the NLOPT_SOLVERS list.
    options : dict
        Configuration options for the solver. Options vary depending on the solver used.

    Returns
    -------
    tuple
        A tuple (success, message) where 'success' is a boolean indicating if the optimization was successful, and 'message' is a string describing the outcome.

    Raises
    ------
    ValueError
        If an invalid solver name is specified.
    """

    # Check if method is valid
    if method not in NLOPT_SOLVERS:
        error_message = (
            f"Invalid solver: '{method}'. \nAvailable options:\n   - "
            + "\n   - ".join(NLOPT_SOLVERS)
            + "."
        )
        raise ValueError(error_message)

    # Define exit flag mapping
    exitflag_mapping = {
        1: "Generic success return value.",
        2: "Optimization stopped because stop value was reached.",
        3: "Optimization stopped because relative or absolute tolerance on function value was reached.",
        4: "Optimization stopped because relative or absolute tolerance on x was reached.",
        5: "Optimization stopped because maximum evaluations limit was reached.",
        6: "Optimization stopped because maximum time limit was reached.",
        -1: "Generic failure code.",
        -2: "Invalid arguments (e.g., lower bounds are bigger than upper bounds, unknown algorithm specified, etc.).",
        -3: "Ran out of memory.",
        -4: "Halted due to roundoff errors limiting progress. Optimization might still return a useful result.",
        -5: "Optimization was forcibly stopped.",
    }

    # Define solver options
    options = options.copy()  # Work with a copy to avoid side effects
    tol = options.pop("tol")
    max_iter = options.pop("max_iter")
    algorithm = pg.algorithm(pg.nlopt(solver=method))
    handle = algorithm.extract(pg.nlopt)
    handle.maxeval = max_iter
    handle.ftol_rel = tol
    handle.xtol_rel = tol

    # Solve the problem
    problem = pg.problem(problem)
    population = pg.population(problem, size=1)
    population.set_x(0, x0)
    population = algorithm.evolve(population)

    # Optimization output
    exitflag = handle.get_last_opt_result()
    message = exitflag_mapping.get(exitflag, "Unknown Status")
    success = exitflag in (1, 2, 3, 4)

    return success, message


def convert_pygmo_to_scipy_bounds(pygmo_bounds):
    """
    Create a function that converts bounds from Pygmo format to SciPy format.

    Parameters
    ----------
    pygmo_method : callable
        A method that, when called, returns a tuple containing two lists: the first list of lower bounds and the second of upper bounds (Pygmo convention).

    Returns
    -------
    callable
        A function that, when called, returns a list of tuples where each tuple contains the lower and upper bound for each decision variable (Scipy convention).

    """

    def converter():
        lower_bounds, upper_bounds = pygmo_bounds()
        scipy_bounds = list(zip(lower_bounds, upper_bounds))
        return scipy_bounds

    return converter


def _extract_snopt_message(info_str):
    # Search for the start of the relevant message
    start = info_str.find("Last optimisation return code: ")
    if start == -1:
        return "Optimization return code not found."

    # Adjust the start position to the actual start of the message
    start += len("Last optimisation return code: ")

    # Find the end of the line
    end = info_str.find("\n", start)
    if end == -1:
        end = len(info_str)

    # Extract and return the message
    return info_str[start:end].strip()


def _set_pygmo_options(handle, options):
    """
    Sets multiple options on a PyGMO solver handle based on the types of values provided in a dictionary.

    This function supports various PyGMO solvers including IPOPT, WORHP, and SNOPT. It dynamically calls
    the appropriate method to set each option (string, integer, or numeric) based on the type of each value
    in the `options` dictionary.

    Parameters
    ----------
    handle : object
        The PyGMO solver handle. This object should have methods for setting options,
        such as `set_string_option`, `set_integer_option`, and `set_numeric_option`.
    options : dict
        A dictionary where each key is the option name and each value is the option value.
        The function determines the method to use based on the type of the value:
        `str` for `set_string_option`, `int` for `set_integer_option`, and `float` for `set_numeric_option`.

    Raises
    ------
    ValueError
        If the value type is not supported (not an int, float, or str).

    Examples
    --------
    >>> algorithm = pg.algorithm(pg.ipopt())
    >>> handle = algorithm.extract(pg.ipopt)
    >>> options = {
    >>>     "sb": "yes",
    >>>     "print_level": 0,
    >>>     "tol": 1e-6,
    >>>     "max_iter": 1000
    >>> }
    >>> set_pygmo_options(handle, options)
    """
    for option_name, value in options.items():
        if isinstance(value, str):
            handle.set_string_option(option_name, value)
        elif isinstance(value, int):
            handle.set_integer_option(option_name, value)
        elif isinstance(value, float):
            handle.set_numeric_option(option_name, value)
        else:
            raise ValueError(
                "Unsupported option type: {} for option {}".format(
                    type(value), option_name
                )
            )
