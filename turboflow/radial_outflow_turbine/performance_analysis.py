import os
import yaml
import copy
import time
import datetime
import itertools
import numpy as np
import pandas as pd
import CoolProp as cp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import dill
import numbers

from scipy.stats import qmc
from scipy import optimize

from .. import math
from .. import pysolver_view as psv
from .. import utilities as utils
from .. import properties as props
from . import geometry_model as geom
from . import flow_model as flow
from . import choking_criterion as ch
from . import deviation_model as dm
from ..properties import perfect_gas_props
# from ..properties import perfect_gas_props_custom_jvp as perfect_gas_props
import jaxprop as jxp
import jaxprop.perfect_gas as pg

import turboflow as tf


jax.config.update(
    "jax_enable_x64", True
)  # By default jax uses 32 bit, for scientific computing we need 64 bit precision


SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}
"""
Available solvers for performance analysis.
"""

# -------------------------------------------------------------------
# NEW: helpers to fail fast on non-numeric contamination
# -------------------------------------------------------------------
NUMERIC = (int, float, np.floating)

def _is_num(x):
    return isinstance(x, NUMERIC)

def assert_numeric_operation_point(op):
    """
    Raise with a precise key if any OP value is not numeric (except 'fluid_name').
    """
    for k, v in op.items():
        if k == "fluid_name":
            if not isinstance(v, str):
                raise TypeError(f"operation_point['fluid_name'] must be str, got {type(v)}")
            continue
        if not _is_num(v):
            raise TypeError(f"operation_point['{k}'] must be numeric, got {v!r} ({type(v)})")

def assert_numeric_geometry(geom_dict):
    """
    Walks the prepared geometry (dict of arrays/scalars). Allows known text keys
    (e.g. 'cascade_type', 'camberline_type'); everything else must be numeric
    scalars/arrays. Raises with the exact offending path.
    """
    ALLOW_STR_KEYS = {"cascade_type", "camberline_type"}
    for k, v in geom_dict.items():
        if k in ALLOW_STR_KEYS:
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            for i, val in enumerate(v):
                if isinstance(val, dict):
                    for kk, vv in val.items():
                        if kk in ALLOW_STR_KEYS:
                            continue
                        if isinstance(vv, (list, tuple, np.ndarray)):
                            if not np.all([_is_num(x) for x in vv]):
                                raise TypeError(f"geometry['{k}'][{i}]['{kk}'] contains non-numerics: {vv}")
                        else:
                            if not _is_num(vv):
                                raise TypeError(f"geometry['{k}'][{i}]['{kk}'] must be numeric, got {vv!r} ({type(vv)})")
                else:
                    if not _is_num(val):
                        raise TypeError(f"geometry['{k}'][{i}] must be numeric, got {val!r} ({type(val)})")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if kk in ALLOW_STR_KEYS:
                    continue
                if isinstance(vv, (list, tuple, np.ndarray)):
                    if not np.all([_is_num(x) for x in vv]):
                        raise TypeError(f"geometry['{k}']['{kk}'] contains non-numerics: {vv}")
                else:
                    if not _is_num(vv):
                        raise TypeError(f"geometry['{k}']['{kk}'] must be numeric, got {vv!r} ({type(vv)})")
        else:
            if not _is_num(v):
                raise TypeError(f"geometry['{k}'] must be numeric, got {v!r} ({type(v)})")


def compute_performance(
    operation_points,
    config,
    out_filename=None,
    out_dir="output",
    stop_on_failure=False,
    export_results=True,
    logger=None
):
    r"""
    Compute and export the performance of each specified operation point to an Excel file.
    """

    # Check if geometry is provided
    if config["geometry"] is None:
        raise ValueError("Geometry is not provided")

    # Check the type of operation_points argument
    if isinstance(operation_points, dict):
        # Convert ranges to a list of operation points
        operation_points = generate_operation_points(operation_points)
    elif not isinstance(operation_points, (list, np.ndarray)):
        msg = "operation_points must be either list of dicts or a dict with ranges."
        raise TypeError(msg)

    # Validate all operation points (keys)
    for operation_point in operation_points:
        validate_operation_point(operation_point)

    # NEW: validate types (no strings except fluid_name)
    for op in operation_points:
        assert_numeric_operation_point(op)

    # Initialize lists to hold dataframes for each operation point
    operation_point_data = []
    overall_data = []
    plane_data = []
    cascade_data = []
    stage_data = []
    solver_data = []
    solution_data = []
    geometry_data = []
    solver_container = []

    # Loop through all operation points
    message = print_operation_points(operation_points)
    for line in message.splitlines():
        logger.info(line)

    for i, operation_point in enumerate(operation_points):
        logger.info("")
        logger.info(f" Computing operation point {i+1} of {len(operation_points)}")

        message = print_boundary_conditions(operation_point)
        for line in message.splitlines():
            logger.info(line)

        # Define initial guess
        if i == 0:
            # Use default initial guess for the first operation point
            initial_guess = config["performance_analysis"]["initial_guess"]
        else:
            closest_x, closest_index = find_closest_operation_point(
                operation_point,
                operation_points[:i],  # Use up to the previous point
                solution_data[:i],     # Use solutions up to the previous point
            )
            logger.info(f" Using solution from point {closest_index+1} as initial guess")
            initial_guess = closest_x

        # Compute performance
        solver, results = compute_single_operation_point(
            operation_point,
            initial_guess,
            config["geometry"],
            config["simulation_options"],
            config["performance_analysis"]["solver_options"],
            logger=logger
        )

        # Retrieve solver data
        solver_status = {
            "completed": True,
            "success": solver.success,
            "message": solver.message,
            "grad_count": solver.convergence_history["grad_count"][-1],
            "func_count": solver.convergence_history["func_count"][-1],
            "func_count_total": solver.convergence_history["func_count_total"][-1],
            "norm_residual": solver.convergence_history["norm_residual"][-1],
            "norm_step": solver.convergence_history["norm_step"][-1],
        }

        # Collect results
        operation_point_data.append(pd.DataFrame([operation_point]))
        overall_data.append(
            pd.DataFrame.from_dict(results["overall"], orient="index").T
        )
        plane_data.append(utils.flatten_dataframe(pd.DataFrame(results["planes"])))
        cascade_data.append(utils.flatten_dataframe(pd.DataFrame(results["cascades"])))
        stage_data.append(utils.flatten_dataframe(pd.DataFrame(results["stage"])))
        geometry_data.append(utils.flatten_dataframe(pd.DataFrame(results["geometry"])))
        solver_data.append(pd.DataFrame([solver_status]))
        solution_data.append(solver.problem.vars_real)
        solver_container.append(solver)

    # Dictionary to hold concatenated dataframes
    dfs = {
        "operation point": pd.concat(operation_point_data, ignore_index=True),
        "overall": pd.concat(overall_data, ignore_index=True),
        "plane": pd.concat(plane_data, ignore_index=True),
        "cascade": pd.concat(cascade_data, ignore_index=True),
        "stage": pd.concat(stage_data, ignore_index=True),
        "geometry": pd.concat(geometry_data, ignore_index=True),
        "solver": pd.concat(solver_data, ignore_index=True),
    }

    if export_results:
        # Create a directory to save simulation results
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Define filename with unique date-time identifier
        if out_filename is None:
            out_filename = "performance"

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filenames = [f"{out_filename}_{current_time}", f"{out_filename}_latest"]
        for out_filename in out_filenames:

            # Export simulation configuration as YAML file
            config_data = {k: v for k, v in config.items() if v}  # Filter empty entries
            config_data = utils.convert_numpy_to_python(config_data, precision=12)
            config_file = os.path.join(out_dir, f"{out_filename}.yaml")
            with open(config_file, "w") as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

            # Export optimal turbine in excel file
            filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                for sheet_name, df in dfs.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)

            # Export optimal turbine as dill object
            filepath = os.path.join(out_dir, f"{out_filename}.pkl")
            solver.problem = None
            with open(filepath, 'wb') as file:
                # Serialize the object and write it to the file
                dill.dump(solver, file)

        logger.info(f" Performance data successfully written to {filepath}")

    # Print final report
    message = print_simulation_summary(solver_container)
    for line in message:
        logger.info(line)

    return solver_container


def compute_single_operation_point(
    operating_point,
    initial_guess,
    geometry,
    simulation_options,
    solver_options,
    logger=None
):
    """
    Compute an operation point for a given set of boundary conditions using multiple solver methods and initial guesses.
    """

    # Initialize problem object
    problem = AxialTurbineProblem(geometry, simulation_options)
    # Update BC
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(solver_options)

    # Get initial guess from sample of heuristic guesses
    initial_guesses = get_initial_guess(
        initial_guess,
        problem,
        problem.boundary_conditions,
        problem.geometry,
        problem.fluid,
        simulation_options["choking_criterion"],
        simulation_options["deviation_model"],
        logger,
    )

    # Get solver method array
    solver_methods = [solver_options["method"]] + [
        method for method in SOLVER_MAP.keys() if method != solver_options["method"]
    ]

    for initial_guess in initial_guesses:
        initial_guess_scaled = problem.scale_values(initial_guess)
        x0 = np.array(list(initial_guess_scaled.values()))
        problem.keys = initial_guess_scaled.keys()

        # NEW: fail fast on non-finite initial guess
        if not np.all(np.isfinite(x0)):
            bad = {k: v for k, v in zip(problem.keys, x0) if not np.isfinite(v)}
            raise ValueError(f"Initial guess contains non-finite values: {bad}")

        for method in solver_methods:
            solver_options["method"] = method
            solver = psv.NonlinearSystemSolver(problem, logger=logger, **solver_options)
            try:
                solver.solve(x0)
            except Exception as e:
                if solver.func_count == 0:
                    raise e
                logger.info(f" Error during solving: {e}")
                solver.success = False
            if solver.success:
                break

        if solver.success:
            break

    if not solver.success:
        logger.info("WARNING: All attempts failed to converge")

    return solver, problem.results


def find_closest_operation_point(current_op_point, operation_points, solution_data):
    """
    Find the solution vector and index of the closest operation point in the historical data.
    """
    min_distance = float("inf")
    closest_point_x = None
    closest_index = None

    for i, op_point in enumerate(operation_points):
        distance = get_operation_point_distance(current_op_point, op_point)
        if distance < min_distance:
            min_distance = distance
            closest_point_x = solution_data[i]
            closest_index = i

    return closest_point_x, closest_index


def get_operation_point_distance(point_1, point_2, delta=1e-8):
    """
    Calculate the normalized distance between two operation points.
    """
    deviation_array = []
    for key in point_1:
        if isinstance(point_1[key], (int, float)) and key in point_2:
            value_1 = point_1[key]
            value_2 = point_2[key]

            if key == "alpha_in":
                deviation = np.abs(value_1 - value_2) / 90
            else:
                max_val = max(abs(value_1), abs(value_2), delta)
                deviation = abs(value_1 - value_2) / max_val

            deviation_array.append(deviation)

    return np.linalg.norm(deviation_array)


def generate_operation_points(performance_map):
    """
    Generates list of operation points from a map (Cartesian product).
    """
    # Make sure all values in the performance_map are iterables
    performance_map = {k: utils.ensure_iterable(v) for k, v in performance_map.items()}

    # Reorder performance map keys so first sweep is always through pressure
    priority_keys = ["p0_in", "p_out"]
    other_keys = [k for k in performance_map.keys() if k not in priority_keys]
    keys_order = other_keys + priority_keys
    performance_map = {
        k: performance_map[k] for k in keys_order if k in performance_map
    }

    # Create all combinations of operation points
    keys, values = zip(*performance_map.items())
    operation_points = [
        dict(zip(keys, combination)) for combination in itertools.product(*values)
    ]

    return operation_points


def validate_operation_point(op_point):
    """
    Validates that an operation point has exactly the required fields.
    """
    REQUIRED_FIELDS = {"fluid_name", "p0_in", "T0_in", "p_out", "alpha_in", "omega"}
    fields = set(op_point.keys())
    if fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - fields
        extra = fields - REQUIRED_FIELDS
        raise ValueError(
            f"Operation point validation error: "
            f"Missing fields: {missing}, Extra fields: {extra}"
        )


def get_initial_guess(
    initial_guess,
    problem,
    boundary_conditions,
    geometry,
    fluid,
    choking_criterion,
    deviation_model,
    logger
):
    # Rename variables
    number_of_cascades = geometry["number_of_cascades"]
    # Three types of initial guess:
    valid_keys_1 = ["efficiency_tt", "efficiency_ke"] + [
        f"ma_{i+1}" for i in range(number_of_cascades)
    ]
    valid_keys_2 = ["efficiency_tt", "efficiency_ke", "ma", "n_samples"]
    valid_keys_3 = [
        "w_out",
        "s_out",
        "beta_out",
        "w_crit_throat",
        "s_crit_throat",
    ]
    valid_keys_3 = ["v_in"] + [
        f"{key}_{i+1}" for i in range(number_of_cascades) for key in valid_keys_3
    ]
    valid_keys_4 = [
        "w_out",
        "s_out",
        "beta_out",
        "v_crit_in",
        "w_crit_throat",
        "s_crit_throat",
    ]
    valid_keys_4 = ["v_in"] + [
        f"{key}_{i+1}" for i in range(number_of_cascades) for key in valid_keys_4
    ]
    valid_keys_5 = ["w_out", "s_out", "beta_out", "w_crit_throat"]
    valid_keys_5 = ["v_in"] + [
        f"{key}_{i+1}" for i in range(number_of_cascades) for key in valid_keys_5
    ]
    check = []
    check.append(set(valid_keys_1) == set(list(initial_guess.keys())))
    check.append(set(valid_keys_2) == set(list(initial_guess.keys())))
    check.append(set(valid_keys_3) == set(list(initial_guess.keys())))
    check.append(set(valid_keys_4) == set(list(initial_guess.keys())))
    check.append(set(valid_keys_5) == set(list(initial_guess.keys())))

    if check[0]:
        if isinstance(initial_guess["efficiency_tt"], (list, np.ndarray)):
            initial_guesses = []
            for i in range(len(initial_guess["efficiency_tt"])):
                ma = np.array(
                    [initial_guess[f"ma_{j+1}"][i] for j in range(number_of_cascades)]
                )
                heuristic_guess = get_heuristic_guess(
                    initial_guess["efficiency_tt"][i],
                    initial_guess["efficiency_ke"][i],
                    ma,
                    boundary_conditions,
                    geometry,
                    fluid,
                    deviation_model,
                )
                initial_guesses.append(heuristic_guess)
        else:
            ma = np.array(
                [initial_guess[f"ma_{j+1}"] for j in range(number_of_cascades)]
            )
            heuristic_guess = get_heuristic_guess(
                initial_guess["efficiency_tt"],
                initial_guess["efficiency_ke"],
                ma,
                boundary_conditions,
                geometry,
                fluid,
                deviation_model,
            )
            initial_guesses = [heuristic_guess]
    elif check[1]:
        bounds = [initial_guess["efficiency_tt"], initial_guess["efficiency_ke"]] + [
            initial_guess["ma"] for i in range(number_of_cascades)
        ]
        n_samples = initial_guess["n_samples"]
        heuristic_inputs = latin_hypercube_sampling(bounds, n_samples)
        norm_residuals = np.array([])
        failures = 0
        for heuristic_input in heuristic_inputs:
            try:
                ma = [heuristic_input[i + 2] for i in range(number_of_cascades)]
                heuristic_guess = get_heuristic_guess(
                    heuristic_input[0],
                    heuristic_input[1],
                    ma,
                    boundary_conditions,
                    geometry,
                    fluid,
                    deviation_model,
                )
                x = problem.scale_values(heuristic_guess)
                problem.keys = x.keys()
                x0 = np.array(list(x.values()))
                residual = problem.residual(x0)
                norm_residuals = np.append(norm_residuals, np.linalg.norm(residual))
            except:
                failures += 1
                norm_residuals = np.append(norm_residuals, np.nan)

        logger.info(f"Generating heuristic inital guesses from latin hypercube sampling")
        logger.info(f"Number of failures: {failures} out of {n_samples} samples")
        logger.info(f"Least norm of residuals: {np.nanmin(norm_residuals)}")
        heuristic_input = heuristic_inputs[np.nanargmin(norm_residuals)]
        initial_guess = dict(zip(valid_keys_1, heuristic_input))
        ma = [heuristic_input[i + 2] for i in range(number_of_cascades)]
        initial_guess = get_heuristic_guess(
            heuristic_input[0],
            heuristic_input[1],
            ma,
            boundary_conditions,
            geometry,
            fluid,
            deviation_model,
        )
        initial_guesses = [initial_guess]
    elif check[2]:
        initial_guesses = [initial_guess]
    elif check[3]:
        initial_guesses = [initial_guess]
    elif check[4]:
        initial_guesses = [initial_guess]
    else:
        raise ValueError(
            "Initial guess must be a dictionary, which require a certain set of keys. See documentation for more information"
        )

    # Check that set of initial guess correspond with choking_criteria
    for initial_guess, i in zip(initial_guesses, range(len(initial_guesses))):
        if choking_criterion == "critical_mach_number":
            initial_guess = {
                key: val
                for key, val in initial_guess.items()
                if not key.startswith("v_crit_in")
            }
        elif choking_criterion == "critical_mass_flow_rate":
            initial_guess = {
                key: val
                for key, val in initial_guess.items()
                if not key.startswith("beta_crit_throat")
            }
        elif choking_criterion == "critical_isentropic_throat":
            initial_guess = {
                key: val
                for key, val in initial_guess.items()
                if not (key.startswith("v_crit_in") or key.startswith("s_crit_throat"))
            }

        initial_guesses[i] = initial_guess

    return initial_guesses

def _unwrap_yaml_geometry_rows(geometry_like):
    """
    Normalize 'geometry' into a flat list[dict], each row having at least 'cascade_type'.
    Accepts: full config dict (with 'geometry'), dict {'rows': ...}, dict of named rows,
    list of named-row dicts (e.g. [{'stator_1': {...}}, {'rotor_1': {...}}]),
    or already flat list of row dicts.
    Ignores non-row keys (e.g. 'turbomachinery', 'operation_points', ...).
    """
    # If full config, pick 'geometry'
    if isinstance(geometry_like, dict) and "geometry" in geometry_like:
        geometry = geometry_like["geometry"]
    else:
        geometry = geometry_like

    # Accept wrapper {'rows': ...}
    if isinstance(geometry, dict) and "rows" in geometry:
        geometry = geometry["rows"]

    # Case: dict of {name: row_dict} (filter only dicts w/ cascade_type)
    if isinstance(geometry, dict):
        rows = []
        for name, row in geometry.items():
            if not isinstance(row, dict):
                continue
            if "cascade_type" in row:
                rows.append({"name": name, **row})
        if rows:
            return rows
        # If we didn’t collect anything here, fall through to error later.

    # Case: list/tuple – flatten any {name: row_dict} entries
    if isinstance(geometry, (list, tuple)):
        rows = []
        for item in geometry:
            if isinstance(item, dict) and "cascade_type" in item:
                rows.append(item)
            elif isinstance(item, dict) and len(item) == 1:
                name, row = next(iter(item.items()))
                if isinstance(row, dict) and "cascade_type" in row:
                    rows.append({"name": name, **row})
            # else: ignore non-row entries silently
        if rows:
            return rows
        raise ValueError("No valid row entries with 'cascade_type' found in geometry list.")

    raise TypeError(f"'geometry' must be list/tuple or dict; got {type(geometry).__name__}")


def _coerce_rows_list(rows_like):
    """
    Final guard: ensure we return a *flat* list[dict] with 'cascade_type'.
    Also collapses one-level nested lists, and unwraps {name: row} dicts.
    """
    # Flatten one nesting layer if needed
    if isinstance(rows_like, list) and len(rows_like) == 1 and isinstance(rows_like[0], list):
        rows_like = rows_like[0]

    out = []
    # If dict -> iterate values
    iterable = rows_like.values() if isinstance(rows_like, dict) else rows_like

    for item in iterable:
        if isinstance(item, dict) and "cascade_type" in item:
            out.append(item)
        elif isinstance(item, dict) and len(item) == 1:
            _, inner = next(iter(item.items()))
            if isinstance(inner, dict) and "cascade_type" in inner:
                out.append(inner)
            else:
                raise TypeError("Found a named block whose value is not a valid row dict.")
        elif isinstance(item, list):
            # Accept a nested list, but items inside must be dict rows
            for sub in item:
                if isinstance(sub, dict) and "cascade_type" in sub:
                    out.append(sub)
                elif isinstance(sub, dict) and len(sub) == 1:
                    _, inner = next(iter(sub.items()))
                    if isinstance(inner, dict) and "cascade_type" in inner:
                        out.append(inner)
                    else:
                        raise TypeError("Nested named block is not a valid row dict.")
                else:
                    raise TypeError("Nested list contains a non-row item.")
        else:
            raise TypeError("Geometry contains an item that is neither a row dict nor a named-row dict.")

    if not out:
        raise ValueError("After coercion, no valid geometry rows with 'cascade_type' were found.")
    return out


# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #


class AxialTurbineProblem(psv.NonlinearSystemProblem):
    """
    Nonlinear system problem for cascade series analysis (radial-outflow compatible).
    """

    def __init__(self, geometry, simulation_options):
        # Unwrap YAML-named rows (stator_1, rotor_1, ...)
        unwrapped_rows = _unwrap_yaml_geometry_rows(geometry)
        rows_list = _coerce_rows_list(unwrapped_rows)

        # Prepare + compute full geometry using the expected row format
        prepared = geom.prepare_radial_outflow_geometry(rows_list)
        self.geometry = geom.calculate_full_radial_outflow_geometry(prepared)

        # Optional: keep your numeric guard if you added it
        # assert_numeric_geometry(self.geometry)

        self.model_options = simulation_options
        self.keys = []

    def residual(self, x):
        """
        Evaluate the system of equations for a given set of decision variables.
        """
        try:
            # Create dictionary of scaled variables
            self.vars_scaled = dict(zip(self.keys, x))

            # Create dictionary of real variables
            self.vars_real = self.scale_values(self.vars_scaled, to_normalized=False)

            # Evaluate cascade series
            self.results = flow.evaluate_axial_turbine(
                self.vars_scaled,
                self.boundary_conditions,
                self.geometry,
                self.fluid,
                self.model_options,
                self.reference_values,
            )

            return jnp.array(
                list(self.results["residuals"].values())
            )
        except Exception as e:
            # NEW: compact diagnostics to pinpoint type contamination quickly
            bc_types = {k: type(v).__name__ for k, v in getattr(self, "boundary_conditions", {}).items()}
            # Show first ~10 geometry entries type & shape
            geom_items = list(self.geometry.items())
            preview = {}
            for k, v in geom_items[:10]:
                if isinstance(v, (list, tuple, np.ndarray)):
                    try:
                        shp = np.array(v, dtype=object).shape
                    except Exception:
                        shp = None
                    preview[k] = (type(v).__name__, shp)
                else:
                    preview[k] = (type(v).__name__, None)

            raise TypeError(
                f"Residual failed: {e}\n"
                f"  OP types: {bc_types}\n"
                f"  First geometry entries (type,shape): {preview}\n"
                f"  Keys in vars_scaled: {list(self.vars_scaled.keys())}"
            ) from e

    def gradient(self, x):
        return jax.jacfwd(self.residual, argnums=0)(x)

    def update_boundary_conditions(self, operation_point):
        """
        Update boundary conditions and compute reference values.
        """
        # Validate OP types before using
        assert_numeric_operation_point(operation_point)

        # Define current operating point
        self.boundary_conditions = operation_point

        # Initialize fluid object
        self.fluid = jxp.FluidJAX(operation_point["fluid_name"])  # Using jaxprop CoolProp model

        # Rename variables
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        p_out = operation_point["p_out"]

        # Stagnation properties at inlet
        state_in_stag = self.fluid.get_props(jxp.PT_INPUTS, p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        # Store inlet stagnation (h,s)
        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Exit static properties (isentropic)
        state_out_s = self.fluid.get_props(jxp.PSmass_INPUTS, p_out, s_in)
        h_isentropic = state_out_s["h"]
        d_isentropic = state_out_s["d"]

        # Exit static properties (isenthalpic)
        state_out_h = self.fluid.get_props(jxp.HmassSmass_INPUTS, h0_in, p_out)
        s_isenthalpic = state_out_h["s"]

        # Spouting velocity
        v0 = np.sqrt(2 * (h0_in - h_isentropic))

        # Reference mass flow rate
        A_out = self.geometry["A_out"][-1]
        mass_flow_ref = A_out * v0 * d_isentropic

        # Reference values
        self.reference_values = {
            "s_range": s_isenthalpic - s_in,
            "s_min": s_in,
            "v0": v0,
            "h_out_s": h_isentropic,
            "d_out_s": d_isentropic,
            "mass_flow_ref": mass_flow_ref,
            "angle_range": 180,
            "angle_min": -90,
        }

        return

    def scale_values(self, variables, to_normalized=True):
        """
        Convert values between normalized and real values.
        """
        # Load parameters
        v0 = self.reference_values["v0"]
        s_range = self.reference_values["s_range"]
        s_min = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]

        # Define dictionary of scaled values
        scaled_variables = {}

        for key, val in variables.items():
            if key.startswith("v") or key.startswith("w"):
                scaled_variables[key] = val / v0 if to_normalized else val * v0
            elif key.startswith("s"):
                scaled_variables[key] = (
                    (val - s_min) / s_range if to_normalized else val * s_range + s_min
                )
            elif key.startswith("b"):
                scaled_variables[key] = (
                    (val - angle_min) / angle_range
                    if to_normalized
                    else val * angle_range + angle_min
                )

        return scaled_variables


def print_simulation_summary(solvers):
    """
    Print a formatted footer summarizing the performance of all operation points.
    """
    # Initialize times list and track failed points
    times = []
    failed_points = []

    for i, solver in enumerate(solvers):
        if solver and hasattr(solver, "elapsed_time"):
            times.append(solver.elapsed_time)
            if not solver.success:
                failed_points.append(i)
        else:
            failed_points.append(i)

    times = np.asarray(times)
    total_points = len(solvers)

    width = 80
    separator = "-" * width
    lines_to_output = [
        "",
        separator,
        "Final summary of performance analysis calculations".center(width),
        separator,
        f" Simulation successful for {total_points - len(failed_points)} out of {total_points} points",
    ]

    if failed_points:
        lines_to_output.append(
            f"Failed operation points: {', '.join(map(str, failed_points))}"
        )

    if times.size > 0:
        lines_to_output.extend(
            [
                f" Average calculation time per operation point: {np.mean(times):.3f} seconds",
                f" Minimum calculation time of all operation points: {np.min(times):.3f} seconds",
                f" Maximum calculation time of all operation points: {np.max(times):.3f} seconds",
                f" Total calculation time for all operation points: {np.sum(times):.3f} seconds",
            ]
        )
    else:
        lines_to_output.append(" No valid calculation times available.")

    lines_to_output.append(separator)
    lines_to_output.append("")

    return lines_to_output


def print_boundary_conditions(BC):
    """
    Pretty-print the boundary conditions.
    """
    column_width = 25
    lines = []
    lines.append("-" * 80)
    lines.append(" Operating point: ")
    lines.append("-" * 80)
    lines.append(f" {'Fluid: ':<{column_width}} {BC['fluid_name']:<}")
    lines.append(f" {'Flow angle in: ':<{column_width}} {BC['alpha_in']:<.2f} deg")
    lines.append(f" {'Total temperature in: ':<{column_width}} {BC['T0_in'] - 273.15:<.2f} degC")
    lines.append(f" {'Total pressure in: ':<{column_width}} {BC['p0_in'] / 1e5:<.3f} bar")
    lines.append(f" {'Static pressure out: ':<{column_width}} {BC['p_out'] / 1e5:<.3f} bar")
    lines.append(f" {'Angular speed: ':<{column_width}} {BC['omega'] * 60 / 2 / np.pi:<.1f} RPM")
    lines.append("-" * 80)
    lines.append("")
    result = "\n".join(lines)
    return result


def print_operation_points(operation_points):
    """
    Prints a summary table of operation points scheduled for simulation.
    """
    length = 80
    index_width = 8
    output_lines = [
        "-" * length,
        " Summary of operation points scheduled for simulation",
        "-" * length,
    ]

    field_specs = {
        "fluid_name": {"name": "Fluid", "unit": "", "width": 8},
        "alpha_in": {"name": "angle_in", "unit": "[deg]", "width": 10, "decimals": 1},
        "T0_in": {"name": "T0_in", "unit": "[degC]", "width": 12, "decimals": 2},
        "p0_in": {"name": "p0_in", "unit": "[kPa]", "width": 12, "decimals": 2},
        "p_out": {"name": "p_out", "unit": "[kPa]", "width": 12, "decimals": 2},
        "omega": {"name": "omega", "unit": "[RPM]", "width": 12, "decimals": 0},
    }

    header_str = f"{'Index':>{index_width}}"
    unit_str = f"{'':>{index_width}}"

    for spec in field_specs.values():
        header_str += f" {spec['name']:>{spec['width']}}"
        unit_str += f" {spec['unit']:>{spec['width']}}"

    output_lines.append(header_str)
    output_lines.append(unit_str)

    def convert_units(key, value):
        if key == "T0_in":
            return value - 273.15
        elif key == "omega":
            return (value * 60) / (2 * np.pi)
        elif key == "alpha_in":
            return np.degrees(value)
        elif key in ["p0_in", "p_out"]:
            return value / 1.0e3
        return value

    for index, op_point in enumerate(operation_points, start=1):
        row = [f"{index:>{index_width}}"]
        for key, spec in field_specs.items():
            value = convert_units(key, op_point[key])
            if isinstance(value, float):
                row.append(f"{value:>{spec['width']}.{spec['decimals']}f}")
            else:
                row.append(f"{value:>{spec['width']}}")
        output_lines.append(" ".join(row))

    output_lines.append("-" * length)
    formatted_output = "\n".join(output_lines)
    return formatted_output


# -------------------------------------------------------------------
# FIXED: proper enum resolution for jaxprop/CoolProp calls
# -------------------------------------------------------------------
def calculate_enthalpy_residual_1(prop1, scale, h0, Ma, fluid, call, prop2):
    """
    Residual for enthalpy balance given a guessed prop1 (scaled).
    `call` can be 'DmassP_INPUTS', 'PSmass_INPUTS', etc.
    Resolves first via jaxprop (preferred), then CoolProp.
    """
    if isinstance(call, str):
        call_attr = getattr(jxp, call, None)
        if call_attr is None:
            try:
                call_attr = getattr(cp, call)
            except AttributeError:
                raise ValueError(f"Invalid input type: {call}")
        call = call_attr

    props = fluid.get_props(call, prop1 * scale, prop2)
    return props["h"] - h0 + 0.5 * Ma**2 * props["speed_sound"] ** 2


def get_unknown(prop1, scale, h0, Ma, fluid, call, prop2):
    """
    Find prop1 (scaled by 'scale') such that enthalpy residual is zero.
    """
    sol = optimize.root_scalar(
        calculate_enthalpy_residual_1,
        args=(scale, h0, Ma, fluid, call, prop2),
        method="secant",
        x0=prop1,
    )
    return sol.root * scale


def get_heuristic_guess(
    efficiency_tt,
    efficiency_ke,
    mach,
    boundary_conditions,
    geometry,
    fluid,
    deviation_model,
):
    p0_first = boundary_conditions["p0_in"]
    T0_first = boundary_conditions["T0_in"]
    p_final = boundary_conditions["p_out"]
    angular_speed = boundary_conditions["omega"]
    alpha_first = boundary_conditions["alpha_in"]
    number_of_cascades = geometry["number_of_cascades"]

    # First stagnation properties
    stag_first = fluid.get_props(jxp.PT_INPUTS, p0_first, T0_first)
    h0_first = stag_first["h"]
    s_first = stag_first["s"]
    d0_first = stag_first["d"]

    # Final isentropic
    static_is = fluid.get_props(jxp.PSmass_INPUTS, p_final, s_first)
    h_final_s = static_is["h"]
    a_final_s = static_is["speed_sound"]

    # Spouting velocity
    v0 = np.sqrt(2 * (h0_first - h_final_s))

    # Exit enthalpy with guessed efficiency
    efficiency_ts = efficiency_tt / (1 + efficiency_tt * efficiency_ke)
    h0_final = h0_first - efficiency_ts * (h0_first - h_final_s)
    v_final = np.sqrt(2 * (h0_first - h_final_s - (h0_first - h0_final) / efficiency_tt))
    h_final = h0_final - 0.5 * v_final**2

    # Exit static state for expansion with guessed efficiency
    static_properties_exit = fluid.get_props(jxp.HmassP_INPUTS, h_final, p_final)
    s_final = static_properties_exit["s"]

    # Linear entropy distribution
    entropy_distribution = np.linspace(s_first, s_final, number_of_cascades + 1)[1:]

    # Initial guess dictionary
    initial_guess = {}

    # Initialize inlet calculation
    s_in = s_first
    rothalpy = h0_first
    alpha_in = alpha_first
    d_in = d0_first

    for i in range(number_of_cascades):
        geometry_cascade = {
            key: values[i]
            for key, values in geometry.items()
            if key not in ["number_of_cascades", "number_of_stages"]
        }

        radius_mean_in = geometry_cascade["radius_mean_in"]
        radius_mean_throat = geometry_cascade["radius_mean_throat"]
        radius_mean_out = geometry_cascade["radius_mean_out"]
        A_throat = geometry_cascade["A_throat"]
        A_out = geometry_cascade["A_out"]
        A_in = geometry_cascade["A_in"]

        # Entropy and Mach for this cascade
        s_out = entropy_distribution[i]
        ma_out = mach[i]

        # Exit pressure from guessed Ma (via PS with guessed s_out)
        blade_speed_out = angular_speed * (i % 2) * radius_mean_out
        h0_rel_out = rothalpy + 0.5 * blade_speed_out**2
        p_out = get_unknown(
            1.0, p0_first, h0_rel_out, ma_out, fluid, "PSmass_INPUTS", s_out
        )

        # Exit state
        static_out = fluid.get_props(jxp.PSmass_INPUTS, p_out, s_out)
        h_out = static_out["h"]
        a_out = static_out["speed_sound"]
        d_out = static_out["d"]
        gamma_out = static_out["gamma"]

        # Exit velocity
        w_out = np.sqrt(2 * (h0_rel_out - h_out))

        # Critical Mach (placeholder 1.0; model available if wanted)
        static_props_is = fluid.get_props(jxp.PSmass_INPUTS, p_out, s_in)
        h_out_s = static_props_is["h"]
        eta = (h0_rel_out - h_out) / (h0_rel_out - h_out_s)
        ma_crit = 1.0

        # Exit flow angle
        beta_out = (-1) ** i * dm.get_subsonic_deviation(
            ma_out, ma_crit, {"A_throat": A_throat, "A_out": A_out}, deviation_model
        )

        # Mass flow rate
        mass_flow = d_out * w_out * math.cosd(beta_out) * A_out

        # Critical state at throat (for guess)
        w_throat_crit = a_out * ma_crit
        h_throat_crit = h0_rel_out - 0.5 * w_throat_crit**2
        s_throat_crit = s_out
        static_state_throat_crit = fluid.get_props(jxp.HmassSmass_INPUTS, h_throat_crit, s_throat_crit)
        rho_throat_crit = static_state_throat_crit["d"]
        m_crit = w_throat_crit * rho_throat_crit * A_throat
        w_m_in_crit = m_crit / d_in / A_in
        v_in_crit = w_m_in_crit / math.cosd(alpha_in)

        # Store initial guess
        index = f"_{i+1}"
        initial_guess.update(
            {
                "w_out" + index: w_out,
                "s_out" + index: s_out,
                "beta_out" + index: (-1) ** i * math.arccosd(A_throat / A_out),
                "v_crit_in" + index: v_in_crit,
                "w_crit_throat" + index: w_throat_crit,
                "s_crit_throat" + index: s_throat_crit,
            }
        )

        # Update variables for next cascade
        if i != (number_of_cascades - 1):
            A_next = geometry["A_in"][i + 1]
            radius_mean_next = geometry["radius_mean_in"][i + 1]
            velocity_triangle_out = flow.evaluate_velocity_triangle_out(
                blade_speed_out, w_out, beta_out
            )
            v_m_in = velocity_triangle_out["v_m"] * A_out / A_next
            v_t_in = velocity_triangle_out["v_t"] * radius_mean_out / radius_mean_next
            v_in = np.sqrt(v_m_in**2 + v_t_in**2)
            alpha_in = math.arctand(v_t_in / v_m_in)
            blade_speed_in = angular_speed * ((i + 1) % 2) * radius_mean_next
            velocity_triangle_in = flow.evaluate_velocity_triangle_in(
                blade_speed_in, v_in, alpha_in
            )
            h0_in = h_out + 0.5 * velocity_triangle_out["v"] ** 2
            h_in = h0_in - 0.5 * v_in**2
            rothalpy = (
                h_in + 0.5 * velocity_triangle_in["w"] ** 2 - 0.5 * blade_speed_in**2
            )
            s_in = s_out
            static_in = fluid.get_props(jxp.HmassSmass_INPUTS, h_in, s_in)
            d_in = static_in["d"]

    # Inlet velocity from mass flow
    initial_guess["v_in"] = mass_flow / (
        d0_first * geometry["A_in"][0] * math.cosd(alpha_first)
    )

    return initial_guess


def latin_hypercube_sampling(bounds, n_samples):
    """
    Generates samples using Latin Hypercube Sampling.
    """
    n_variables = len(bounds)
    sampler = qmc.LatinHypercube(d=n_variables, seed=1)
    unit_hypercube_samples = sampler.random(n=n_samples)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(unit_hypercube_samples, lower_bounds, upper_bounds)
    return scaled_samples
