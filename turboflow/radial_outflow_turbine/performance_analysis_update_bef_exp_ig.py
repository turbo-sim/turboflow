# performance_analysis.py  (mapping-based dispatch; component-wise geometry; tree_at updates)

from __future__ import annotations

import os
import yaml
import copy
import datetime
import itertools
import pandas as pd

import time
import jax
import jax.numpy as jnp
import equinox as eqx

from scipy.stats import qmc
from scipy import optimize

from .. import math
from .. import pysolver_view as psv
from .. import utilities as utils
from . import flow_model_update as flow
from . import deviation_model as dm
import jaxprop as jxp


import turboflow as tf

from .blade_row import (
    evaluate_velocity_triangle_out,
    evaluate_velocity_triangle_in,
)
from .blade_row import BladeRow
from .vaneless_channel import VanelessChannel
from .interspace_model import Interspace

jax.config.update("jax_enable_x64", True)

SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}
NUMERIC = (int, float, jnp.floating, jnp.integer)


# =========================== mappings ===========================

# Component type → class
COMPONENT_CLASSES = {
    "axial_cascade": BladeRow,
    "radial_cascade": BladeRow,
    "vaneless_channel": VanelessChannel,
}


# ====================== small helpers ======================


def _is_num(x):
    if isinstance(x, NUMERIC):
        return True
    if isinstance(x, jax.Array):
        return x.ndim == 0
    return False


def assert_numeric_operation_point(op):
    for k, v in op.items():
        if not _is_num(v):
            raise TypeError(
                f"operation_point['{k}'] must be numeric, got {v!r} ({type(v)})"
            )


def _looks_like_expr(s: str) -> bool:
    return any(t in s for t in ("jnp.", "(", ")", "[", "]", "*", "/", "+", "-", "**"))


def _eval_item_if_str(x, ctx):
    if isinstance(x, str):
        if not _looks_like_expr(x):
            return x
        try:
            return eval(x, {"__builtins__": {}, "jnp": jnp}, ctx)
        except Exception:
            return x
    elif isinstance(x, list):
        return [_eval_item_if_str(e, ctx) for e in x]
    elif isinstance(x, tuple):
        return tuple(_eval_item_if_str(e, ctx) for e in x)
    else:
        return x


def _evaluate_map_expressions(performance_map):
    pm = dict(performance_map)
    ctx = {k: v for k, v in pm.items() if isinstance(v, (int, float))}
    ctx.update(
        {
            k: v
            for k, v in pm.items()
            if isinstance(v, list) and all(isinstance(e, (int, float)) for e in v)
        }
    )
    for k, v in list(pm.items()):
        evaluated = _eval_item_if_str(v, ctx)
        if hasattr(evaluated, "tolist"):
            evaluated = evaluated.tolist()
        pm[k] = evaluated
        if isinstance(evaluated, (int, float)):
            ctx[k] = evaluated
        elif isinstance(evaluated, list) and all(
            isinstance(e, (int, float)) for e in evaluated
        ):
            ctx[k] = evaluated
    return pm


def _numpy_to_native(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    elif isinstance(x, (list, tuple)):
        return type(x)(_numpy_to_native(e) for e in x)
    else:
        return x
    
def initialize_fluid_from_config(fluid_config):
    fluid_name = fluid_config.get("name")
    model = fluid_config.get("model")
    model_options = fluid_config.get("model_options")
    if model == "perfect_gas":
        fluid = jxp.FluidPerfectGas(fluid_name, model_options["T_ref"], model_options["p_ref"])
    elif model == "bicubic":
        fluid = jxp.FluidBicubic(fluid_name, 
                                backend= model_options.get("backend"),
                                h_min=model_options.get("h_min"),
                                h_max=model_options.get("h_max"),
                                p_min=model_options.get("p_min"),
                                p_max=model_options.get("p_max"),
                                N_h=model_options.get("N_h"),
                                N_p=model_options.get("N_p"),
                                )
    elif model == "coolprop":
        fluid = jxp.FluidJAX(fluid_name,
                             backend= model_options.get("backend"),
                             )
    else:
        raise ValueError(f"Unknown fluid model: {model}")
    
    return fluid



def generate_operation_points(performance_map):
    performance_map = _evaluate_map_expressions(performance_map)
    performance_map = {k: utils.ensure_iterable(v) for k, v in performance_map.items()}
    priority_keys = ["p0_in", "p_out"]
    other_keys = [k for k in performance_map.keys() if k not in priority_keys]
    keys_order = other_keys + [k for k in priority_keys if k in performance_map]
    perf_map_ordered = {k: performance_map[k] for k in keys_order}
    keys, values = zip(*perf_map_ordered.items())
    base_combos = itertools.product(*values)
    operation_points = []
    for combo in base_combos:
        op = dict(zip(keys, combo))
        if "p_out" not in op:
            raise ValueError("Each operation point must define 'p_out'.")
        operation_points.append(op)
    return operation_points


def validate_operation_point(op_point):
    REQUIRED_FIELDS = {"p0_in", "T0_in", "p_out", "alpha_in", "omega"}
    fields = set(op_point.keys())
    if fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - fields
        extra = fields - REQUIRED_FIELDS
        raise ValueError(
            f"Operation point validation error: Missing fields: {missing}, Extra fields: {extra}"
        )


def print_operation_points(operation_points):
    length = 80
    index_width = 8
    output = [
        "-" * length,
        " Summary of operation points scheduled for simulation",
        "-" * length,
    ]
    field_specs = {
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
    output.append(header_str)
    output.append(unit_str)

    def convert_units(key, value):
        if key == "T0_in":
            return value - 273.15
        if key == "omega":
            return (value * 60) / (2 * jnp.pi)
        if key == "alpha_in":
            # return jnp.degrees(value)
            return value
        if key in ["p0_in", "p_out"]:
            return value / 1.0e3
        return value

    for index, op in enumerate(operation_points, start=1):
        row = [f"{index:>{index_width}}"]
        for key, spec in field_specs.items():
            val = convert_units(key, op[key])
            if isinstance(val, float):
                row.append(f"{val:>{spec['width']}.{spec['decimals']}f}")
            else:
                row.append(f"{val:>{spec['width']}}")
        output.append(" ".join(row))
    output.append("-" * length)
    return "\n".join(output)


def print_boundary_conditions(BC):
    column_width = 25
    lines = []
    lines.append("-" * 80)
    lines.append(" Operating point: ")
    lines.append("-" * 80)
    lines.append(f" {'Flow angle in: ':<{column_width}} {BC['alpha_in']:<.2f} deg")
    lines.append(
        f" {'Total temperature in: ':<{column_width}} {BC['T0_in'] - 273.15:<.2f} degC"
    )
    lines.append(
        f" {'Total pressure in: ':<{column_width}} {BC['p0_in'] / 1e5:<.3f} bar"
    )
    lines.append(
        f" {'Static pressure out: ':<{column_width}} {BC['p_out'] / 1e5:<.3f} bar"
    )
    lines.append(
        f" {'Angular speed: ':<{column_width}} {BC['omega'] * 60 / 2 / jnp.pi:<.1f} RPM"
    )
    lines.append("-" * 80)
    lines.append("")
    return "\n".join(lines)


def print_simulation_summary(solvers):
    width = 80
    sep = "-" * width
    times, failed_points = [], []

    for i, solver in enumerate(solvers):
        if solver is None:
            failed_points.append(i)
            continue

        ok = getattr(solver, "success", False)
        if not ok:
            failed_points.append(i)

        t = getattr(solver, "elapsed_time", None)
        if t is not None:
            try:
                times.append(float(t))
            except Exception:
                # Ignore non-castable timings
                pass

    total_points = len(solvers)
    lines = [
        "",
        sep,
        "Final summary of performance analysis calculations".center(width),
        sep,
        f" Simulation successful for {total_points - len(failed_points)} out of {total_points} points",
    ]

    if failed_points:
        lines.append(
            f" Failed operation points: {', '.join(map(str, failed_points))}"
        )

    if times:
        # Convert list → JAX array once, then use jnp reductions
        times_arr = jnp.asarray(times, dtype=jnp.float64)

        avg_time = jnp.mean(times_arr)
        min_time = jnp.min(times_arr)
        max_time = jnp.max(times_arr)
        sum_time = jnp.sum(times_arr)

        lines.extend(
            [
                f" Average calculation time per operation point: {avg_time:.3f} seconds",
                f" Minimum calculation time of all operation points: {min_time:.3f} seconds",
                f" Maximum calculation time of all operation points: {max_time:.3f} seconds",
                f" Total calculation time for all operation points:   {sum_time:.3f} seconds",
            ]
        )
    else:
        lines.append(" No valid calculation times available.")

    lines.append(sep)
    lines.append("")
    return lines

def latin_hypercube_sampling(bounds, n_samples):
    n_variables = len(bounds)
    sampler = qmc.LatinHypercube(d=n_variables, seed=1)
    unit_samples = sampler.random(n=n_samples)
    lower_bounds = jnp.array([b[0] for b in bounds])
    upper_bounds = jnp.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, lower_bounds, upper_bounds)

# ============================ public API ============================

def compute_performance(
    operation_points,
    config,
    out_filename=None,
    out_dir="output",
    stop_on_failure=False,
    export_results=True,
    logger=None,
    use_previous_solution=False,
    previous_pkl_path=None,
    require_previous_pkl_converged=True,
):
    if not config.get("components"):
        raise ValueError(
            "No 'components' found in config. Provide a list of components."
        )

    # Expand operation_points if given as a dict (performance map)
    if isinstance(operation_points, dict):
        operation_points = generate_operation_points(operation_points)
    elif not isinstance(operation_points, (list, jnp.ndarray)):
        raise TypeError(
            "operation_points must be either list of dicts or a dict with ranges."
        )

    # Validate all operation points
    for op in operation_points:
        validate_operation_point(op)
        assert_numeric_operation_point(op)

    operation_point_data, overall_data = [], []
    plane_data, cascade_data, stage_data = [], [], []
    solver_data, solution_data, geometry_data = [], [], []
    solver_container = []

    # Print operation point summary
    message = print_operation_points(operation_points)
    for line in message.splitlines():
        logger.info(line)

    # Keep latest converged solution (real/unscaled) for warm-starting.
    # Failed points must never overwrite this.
    last_converged_initial_guess_real = None

    # Loop over all operation points in the map / list
    for i, operation_point in enumerate(operation_points):
        logger.info("")
        logger.info(f" Computing operation point {i+1} of {len(operation_points)}")
        for line in print_boundary_conditions(operation_point).splitlines():
            logger.info(line)

        # Initialize fluid for this run
        fluid = initialize_fluid_from_config(config["fluid"])

        # --------------------------------------------------------------
        # Decide initial guess for this operation point
        #
        # Rules:
        # - If i == 0:
        #     - if use_previous_solution and previous_pkl_path given:
        #           use that PKL solution as initial guess
        #     - else:
        #           build initial guess from components
        # - If i > 0:
        #     - use the most recent converged solution from this run
        #       (if available), otherwise build from components
        # --------------------------------------------------------------
        initial_guess_real = None

        if i == 0:
            if use_previous_solution and previous_pkl_path is not None:
                import dill

                with open(previous_pkl_path, "rb") as f:
                    prev_solver = dill.load(f)

                prev_success = getattr(prev_solver, "success", False)
                keys_prev = getattr(prev_solver, "solution_keys", None)
                x_real_prev = getattr(prev_solver, "x_solution_real", None)

                has_data = (keys_prev is not None) and (x_real_prev is not None)
                can_use_prev = has_data and (
                    (not require_previous_pkl_converged) or prev_success
                )

                if can_use_prev:
                    initial_guess_real = {k: v for k, v in zip(keys_prev, x_real_prev)}
                    if require_previous_pkl_converged:
                        logger.info(
                            f" Using previous converged solution from '{previous_pkl_path}' "
                            f"as initial guess for operation point {i+1}"
                        )
                    else:
                        logger.info(
                            f" Using previous PKL solution from '{previous_pkl_path}' "
                            f"(convergence flag ignored) as initial guess for operation point {i+1}"
                        )
                        if not prev_success:
                            logger.warning(
                                " Loaded previous PKL is marked unconverged, "
                                "but it is accepted because "
                                "'require_previous_pkl_converged=False'."
                            )
                elif not has_data:
                    logger.warning(
                        " Previous PKL solution is missing required fields "
                        "('solution_keys' and/or 'x_solution_real'). "
                        "Falling back to component-based initial guess."
                    )
                else:
                    logger.warning(
                        " Previous PKL solution is not converged. "
                        "Falling back to component-based initial guess."
                    )
                    logger.info(
                        " Set 'require_previous_pkl_converged=False' to force-accept "
                        "the external PKL initial guess for the first operation point."
                    )
            else:
                logger.info(
                    f" Building initial guess from components for operation point {i+1}"
                )
        else:
            if last_converged_initial_guess_real is not None:
                initial_guess_real = copy.deepcopy(last_converged_initial_guess_real)
                logger.info(
                    f" Using most recent converged solution "
                    f"as initial guess for operation point {i+1}"
                )
            else:
                logger.warning(
                    f" No converged previous solution available before operation point {i+1}. "
                    "Using component-based initial guess."
                )

        # --------------------------------------------------------------
        # Solve single operation point.
        # If warm-start fails, retry once using component/YAML initial guess.
        # --------------------------------------------------------------
        warm_start_attempted = initial_guess_real is not None
        fallback_attempted = False
        fallback_success = False

        solver, results = compute_single_operation_point(
            operation_point,
            fluid,
            config["components"],
            config.get("simulation_options", {}),
            config["performance_analysis"]["solver_options"],
            logger=logger,
            initial_guess_real=initial_guess_real,
        )

        if warm_start_attempted and (not solver.success):
            fallback_attempted = True
            logger.warning(
                f" Warm-start failed at operation point {i+1}. "
                "Retrying with component-based initial guess."
            )

            retry_solver, retry_results = compute_single_operation_point(
                operation_point,
                fluid,
                config["components"],
                config.get("simulation_options", {}),
                config["performance_analysis"]["solver_options"],
                logger=logger,
                initial_guess_real=None,
            )

            if retry_solver.success:
                fallback_success = True
                logger.info(
                    f" Retry with component-based initial guess converged "
                    f"for operation point {i+1}."
                )
            else:
                logger.warning(
                    f" Retry with component-based initial guess also failed "
                    f"for operation point {i+1}."
                )

            # Keep retry result as final result for this operation point.
            solver, results = retry_solver, retry_results

        solver_status = {
            "completed": True,
            "success": solver.success,
            "message": solver.message,
            "warm_start_attempted": warm_start_attempted,
            "fallback_attempted": fallback_attempted,
            "fallback_success": fallback_success,
            "grad_count": solver.convergence_history["grad_count"][-1],
            "func_count": solver.convergence_history["func_count"][-1],
            "func_count_total": solver.convergence_history["func_count_total"][-1],
            "norm_residual": solver.convergence_history["norm_residual"][-1],
            "norm_step": solver.convergence_history["norm_step"][-1],
        }

        # Collect data for this OP
        operation_point_data.append(pd.DataFrame([operation_point]))
        overall_data.append(
            pd.DataFrame.from_dict(results["overall"], orient="index").T
        )
        plane_data.append(utils.flatten_dataframe(pd.DataFrame(results["planes"])))
        cascade_data.append(utils.flatten_dataframe(pd.DataFrame(results["cascades"])))
        stage_data.append(utils.flatten_dataframe(pd.DataFrame(results["stage"])))
        geom_rows_df = pd.DataFrame(results["geometry_components"])
        geometry_data.append(utils.flatten_dataframe(geom_rows_df))
        solver_data.append(pd.DataFrame([solver_status]))

        # Store real (unscaled) solution vectors.
        solution_data.append(solver.problem.vars_real)
        solver_container.append(solver)

        # Update warm-start state ONLY for converged points.
        if solver.success:
            sol_keys = solver.problem.keys
            sol_vals_real = solver.problem.vars_real
            last_converged_initial_guess_real = {
                k: v for k, v in zip(sol_keys, sol_vals_real)
            }
        else:
            logger.warning(
                f" Operation point {i+1} did not converge; "
                "its solution will not be used to initialize subsequent points."
            )

    # --------------------------------------------------------------
    # Aggregate results into DataFrames
    # --------------------------------------------------------------
    dfs = {
        "operation point": pd.concat(operation_point_data, ignore_index=True),
        "overall": pd.concat(overall_data, ignore_index=True),
        "plane": pd.concat(plane_data, ignore_index=True),
        "cascade": pd.concat(cascade_data, ignore_index=True),
        "stage": pd.concat(stage_data, ignore_index=True),
        "geometry": pd.concat(geometry_data, ignore_index=True),
        "solver": pd.concat(solver_data, ignore_index=True),
    }

    # --------------------------------------------------------------
    # Export results (YAML, XLSX, PKL)
    # --------------------------------------------------------------
    if export_results:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if out_filename is None:
            out_filename = "performance"

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filenames = [f"{out_filename}_{current_time}", f"{out_filename}_latest"]

        # Attach solution from the last converged operation point (if any).
        last_converged_idx = None
        for idx in range(len(solver_container) - 1, -1, -1):
            if getattr(solver_container[idx], "success", False):
                last_converged_idx = idx
                break

        if last_converged_idx is None:
            solver = solver_container[-1]
            logger.warning(
                " No converged operation point found. "
                "Exporting latest (unconverged) solver state."
            )
        else:
            solver = solver_container[last_converged_idx]
            logger.info(
                f" Exporting solution from last converged operation point "
                f"{last_converged_idx + 1}"
            )

        solver.x_solution_scaled = copy.deepcopy(solver.x_final)           # scaled vector
        solver.x_solution_real   = copy.deepcopy(solver.problem.vars_real) # unscaled physical vector
        solver.solution_keys     = copy.deepcopy(solver.problem.keys)      # variable names

        # Optionally: keep last results too, if you want them in the pickle
        # solver.solution_results = copy.deepcopy(solver.problem.results)

        # Remove problem to keep pickle light / avoid recursive structures
        solver.problem = None

        import dill

        for fname in out_filenames:
            # YAML config export
            config_data = {k: v for k, v in config.items() if v}
            config_data = utils.convert_numpy_to_python(config_data, precision=12)
            with open(os.path.join(out_dir, f"{fname}.yaml"), "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            # XLSX export
            filepath_xlsx = os.path.join(out_dir, f"{fname}.xlsx")
            with pd.ExcelWriter(filepath_xlsx, engine="openpyxl") as writer:
                for sheet_name, df in dfs.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)

            # PKL export (contains last solver with attached solution fields)
            filepath_pkl = os.path.join(out_dir, f"{fname}.pkl")
            with open(filepath_pkl, "wb") as f:
                dill.dump(solver, f)

        logger.info(f" Performance data successfully written to {filepath_xlsx}")

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------
    message = print_simulation_summary(solver_container)
    for line in message:
        logger.info(line)

    return solver_container

# ================= one operation point (solver) =================

def compute_single_operation_point(
    operating_point,
    fluid,
    components,
    simulation_options,
    solver_options,
    logger=None,
    initial_guess_real=None,   # <-- NEW
):
    problem = TurbomachineryProblem(components, simulation_options, fluid)
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(solver_options)

    # ------------------------------------------------------------------
    # Build initial guess
    #   1) If initial_guess_real is provided: use it.
    #   2) Else: build per-component initial guess (original logic).
    # ------------------------------------------------------------------

    # Pick an epsilon so arctanh never sees ±1
    bound_eps = float(simulation_options.get("bound_scaled_eps", 1e-3))
    bound_eps = max(1e-12, bound_eps)
    z_lo, z_hi = -1.0 + bound_eps, 1.0 - bound_eps

    if initial_guess_real is not None:
        # initial_guess_real is a dict: {var_name: real_value}
        initial_guess_scaled = problem.scale_values(initial_guess_real)
        x0 = jnp.array(list(initial_guess_scaled.values()), dtype=float)
        problem.keys = list(initial_guess_scaled.keys())

        # z0 = jnp.array(list(initial_guess_scaled.values()), dtype=jnp.float64)
        # z0 = jnp.clip(z0, z_lo, z_hi)

        # # solver works on y in R, we bound inside residual with z=tanh(y)
        # x0 = jnp.arctanh(z0).astype(float)

        if not jnp.all(jnp.isfinite(x0)):
            bad = {k: v for k, v in zip(problem.keys, x0) if not jnp.isfinite(v)}
            raise ValueError(
                f"Initial guess (from previous solution) contains non-finite values: {bad}"
            )

        if logger:
            logger.info(" Using previous solution as initial guess")

    else:
        # ------- ORIGINAL per-row initial guess from components -------
        omega = problem.boundary_conditions["omega"]
        # alpha_in = problem.boundary_conditions["alpha_in"]
        alpha_in_deg = problem.boundary_conditions["alpha_in"]
        # alpha_in_deg = (
        #     jnp.degrees(alpha_in) if abs(alpha_in) <= jnp.pi * 1.01 else alpha_in
        # )

        inlet_seed = {
            "h0_in": problem.boundary_conditions["h0_in"],
            "s_in": problem.boundary_conditions["s_in"],
            "alpha_in": alpha_in_deg,
            "v_in": 0.1 * problem.reference_values["v0"],
            "p0_in": problem.boundary_conditions.get("p0_in"),
        }

        jax.debug.print("Initial inlet seed: {inlet_seed}", inlet_seed=inlet_seed)

        row_guess_dict: Dict[str, Any] = {}
        cascade_index = 0  # only counts BladeRow components

        omega_global_jax = jnp.asarray(omega, dtype=jnp.float64)

        for obj in problem.comp_objects:
            # Decide the angular speed seen by this component
            if isinstance(obj, BladeRow):
                # rotor rows rotate; stators do not
                is_rotor = "rotor" in str(obj.cascade_type).lower()
                omega_i = omega_global_jax if is_rotor else jnp.asarray(0.0)
                cascade_index += 1
                row_index = cascade_index
            else:
                # non-cascade components (e.g. VanelessChannel) ignore row_index and omega
                omega_i = omega_global_jax
                row_index = 0

            use_pr_zeta = (
                isinstance(obj, BladeRow)
                and obj.initial_guess_spec is not None
                and {"PR_ts", "zeta_h"} <= set(obj.initial_guess_spec.keys())
            )

            if use_pr_zeta:
                ig_row, inlet_seed = obj.build_initial_guess_pr_zeta(
                    inlet_state=inlet_seed,
                    omega=omega_i,
                    row_index=row_index,
                )
            else:
                ig_row, inlet_seed = obj.build_initial_guess(
                    inlet_state=inlet_seed,
                    omega=omega_i,
                    row_index=row_index,
                )

            # BladeRow returns dict with keys: w_out_i, s_out_i, beta_out_i, *crit*_i
            # VanelessChannel returns {}
            row_guess_dict.update(ig_row)


        # Global inlet velocity variable if solver uses it
        if "v_in" not in row_guess_dict:
            row_guess_dict["v_in"] = 0.1 * problem.reference_values["v0"]

        # ---- pack & scale for solver ----
        initial_guess_scaled = problem.scale_values(row_guess_dict)
        x0 = jnp.array(list(initial_guess_scaled.values()), dtype=float)
        problem.keys = list(initial_guess_scaled.keys())

        # z0 = jnp.array(list(initial_guess_scaled.values()), dtype=jnp.float64)
        # z0 = jnp.clip(z0, z_lo, z_hi)

        # x0 = jnp.arctanh(z0).astype(float)

        if not jnp.all(jnp.isfinite(x0)):
            bad = {k: v for k, v in zip(problem.keys, x0) if not jnp.isfinite(v)}
            raise ValueError(f"Initial guess contains non-finite values: {bad}")

    # ------------------------------------------------------------------
    # Solve with LM
    # ------------------------------------------------------------------
    solver_options["method"] = "lm"
    solver = psv.NonlinearSystemSolver(problem, logger=logger, **solver_options)
    solver.solve(x0)

    if not solver.success and logger:
        logger.info("WARNING: All attempts failed to converge")

    return solver, problem.results

# =================== problem (mapping-based) ===================


class TurbomachineryProblem(psv.NonlinearSystemProblem):
    """
    Component-wise turbine analysis.

    Responsibilities:
      - Build per-component geometry once (via _build_component_geometry).
      - Instantiate BladeRow / VanelessChannel / Interspace objects once from the YAML components + geometry.
      - Inject the fluid into each component using eqx.tree_at in update_boundary_conditions.
      - Provide residual(x) that delegates to flow.evaluate_turbomachine.
      - Provide scale_values(...) and gradient(...) for the solver.
    """

    def __init__(self, components, simulation_options, fluid):
        self.components = components  # raw YAML component dicts
        self.model_options = simulation_options
        self.keys = []

        # ---- Fluid ----
        self.fluid = fluid

        # 1) Instantiate component objects (BladeRow / VanelessChannel / Interspace / others)
        comp_objs: list[Any] = []
        cascade_geoms: list[Dict[str, Any]] = []
        cascade_indices: list[int] = []

        for idx, comp in enumerate(self.components):
            ctype = str(comp.get("component_type", "")).lower()

            # ------------------------------
            # Axial or radial cascade → BladeRow
            # ------------------------------
            if ctype in ("axial_cascade", "radial_cascade"):
                partial_geom = comp["geometry"]
                cfg_row = {
                    "name": comp.get("name", f"row_{idx+1}"),
                    "cascade_type": partial_geom["cascade_type"],
                    "component_type": ctype,
                    "geometry": partial_geom,             # RAW geometry goes in
                    "model_options": comp.get("model_options", {}),
                    "initial_guess": comp.get("initial_guess", {}),
                }
                # fluid=self.fluid so it's immediately available; boundary BCs
                # are still injected later in update_boundary_conditions.
                row = BladeRow.from_dict(
                    cfg_row,
                    fluid=self.fluid,
                    model_options_global=self.model_options,
                )

                comp_objs.append(row)
                cascade_geoms.append(row.geometry)   # full geometry from the row
                cascade_indices.append(idx)
                continue

            # ------------------------------
            # Vaneless channel → VanelessChannel
            # ------------------------------
            if ctype == "vaneless_channel":
                partial_geom = comp["geometry"]
                solver_opts = comp.get("solver_options", None)
                cfg_ch = {
                    "name": comp.get("name", f"channel_{idx+1}"),
                    "geometry": partial_geom,            # RAW geometry passed in
                    "model_options": comp.get("model_options", {}),
                    "solver_options": solver_opts,
                    "operating_conditions": {
                        "p_in": 1.0e5,    # arbitrary but valid
                        "h_in": 1.0e5,    # arbitrary but valid
                        "v_in": 1.0,      # non-zero so nothing divides by zero
                        "alpha_in": 0.0,  # degrees
                    },
                }
                ch = VanelessChannel.from_dict(
                    cfg_ch,
                    fluid=self.fluid,
                )
                comp_objs.append(ch)
                continue

            # ------------------------------
            # Interspace → Interspace
            # ------------------------------
            if ctype == "interspace":
                partial_geom = comp["geometry"]
                inter_cfg = {
                    "name": comp.get("name", f"interspace_{idx+1}"),
                    "geometry": partial_geom,
                    }
                
                inter = Interspace.from_dict(inter_cfg, fluid=self.fluid)
                comp_objs.append(inter)
                # no cascade geometry added here; interspace is not a cascade
                continue

            # ------------------------------
            # Fallback: keep as-is (must be handled in flow_model)
            # ------------------------------
            comp_objs.append(comp)

        self.comp_objects = comp_objs

        # Cascades-only geometry + indices (used for per-row seeding / stage KPIs, etc.)
        self.geometry_components_cascades = cascade_geoms
        self.cascade_comp_indices = cascade_indices
        self.num_cascades = len(cascade_indices)

        # These will be set in update_boundary_conditions
        self.boundary_conditions: Dict[str, Any] = {}
        self.reference_values: Dict[str, Any] = {}

        # --- placeholders for solution state ---
        self.vars_scaled = {}   # normalized vars as dict
        self.vars_real = None   # unscaled solver vector (1D array)
        self.results = None     # last flow-model result dict


    # ------------------------------------------------------------------
    # Boundary conditions + fluid injection
    # ------------------------------------------------------------------
    def update_boundary_conditions(self, operation_point):
        """
        Set boundary conditions, reference values, and inject fluid into
        all component objects (BladeRow, VanelessChannel, ...).
        """

        assert_numeric_operation_point(operation_point)
        # Copy to avoid mutating the caller's operation_point dict in-place.
        # This matters for retry logic, where the same input dict can be reused.
        self.boundary_conditions = dict(operation_point)

        # ---- Inject fluid into all components that have a `.fluid` field ----
        new_list = []
        for comp in self.comp_objects:
            if hasattr(comp, "fluid"):
                comp = eqx.tree_at(lambda c: c.fluid, comp, self.fluid)
            new_list.append(comp)
        self.comp_objects = new_list

        # ---- Compute inlet stagnation state ----
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        p_out = operation_point["p_out"]

        st_in = self.fluid.get_state(jxp.PT_INPUTS, p0_in, T0_in)
        h0_in = st_in["h"]
        s_in = st_in["s"]

        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # ---- Isentropic outlet ----
        st_out_s = self.fluid.get_state(jxp.PSmass_INPUTS, p_out, s_in)
        h_out_s = st_out_s["h"]
        d_out_s = st_out_s["d"]

        # ---- Reference velocity ----
        v0 = jnp.sqrt(2 * (h0_in - h_out_s)) # spouting velocity

        # ---- Reference mass flow (use last component with A_out) ----
        A_out = None
        for obj in reversed(self.comp_objects):
            geom = getattr(obj, "geometry", None)
            if isinstance(geom, dict) and "A_out" in geom:
                A_out = geom["A_out"]
                break

        if A_out is None:
            raise ValueError(
                "Cannot determine reference A_out (no component with 'A_out' found)."
            )

        mass_flow_ref = A_out * d_out_s * v0

        # ---- Store reference values ----
        self.reference_values = {
            "v0": v0,
            "h_out_s": h_out_s,
            "d_out_s": d_out_s,
            "mass_flow_ref": mass_flow_ref,
            "s_min": s_in,
            "s_range": self.fluid.get_state(jxp.HmassP_INPUTS, h0_in, p_out)["s"] - s_in,
            "angle_min": -90.0,
            "angle_range": 180.0,
        }

        # Optional constraint configuration (kept out of the default flow unless provided)
        # Example in YAML:
        #   simulation_options:
        #     subsonic_constraint:
        #       enabled: true
        #       Ma_rel_out_max: 0.99
        #       scale: 0.05
        subsonic = self.model_options.get("subsonic_constraint", {}) or {}
        if isinstance(subsonic, dict) and subsonic.get("enabled", False):
            if "Ma_rel_out_max" in subsonic:
                self.reference_values["Ma_rel_out_max"] = subsonic["Ma_rel_out_max"]
            if "scale" in subsonic:
                self.reference_values["Ma_rel_out_scale"] = subsonic["scale"]
        else:
            # Allow flat keys for quick experimentation
            if "Ma_rel_out_max" in self.model_options:
                self.reference_values["Ma_rel_out_max"] = self.model_options["Ma_rel_out_max"]
            if "Ma_rel_out_scale" in self.model_options:
                self.reference_values["Ma_rel_out_scale"] = self.model_options["Ma_rel_out_scale"]


        # ---- Inlet angle in degrees ----
        alpha = operation_point["alpha_in"]
        # alpha_deg = jnp.degrees(alpha) if abs(alpha) < jnp.pi * 1.1 else alpha
        alpha_deg = alpha
        self.boundary_conditions["alpha_deg"] = float(alpha_deg)


    # ------------------------------------------------------------------
    # Scaling utilities
    # ------------------------------------------------------------------
    
    # def scale_values(self, variables, to_normalized=True):
    #     """
    #     Legacy scaling using v0 / s_range / angle_range with an entropy floor.
    #     """
    #     v0 = self.reference_values["v0"]
    #     s_range = self.reference_values["s_range"]
    #     s_min = self.reference_values["s_min"]
    #     angle_range = self.reference_values["angle_range"]
    #     angle_min = self.reference_values["angle_min"]
    
    #     s_floor = jnp.maximum(
    #         jnp.array(1e-6, dtype=jnp.float64),
    #         0.01 * jnp.maximum(jnp.abs(s_min), 1.0),
    #     )
    #     s_sigma = jnp.maximum(s_range, s_floor)
    
    #     scaled_variables = {}
    #     for key, val in variables.items():
    #         if key.startswith(("v", "w")):
    #             scaled_variables[key] = val / v0 if to_normalized else val * v0
    #         elif key.startswith("s"):
    #             scaled_variables[key] = (
    #                 (val - s_min) / s_sigma if to_normalized else val * s_sigma + s_min
    #             )
    #         elif key.startswith("b"):
    #             scaled_variables[key] = (
    #                 (val - angle_min) / angle_range
    #                 if to_normalized
    #                 else val * angle_range + angle_min
    #             )
    #     return scaled_variables

    def scale_values(self, variables, to_normalized: bool = True):
        """
        Mu/sigma scaling:
          - v*/w*: mu=0,          sigma=v0
          - s*    : mu=s_min+0.5*s_range, sigma=max(0.5*s_range, s_floor)
          - beta* : mu=angle_min+0.5*angle_range, sigma=0.5*angle_range
        """
        v0 = self.reference_values["v0"]
        s_range = self.reference_values["s_range"]
        s_min = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]

        s_floor = jnp.maximum(
            jnp.array(1e-6, dtype=jnp.float64),
            0.01 * jnp.maximum(jnp.abs(s_min), 1.0),
        )
        mu_s = s_min + 0.5 * s_range
        # sigma_s = jnp.maximum(0.5 * s_range, s_floor)
        sigma_s = 0.5 * s_range

        mu_b = angle_min + 0.5 * angle_range
        sigma_b = 0.5 * angle_range

        scaled_variables = {}
        for key, val in variables.items():
            if key.startswith(("v", "w")):
                mu = jnp.array(0.0, dtype=jnp.float64)
                sigma = v0
            elif key.startswith("s"):
                mu = mu_s
                sigma = sigma_s
            elif key.startswith("b"):
                mu = mu_b
                sigma = sigma_b
            else:
                # Unknown key pattern; leave unchanged
                scaled_variables[key] = val
                continue

            if to_normalized:
                scaled_variables[key] = (val - mu) / sigma
            else:
                scaled_variables[key] = val * sigma + mu

        return scaled_variables

    # ------------------------------------------------------------------
    # Residual & gradient
    # ------------------------------------------------------------------
    def residual(self, x):
        """
        Map the solver vector x → dict of scaled vars → call
        flow.evaluate_axial_turbine_componentwise.
        """

        #####
        # print("Residual function:")
        # print(x)
        #####
        try:
            # 1) unpack x into a dict with the keys determined in compute_single_operation_point
            # ////////////////////////////////////

            # t0 = time.perf_counter()

            # time this part

            # vars_unbounded = dict(zip(self.keys, x))
            # self.vars_scaled = {k: jnp.tanh(v) for k, v in vars_unbounded.items()}  # enforce boundedness

            self.vars_scaled = dict(zip(self.keys, x))
            
            # print(self.vars_scaled)
            # print("x", x, "x_norm", jnp.linalg.norm(x))

            vars_real_dict = self.scale_values(
                self.vars_scaled,
                to_normalized=False,
            )
            self.vars_real = jnp.array(
                [vars_real_dict[k] for k in self.keys],
                dtype=jnp.float64,
            )

            # t1 = time.perf_counter()
            # print(
            #     f"[residual] unpack+scale took {t1 - t0:.6e} s "
            #     f"(len(x) = {len(x)})"
            # )


            # //////////////////////////////////////////////
            # time this part

            # t2 = time.perf_counter()

            # 2) Call the flow model with *scaled* variables; it will unscale internally
            self.results = flow.evaluate_turbomachine(
                variables=self.vars_scaled,
                boundary_conditions=self.boundary_conditions,
                comp_objects=self.comp_objects,
                fluid=self.fluid,
                reference_values=self.reference_values,
            )

            res_vec = jnp.array(list(self.results["residuals"].values()))


            # jax.debug.print("vars_scaled = {}", self.vars_scaled)
            # jax.debug.print("residuals = {}", self.results["residuals"])

            # print(self.results["residuals"])

            # jax.block_until_ready(res_vec)

            # t3 = time.perf_counter()
            # print(
            #     f"[residual] flow.evaluate_turbomachine took {t3 - t2:.6e} s "
            #     f"(n_residuals = {res_vec.size})"
            # )

            # =======================
            # 3) return residual vector
            # =======================

            return res_vec
        
        except Exception as e:
            bc_types = {
                k: type(v).__name__
                for k, v in getattr(self, "boundary_conditions", {}).items()
            }
            raise TypeError(
                f"Residual failed: {e}\n"
                f"  OP types: {bc_types}\n"
                f"  Keys in vars_scaled: {list(getattr(self, 'vars_scaled', {}).keys())}"
            ) from e

    def gradient(self, x):
        """
        Jacobian of the residual w.r.t. x, used by the nonlinear solver.
        """

        #####
        # print("Gradient function:")
        # print(x)
        #####

        return jax.jacfwd(self.residual, argnums=0)(x)


# ================= IG & distance utilities (unchanged) =================


def find_closest_operation_point(current_op_point, operation_points, solution_data):
    min_distance, closest_point_x, closest_index = float("inf"), None, None
    for i, op_point in enumerate(operation_points):
        d = get_operation_point_distance(current_op_point, op_point)
        if d < min_distance:
            min_distance, closest_point_x, closest_index = d, solution_data[i], i
    return closest_point_x, closest_index


def get_operation_point_distance(point_1, point_2, delta=1e-8):
    deviation_array = []
    for key in point_1:
        if isinstance(point_1[key], (int, float)) and key in point_2:
            v1, v2 = point_1[key], point_2[key]
            if key == "alpha_in":
                deviation = jnp.abs(v1 - v2) / 90
            else:
                max_val = max(abs(v1), abs(v2), delta)
                deviation = abs(v1 - v2) / max_val
            deviation_array.append(deviation)
    return jnp.linalg.norm(deviation_array)
