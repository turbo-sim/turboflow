import os
import yaml
import copy
import datetime
import itertools
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import dill

from scipy.stats import qmc
from scipy import optimize

from .. import math
from .. import pysolver_view as psv
from .. import utilities as utils
from . import geometry_model_axial as geom
# from . import geometry_model_radial as geom
from . import flow_model as flow
from . import deviation_model as dm
import jaxprop as jxp
import jaxprop.perfect_gas as pg

import turboflow as tf
# NEW: bring in the triangle helpers from the BladeRow module
from .blade_row import (
    evaluate_velocity_triangle_out,
    evaluate_velocity_triangle_in,
)


jax.config.update("jax_enable_x64", True)  # 64-bit for scientific computing

SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}

# -------------------------------------------------------------------
# helpers to fail fast on non-numeric contamination
# -------------------------------------------------------------------
NUMERIC = (int, float, np.floating)

def _is_num(x):
    return isinstance(x, NUMERIC)

def assert_numeric_operation_point(op):
    for k, v in op.items():
        if k == "fluid_name":
            if not isinstance(v, str):
                raise TypeError(f"operation_point['fluid_name'] must be str, got {type(v)}")
            continue
        if not _is_num(v):
            raise TypeError(f"operation_point['{k}'] must be numeric, got {v!r} ({type(v)})")

def _prune_vars_to_match_choking(initial_guess: dict, components: list, global_choking: str) -> dict:
    """
    Keep only the unknowns required by each component's choking model.
    Always keep: v_in and per-component w_out_i, s_out_i, beta_out_i.
    For choking model per component i:
      - critical_mach_number: keep w_crit_throat_i, s_crit_throat_i; drop v_crit_in_i
      - critical_isentropic_throat: keep w_crit_throat_i; drop v_crit_in_i, s_crit_throat_i
      - critical_mass_flow_rate: keep v_crit_in_i, w_crit_throat_i, s_crit_throat_i
    """
    out = {}
    out["v_in"] = initial_guess.get("v_in", None)
    n = sum(1 for c in components if str(c.get("component_type","")).lower() == "axial_cascade")

    for i in range(n):
        tag = f"_{i+1}"
        # always needed per component
        for base in ("w_out", "s_out", "beta_out"):
            k = base + tag
            if k in initial_guess:
                out[k] = initial_guess[k]

        # resolve the choking model for this component
        comp_opts = (components[i].get("model_options") or {})
        crit = comp_opts.get("choking_criterion", global_choking)

        if crit == "critical_mach_number":
            # keep: w_crit_throat_i, s_crit_throat_i
            for k in (f"w_crit_throat{tag}", f"s_crit_throat{tag}"):
                if k in initial_guess:
                    out[k] = initial_guess[k]
            # drop v_crit_in_i (do nothing)
        elif crit == "critical_isentropic_throat":
            # keep only w_crit_throat_i
            k = f"w_crit_throat{tag}"
            if k in initial_guess:
                out[k] = initial_guess[k]
        elif crit == "critical_mass_flow_rate":
            # keep all three
            for k in (f"v_crit_in{tag}", f"w_crit_throat{tag}", f"s_crit_throat{tag}"):
                if k in initial_guess:
                    out[k] = initial_guess[k]
        else:
            # default conservative: keep none of the extra critical vars
            pass

    # remove None if v_in was missing
    out = {k: v for k, v in out.items() if v is not None}
    return out

def _expand_ratios(performance_map: dict) -> dict:
    """
    In map mode, allow specifying p_out via ratios to p0_in:
      - p_out_ratio_range: [r_min, r_max]
      - p_out_ratio_points: N
      - OR p_out_ratio_values: [r1, r2, ...]
    Returns a *new* map dict with 'p_out_ratio' as a list of ratios (if provided).
    Does nothing if neither field is present.
    """
    pm = dict(performance_map)  # shallow copy

    if "p_out_ratio_values" in pm:
        ratios = list(pm["p_out_ratio_values"])
        pm["p_out_ratio"] = ratios

    elif "p_out_ratio_range" in pm and "p_out_ratio_points" in pm:
        lo, hi = pm["p_out_ratio_range"]
        n = int(pm["p_out_ratio_points"])
        ratios = np.linspace(lo, hi, n).tolist()
        pm["p_out_ratio"] = ratios

    # Clean helper keys (optional)
    for k in ("p_out_ratio_values", "p_out_ratio_range", "p_out_ratio_points"):
        if k in pm:
            del pm[k]

    return pm

# --- put near the other helpers in performance_analysis.py ---

def _eval_item_if_str(x, ctx):
    """Evaluate x if it's a string expression; recurse into lists/tuples.
    If evaluation fails or it's plain text (like 'air'), return original.
    """
    import numpy as np

    def _looks_like_expr(s: str) -> bool:
        # Heuristic: treat as expression if it references numpy or has math operators/paren/brackets
        expr_tokens = ("np.", "(", ")", "[", "]", "*", "/", "+", "-", "**")
        return any(t in s for t in expr_tokens)

    if isinstance(x, str):
        if not _looks_like_expr(x):
            # Not an expression (likely plain text like "air") → leave as-is
            return x
        try:
            return eval(x, {"__builtins__": {}, "np": np}, ctx)
        except (NameError, SyntaxError, AttributeError, TypeError, ZeroDivisionError):
            # If anything goes wrong, fall back to original string
            return x
    elif isinstance(x, list):
        return [_eval_item_if_str(e, ctx) for e in x]
    elif isinstance(x, tuple):
        return tuple(_eval_item_if_str(e, ctx) for e in x)
    else:
        return x


def _evaluate_map_expressions(performance_map):
    """
    Evaluate simple NumPy/math expressions embedded as strings in the map.
    Supports references to already-present numeric keys (e.g., 'p0_in') and 'np'.
    Leaves plain text (e.g., 'air') untouched.
    """
    import numpy as np

    pm = dict(performance_map)  # shallow copy

    # Build context with simple numeric values (scalars) so expressions can reference them, e.g., p0_in
    ctx = {k: v for k, v in pm.items() if isinstance(v, (int, float))}
    # Optionally include numeric lists so they can also be referenced
    ctx.update({k: v for k, v in pm.items()
                if isinstance(v, list) and all(isinstance(e, (int, float)) for e in v)})

    for k, v in list(pm.items()):
        evaluated = _eval_item_if_str(v, ctx)

        # Convert numpy arrays to native lists for downstream code
        if hasattr(evaluated, "tolist"):
            evaluated = evaluated.tolist()

        pm[k] = evaluated

        # Refresh context if we just created something numeric that others might reference
        if isinstance(evaluated, (int, float)):
            ctx[k] = evaluated
        elif isinstance(evaluated, list) and all(isinstance(e, (int, float)) for e in evaluated):
            ctx[k] = evaluated

    return pm


def _numpy_to_native(x):
    """Convert numpy arrays to nested Python lists; leave scalars alone."""
    import numpy as np
    if hasattr(x, "tolist"):
        return x.tolist()
    elif isinstance(x, (list, tuple)):
        return type(x)(_numpy_to_native(e) for e in x)
    else:
        return x

# --- add this small helper near the other helpers (top of file is fine) ---

def _infer_num_cascades(geometry_arrayview, components=None):
    """
    Return the number of *axial cascades*.
    When components is provided, count only items with component_type == 'axial_cascade'.
    Otherwise use geometry_arrayview['number_of_cascades'] or geometry array lengths.
    """

    if components is not None and isinstance(components, (list, tuple)):
        def _is_cascade(c):
            return str(c.get("component_type", "")).lower() == "axial_cascade"
        return sum(1 for c in components if _is_cascade(c))

    if isinstance(geometry_arrayview, dict):
        val = geometry_arrayview.get("number_of_cascades", None)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (list, tuple, np.ndarray)):
            return len(val)
        for key in ("A_in", "A_out", "chord", "pitch"):
            v = geometry_arrayview.get(key, None)
            if isinstance(v, (list, tuple, np.ndarray)):
                return len(v)

    raise ValueError("Could not infer number_of_cascades from geometry/components.")


# ===================================================================
# Public entry: performance over a set/map of operation points
# ===================================================================
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

    # Expect components list in config
    if not config.get("components"):
        raise ValueError("No 'components' found in config. Provide a list of components.")

    # Ranges → list of operation points
    if isinstance(operation_points, dict):
        operation_points = generate_operation_points(operation_points)
    elif not isinstance(operation_points, (list, np.ndarray)):
        raise TypeError("operation_points must be either list of dicts or a dict with ranges.")

    # Validate
    for op in operation_points:
        validate_operation_point(op)
        assert_numeric_operation_point(op)

    # Collectors
    operation_point_data, overall_data = [], []
    plane_data, cascade_data, stage_data = [], [], []
    solver_data, solution_data, geometry_data = [], [], []
    solver_container = []

    # Pretty print OPs
    message = print_operation_points(operation_points)
    for line in message.splitlines():
        logger.info(line)

    for i, operation_point in enumerate(operation_points):
        logger.info("")
        logger.info(f" Computing operation point {i+1} of {len(operation_points)}")
        for line in print_boundary_conditions(operation_point).splitlines():
            logger.info(line)

        # Initial guess selection
        if i == 0:
            initial_guess_cfg = extract_initial_guess_from_components(config["components"])
        else:
            closest_x, closest_index = find_closest_operation_point(
                operation_point,
                operation_points[:i],
                solution_data[:i],
            )
            logger.info(f" Using solution from point {closest_index+1} as initial guess")
            initial_guess_cfg = closest_x

        # Solve one OP
        solver, results = compute_single_operation_point(
            operation_point,
            initial_guess_cfg,
            config["components"],                       # << components list
            config.get("simulation_options", {}),       # global fallbacks
            config["performance_analysis"]["solver_options"],
            logger=logger
        )

        # Solver summary
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

        # Collect
        operation_point_data.append(pd.DataFrame([operation_point]))
        overall_data.append(pd.DataFrame.from_dict(results["overall"], orient="index").T)
        plane_data.append(utils.flatten_dataframe(pd.DataFrame(results["planes"])))
        cascade_data.append(utils.flatten_dataframe(pd.DataFrame(results["cascades"])))
        stage_data.append(utils.flatten_dataframe(pd.DataFrame(results["stage"])))
        # geometry: flatten per-component results
        geom_rows_df = pd.DataFrame(results["geometry_components"])
        geometry_data.append(utils.flatten_dataframe(geom_rows_df))
        solver_data.append(pd.DataFrame([solver_status]))
        solution_data.append(solver.problem.vars_real)
        solver_container.append(solver)

    # Export dataframes
    dfs = {
        "operation point": pd.concat(operation_point_data, ignore_index=True),
        "overall":         pd.concat(overall_data, ignore_index=True),
        "plane":           pd.concat(plane_data, ignore_index=True),
        "cascade":         pd.concat(cascade_data, ignore_index=True),
        "stage":           pd.concat(stage_data, ignore_index=True),
        "geometry":        pd.concat(geometry_data, ignore_index=True),
        "solver":          pd.concat(solver_data, ignore_index=True),
    }

    if export_results:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if out_filename is None:
            out_filename = "performance"

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filenames = [f"{out_filename}_{current_time}", f"{out_filename}_latest"]

        for fname in out_filenames:
            # dump config (as provided)
            config_data = {k: v for k, v in config.items() if v}
            config_data = utils.convert_numpy_to_python(config_data, precision=12)
            with open(os.path.join(out_dir, f"{fname}.yaml"), "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            # excel bundle
            filepath_xlsx = os.path.join(out_dir, f"{fname}.xlsx")
            with pd.ExcelWriter(filepath_xlsx, engine="openpyxl") as writer:
                for sheet_name, df in dfs.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)

            # pickle solver (lightweight)
            filepath_pkl = os.path.join(out_dir, f"{fname}.pkl")
            solver.problem = None
            with open(filepath_pkl, 'wb') as f:
                dill.dump(solver, f)

        logger.info(f" Performance data successfully written to {filepath_xlsx}")

    # Footer summary
    message = print_simulation_summary(solver_container)
    for line in message:
        logger.info(line)

    return solver_container


# ===================================================================
# One operation point
# ===================================================================

def compute_single_operation_point(
    operating_point,
    initial_guess,
    components,            # << list of component dicts (from YAML)
    simulation_options,
    solver_options,
    logger=None
):
    """
    Compute one operation point for given boundary conditions.
    """

    problem = AxialTurbineProblem(components, simulation_options)
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(solver_options)

    # Build initial guesses (component-wise first; fall back to legacy)
    initial_guesses = get_initial_guess(
        initial_guess,
        problem,
        problem.boundary_conditions,
        problem.geometry_components_arrayview,  # dict-of-arrays view for legacy heuristic
        problem.fluid,
        simulation_options.get("choking_criterion", "critical_mach_number"),
        simulation_options.get("deviation_model", "aungier"),
        logger,
        components=problem.components,
    )

    if not initial_guesses or not isinstance(initial_guesses[0], dict):
        raise ValueError(
            "Initial guess construction returned no valid guesses. "
            "Provide either the heuristic keys: "
            "  {'efficiency_tt', 'efficiency_ke', 'ma_1..ma_N'} "
            "or the direct keys (which must include 'v_in'): "
            "  {'v_in', 'w_out_i', 's_out_i', 'beta_out_i', ...} per cascade."
        )

    # ---- PRUNE unknowns to match each cascade's choking criterion ----
    cascade_indices = problem.cascade_comp_indices
    global_choking = simulation_options.get("choking_criterion", "critical_mach_number")

    per_cascade_choking = []
    for idx in cascade_indices:
        comp_opts = (problem.components[idx].get("model_options") or {})
        per_cascade_choking.append(comp_opts.get("choking_criterion", global_choking))

    pruned_initial_guesses = []
    n_casc = problem.num_cascades

    for ig in initial_guesses:
        pruned = {}
        if "v_in" in ig:
            pruned["v_in"] = ig["v_in"]

        # tags _1, _2, ... are per-cascade order
        for i in range(n_casc):
            tag = f"_{i+1}"

            # always keep these per cascade
            for base in ("w_out", "s_out", "beta_out"):
                k = base + tag
                if k in ig:
                    pruned[k] = ig[k]

            crit = per_cascade_choking[i]
            if crit == "critical_mach_number":
                for k in (f"w_crit_throat{tag}", f"s_crit_throat{tag}"):
                    if k in ig:
                        pruned[k] = ig[k]
            elif crit == "critical_isentropic_throat":
                k = f"w_crit_throat{tag}"
                if k in ig:
                    pruned[k] = ig[k]
            elif crit == "critical_mass_flow_rate":
                for k in (f"v_crit_in{tag}", f"w_crit_throat{tag}", f"s_crit_throat{tag}"):
                    if k in ig:
                        pruned[k] = ig[k]
            else:
                # unknown criterion -> only the base three are kept
                pass

        pruned_initial_guesses.append(pruned)

    initial_guesses = pruned_initial_guesses
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    solver_methods = [solver_options["method"]] + [
        m for m in SOLVER_MAP.keys() if m != solver_options["method"]
    ]

    for ig in initial_guesses:
        initial_guess_scaled = problem.scale_values(ig)
        x0 = np.array(list(initial_guess_scaled.values()))
        problem.keys = initial_guess_scaled.keys()

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


# ===================================================================
# Problem definition (component-wise)
# ===================================================================
class AxialTurbineProblem(psv.NonlinearSystemProblem):
    """
    Nonlinear system problem for a component-wise axial turbine analysis.
    """

    def __init__(self, components, simulation_options):
        self.components = components
        self.model_options = simulation_options
        self.keys = []

        # --- NEW: filter components by type ---
        def _is_cascade(c):
            return str(c.get("component_type", "")).lower() == "axial_cascade"

        self.cascade_comp_indices = [i for i, c in enumerate(self.components) if _is_cascade(c)]
        self.num_cascades = len(self.cascade_comp_indices)

        # Build per-cascade geometry (list[dict]) using your existing geometry model
        components_cascades = [self.components[i] for i in self.cascade_comp_indices]
        self.geometry_components_cascades = geom.calculate_full_geometry(components_cascades)

        # Dict-of-arrays arrayview for legacy helpers (cascades only)
        self.geometry_components_arrayview = self._to_array_geometry(self.geometry_components_cascades)

    def _to_array_geometry(self, rows):
        all_keys = set().union(*[row.keys() for row in rows]) if rows else set()
        array_geom = {"number_of_cascades": len(rows), "number_of_stages": max(0, len(rows)//2)}
        for k in all_keys:
            if k in ("cascade_type",):
                array_geom[k] = [row.get(k) for row in rows]
            else:
                array_geom[k] = [row.get(k) for row in rows]
        return array_geom

    def residual(self, x):
        """
        Evaluate residuals for given decision variables.
        """
        try:
            self.vars_scaled = dict(zip(self.keys, x))
            self.vars_real = self.scale_values(self.vars_scaled, to_normalized=False)

            self.results = flow.evaluate_axial_turbine_componentwise(
                self.vars_scaled,
                self.boundary_conditions,
                self.geometry_components_cascades, 
                self.fluid,
                self.reference_values,
                self.components,
                self.model_options,
            )


            return jnp.array(list(self.results["residuals"].values()))
        except Exception as e:
            bc_types = {k: type(v).__name__ for k, v in getattr(self, "boundary_conditions", {}).items()}
            raise TypeError(
                f"Residual failed: {e}\n"
                f"  OP types: {bc_types}\n"
                f"  Keys in vars_scaled: {list(self.vars_scaled.keys())}"
            ) from e

    def gradient(self, x):
        return jax.jacfwd(self.residual, argnums=0)(x)

    def update_boundary_conditions(self, operation_point):
        """
        Set boundary conditions and reference values.
        """
        assert_numeric_operation_point(operation_point)
        self.boundary_conditions = operation_point

        # Fluid
        self.fluid = jxp.FluidJAX(operation_point["fluid_name"])

        # Short-hands
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        p_out = operation_point["p_out"]

        # Inlet stagnation
        state_in_stag = self.fluid.get_state(jxp.PT_INPUTS, p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Exit static (isentropic)
        state_out_s = self.fluid.get_state(jxp.PSmass_INPUTS, p_out, s_in)
        h_isentropic = state_out_s["h"]
        d_isentropic = state_out_s["d"]

        # Exit static (isenthalpic) for s_range
        state_out_h = self.fluid.get_state(jxp.HmassP_INPUTS, h0_in, p_out)
        s_isenthalpic = state_out_h["s"]

        # Spouting velocity
        v0 = np.sqrt(2 * (h0_in - h_isentropic))

        # Reference mass flow uses last component exit area
        A_out_last = self.geometry_components_cascades[-1]["A_out"]
        mass_flow_ref = A_out_last * v0 * d_isentropic

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

    def scale_values(self, variables, to_normalized=True):
        """
        Convert values between normalized and real values.
        """
        v0 = self.reference_values["v0"]
        s_range = self.reference_values["s_range"]
        s_min = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]

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


# ===================================================================
# Initial guess handling
# ===================================================================
def extract_initial_guess_from_components(components):
    ig = {}
    eff_tt = None
    eff_ke = None
    ma_list = []

    def _is_cascade(c):
        return str(c.get("component_type", "")).lower() == "axial_cascade"

    for comp in components:
        if not _is_cascade(comp):
            continue
        ig_c = comp.get("initial_guess", {}) or {}
        if eff_tt is None and "efficiency_tt" in ig_c:
            eff_tt = ig_c["efficiency_tt"]
        if eff_ke is None and "efficiency_ke" in ig_c:
            eff_ke = ig_c["efficiency_ke"]

        ma = (
            ig_c.get("ma") or ig_c.get("ma_out") or ig_c.get("ma_rel_out")
            or ig_c.get("ma_exit") or ig_c.get("ma_2") or ig_c.get("ma_1")
        )
        ma_list.append(ma if isinstance(ma, (int, float)) else None)

    marker = {}
    if eff_tt is not None:
        marker["efficiency_tt"] = eff_tt
    if eff_ke is not None:
        marker["efficiency_ke"] = eff_ke
    if ma_list:
        for i, m in enumerate(ma_list):
            marker[f"ma_{i+1}"] = 0.8 if m is None else m

    return marker if marker else {"_empty_": True}
def get_initial_guess(
    initial_guess,
    problem,
    boundary_conditions,
    geometry,          # dict-of-arrays "arrayview"
    fluid,
    choking_criterion,
    deviation_model,
    logger,
    components=None,   # <-- keep this arg, we use it to infer cascade count
):
    import numpy as np

    # Robust cascade count (works for axial + radial)
    number_of_cascades = _infer_num_cascades(geometry, components=components)

    # If we only got the "_empty_" marker, synthesize a heuristic seed.
    if isinstance(initial_guess, dict) and initial_guess.get("_empty_", False):
        # Reasonable, solver-friendly defaults
        ig_default = {
            "efficiency_tt": 0.90,
            "efficiency_ke": 0.20,
        }
        # provide per-cascade Mach hints; 0.8 is a typical starting value
        for i in range(number_of_cascades):
            ig_default[f"ma_{i+1}"] = 0.80
        initial_guess = ig_default

    # --- supported key sets ---
    valid_keys_1 = ["efficiency_tt", "efficiency_ke"] + [
        f"ma_{i+1}" for i in range(number_of_cascades)
    ]
    valid_keys_2 = ["efficiency_tt", "efficiency_ke", "ma", "n_samples"]

    # direct (must include v_in)
    base3 = ["w_out", "s_out", "beta_out", "w_crit_throat", "s_crit_throat"]
    valid_keys_3 = ["v_in"] + [f"{k}_{i+1}" for i in range(number_of_cascades) for k in base3]

    base4 = ["w_out", "s_out", "beta_out", "v_crit_in", "w_crit_throat", "s_crit_throat"]
    valid_keys_4 = ["v_in"] + [f"{k}_{i+1}" for i in range(number_of_cascades) for k in base4]

    base5 = ["w_out", "s_out", "beta_out", "w_crit_throat"]
    valid_keys_5 = ["v_in"] + [f"{k}_{i+1}" for i in range(number_of_cascades) for k in base5]

    # Normalize input dict’s keys set
    in_keys = set(list(initial_guess.keys()))

    # ------------------------------
    # Decide path & build guesses
    # ------------------------------
    initial_guesses = None

    # Heuristic with explicit per-cascade ma_i
    if set(valid_keys_1) == in_keys:
        if isinstance(initial_guess["efficiency_tt"], (list, np.ndarray)):
            initial_guesses = []
            for i in range(len(initial_guess["efficiency_tt"])):
                ma = np.array([
                    initial_guess[f"ma_{j+1}"][i]
                    if isinstance(initial_guess.get(f"ma_{j+1}", 0.80), (list, np.ndarray))
                    else initial_guess.get(f"ma_{j+1}", 0.80)
                    for j in range(number_of_cascades)
                ])
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
            ma = np.array([initial_guess.get(f"ma_{j+1}", 0.80) for j in range(number_of_cascades)])
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

    # Heuristic via LHS box: {'eff_tt' range, 'eff_ke' range, 'ma' range, n_samples}
    elif set(valid_keys_2) == in_keys:
        bounds = [initial_guess["efficiency_tt"], initial_guess["efficiency_ke"]] + [
            initial_guess["ma"] for _ in range(number_of_cascades)
        ]
        n_samples = int(initial_guess["n_samples"])
        heuristic_inputs = latin_hypercube_sampling(bounds, n_samples)
        norm_residuals = np.array([])
        failures = 0
        for sample in heuristic_inputs:
            try:
                ma = [sample[i + 2] for i in range(number_of_cascades)]
                heuristic_guess = get_heuristic_guess(
                    sample[0],
                    sample[1],
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
            except Exception:
                failures += 1
                norm_residuals = np.append(norm_residuals, np.nan)

        if logger:
            logger.info("Generating heuristic initial guesses from latin hypercube sampling")
            logger.info(f"Number of failures: {failures} out of {n_samples} samples")
            if np.isfinite(norm_residuals).any():
                logger.info(f"Least norm of residuals: {np.nanmin(norm_residuals)}")

        best = heuristic_inputs[np.nanargmin(norm_residuals)]
        ma = [best[i + 2] for i in range(number_of_cascades)]
        initial_guess_best = get_heuristic_guess(
            best[0],
            best[1],
            ma,
            boundary_conditions,
            geometry,
            fluid,
            deviation_model,
        )
        initial_guesses = [initial_guess_best]

    # Direct guesses (must include v_in)
    elif set(valid_keys_3) == in_keys:
        initial_guesses = [initial_guess]
    elif set(valid_keys_4) == in_keys:
        initial_guesses = [initial_guess]
    elif set(valid_keys_5) == in_keys:
        initial_guesses = [initial_guess]

    # If none matched, fail fast with a clear message
    if initial_guesses is None:
        raise ValueError(
            "Initial guess must match one of the supported key sets.\n"
            "EITHER:\n"
            "  (Heuristic) {'efficiency_tt','efficiency_ke','ma_1..ma_N'}\n"
            "  (Heuristic LHS) {'efficiency_tt','efficiency_ke','ma','n_samples'}\n"
            "OR (Direct; MUST include 'v_in')\n"
            "  {'v_in', 'w_out_i','s_out_i','beta_out_i','w_crit_throat_i','s_crit_throat_i'}\n"
            "  {'v_in', 'w_out_i','s_out_i','beta_out_i','w_crit_throat_i'}\n"
            "  {'v_in', 'w_out_i','s_out_i','beta_out_i','v_crit_in_i','w_crit_throat_i','s_crit_throat_i'}\n"
            f"Got keys: {sorted(in_keys)}"
        )

    # Filter keys based on global choking criterion (keep v_in intact)
    pruned_list = []
    for i, ig in enumerate(initial_guesses):
        pruned = dict(ig)
        if choking_criterion == "critical_mach_number":
            pruned = {k: v for k, v in pruned.items() if not k.startswith("v_crit_in")}
        elif choking_criterion == "critical_mass_flow_rate":
            pruned = {k: v for k, v in pruned.items() if not k.startswith("beta_crit_throat")}
        elif choking_criterion == "critical_isentropic_throat":
            pruned = {k: v for k, v in pruned.items()
                      if not (k.startswith("v_crit_in") or k.startswith("s_crit_throat"))}
        pruned_list.append(pruned)

    initial_guesses = pruned_list

    # ---- HARD GUARD: every guess must include v_in
    missing_vin = [i for i, ig in enumerate(initial_guesses)
                   if (not isinstance(ig, dict)) or ("v_in" not in ig)]
    if missing_vin:
        raise ValueError(
            "Initial guess missing 'v_in' for the following guess indices: "
            f"{missing_vin}. "
            "When supplying direct per-cascade variables, include 'v_in'. "
            "If you prefer not to, use the heuristic form "
            "['efficiency_tt','efficiency_ke','ma_1..ma_N'] which computes 'v_in' automatically."
        )

    return initial_guesses

# ===================================================================
# Misc. utilities (OPs, printing, heuristic etc.)
# ===================================================================
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
                deviation = np.abs(v1 - v2) / 90
            else:
                max_val = max(abs(v1), abs(v2), delta)
                deviation = abs(v1 - v2) / max_val
            deviation_array.append(deviation)
    return np.linalg.norm(deviation_array)

def generate_operation_points(performance_map):
    """
    Generates list of operation points from a map (Cartesian product).
    Supports:
      - direct p_out arrays/lists
      - expressions in strings (np.linspace, arithmetic referencing p0_in, etc.)
      - your existing absolute fields
    """
    # 1) Evaluate any string expressions
    performance_map = _evaluate_map_expressions(performance_map)

    # 2) Ensure iterables
    performance_map = {k: utils.ensure_iterable(v) for k, v in performance_map.items()}

    # 3) Build Cartesian product (your original logic)
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
            raise ValueError("Each operation point must define 'p_out' (directly or via expression).")
        operation_points.append(op)

    return operation_points


def validate_operation_point(op_point):
    REQUIRED_FIELDS = {"fluid_name", "p0_in", "T0_in", "p_out", "alpha_in", "omega"}
    fields = set(op_point.keys())
    if fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - fields
        extra = fields - REQUIRED_FIELDS
        raise ValueError(
            f"Operation point validation error: Missing fields: {missing}, Extra fields: {extra}"
        )

def print_simulation_summary(solvers):
    """
    Return a list of pretty-printed lines summarizing all operation points.
    Never returns None.
    """
    width = 80
    sep = "-" * width

    times = []
    failed_points = []

    # Collect stats robustly
    for i, solver in enumerate(solvers):
        if solver is None:
            failed_points.append(i)
            continue
        # success flag
        ok = getattr(solver, "success", False)
        if not ok:
            failed_points.append(i)
        # elapsed time (may be None)
        t = getattr(solver, "elapsed_time", None)
        if t is not None:
            try:
                times.append(float(t))
            except Exception:
                pass  # ignore non-castable timings silently

    total_points = len(solvers)

    lines = [
        "",
        sep,
        "Final summary of performance analysis calculations".center(width),
        sep,
        f" Simulation successful for {total_points - len(failed_points)} out of {total_points} points",
    ]

    if failed_points:
        lines.append(f" Failed operation points: {', '.join(map(str, failed_points))}")

    if times:
        lines.extend([
            f" Average calculation time per operation point: {np.mean(times):.3f} seconds",
            f" Minimum calculation time of all operation points: {np.min(times):.3f} seconds",
            f" Maximum calculation time of all operation points: {np.max(times):.3f} seconds",
            f" Total calculation time for all operation points:   {np.sum(times):.3f} seconds",
        ])
    else:
        lines.append(" No valid calculation times available.")

    lines.append(sep)
    lines.append("")
    return lines

def print_boundary_conditions(BC):
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
    return "\n".join(lines)

def print_operation_points(operation_points):
    length = 80
    index_width = 8
    output = [
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
    output.append(header_str)
    output.append(unit_str)

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

# -------------------------------------------------------------------
# Enthalpy helper (kept for heuristic)
# -------------------------------------------------------------------
def calculate_enthalpy_residual_1(prop1, scale, h0, Ma, fluid, call, prop2):
    if isinstance(call, str):
        call_attr = getattr(jxp, call, None)
        if call_attr is None:
            raise ValueError(f"Invalid input type: {call}")
        call = call_attr
    props = fluid.get_state(call, prop1 * scale, prop2)
    return props["h"] - h0 + 0.5 * Ma**2 * props["speed_sound"] ** 2

def get_unknown(prop1, scale, h0, Ma, fluid, call, prop2):
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
    geometry,          # dict-of-arrays "arrayview"
    fluid,
    deviation_model,
):
    # --- robust cascade count (works for axial & radial) ---
    num_casc = _infer_num_cascades(geometry)

    p0_first = boundary_conditions["p0_in"]
    T0_first = boundary_conditions["T0_in"]
    p_final  = boundary_conditions["p_out"]
    angular_speed = boundary_conditions["omega"]
    alpha_first   = boundary_conditions["alpha_in"]

    # First stagnation properties
    stag_first = fluid.get_state(jxp.PT_INPUTS, p0_first, T0_first)
    h0_first = stag_first["h"]
    s_first  = stag_first["s"]
    d0_first = stag_first["d"]

    # Final isentropic
    static_is   = fluid.get_state(jxp.PSmass_INPUTS, p_final, s_first)
    h_final_s   = static_is["h"]
    a_final_s   = static_is["speed_sound"]

    # Spouting velocity
    v0 = np.sqrt(2 * (h0_first - h_final_s))

    # Exit enthalpy with guessed efficiency
    efficiency_ts = efficiency_tt / (1 + efficiency_tt * efficiency_ke)
    h0_final = h0_first - efficiency_ts * (h0_first - h_final_s)

    v_final = np.sqrt(
        2 * (h0_first - h_final_s - (h0_first - h0_final) / efficiency_tt)
    )
    h_final = h0_final - 0.5 * v_final**2

    # Exit static state for expansion with guessed efficiency
    static_properties_exit = fluid.get_state(jxp.HmassP_INPUTS, h_final, p_final)
    s_final = static_properties_exit["s"]

    # Linear entropy distribution
    entropy_distribution = np.linspace(s_first, s_final, int(num_casc) + 1)[1:]

    # Initial guess dictionary
    initial_guess = {}

    # Initialize inlet calculation
    s_in = s_first
    rothalpy = h0_first
    alpha_in = alpha_first
    d_in = d0_first

    # Ensure 'mach' is iterable of length num_casc
    if isinstance(mach, (list, tuple, np.ndarray)):
        if len(mach) != num_casc:
            raise ValueError(f"'mach' length {len(mach)} != number of cascades {num_casc}")
        mach_list = list(mach)
    else:
        mach_list = [mach] * num_casc

    for i in range(num_casc):
        geometry_cascade = {
            key: values[i]
            for key, values in geometry.items()
            if key not in ["number_of_cascades", "number_of_stages"]
        }

        radius_mean_in     = geometry_cascade["radius_mean_in"]
        radius_mean_throat = geometry_cascade["radius_mean_throat"]
        radius_mean_out    = geometry_cascade["radius_mean_out"]
        A_throat = geometry_cascade["A_throat"]
        A_out    = geometry_cascade["A_out"]
        A_in     = geometry_cascade["A_in"]

        # Entropy and Mach for this cascade
        s_out = entropy_distribution[i]
        ma_out = mach_list[i]

        # Exit pressure from guessed Ma (via PS with guessed s_out)
        blade_speed_out = angular_speed * (i % 2) * radius_mean_out
        h0_rel_out = rothalpy + 0.5 * blade_speed_out**2
        p_out = get_unknown(
            1.0, p0_first, h0_rel_out, ma_out, fluid, "PSmass_INPUTS", s_out
        )

        # Exit state
        static_out = fluid.get_state(jxp.PSmass_INPUTS, p_out, s_out)
        h_out   = static_out["h"]
        a_out   = static_out["speed_sound"]
        d_out   = static_out["d"]
        gamma_out = static_out["gamma"]

        # Exit velocity
        w_out = np.sqrt(2 * (h0_rel_out - h_out))

        # Critical Mach (placeholder 1.0; model available if wanted)
        static_props_is = fluid.get_state(jxp.PSmass_INPUTS, p_out, s_in)
        h_out_s = static_props_is["h"]
        eta = (h0_rel_out - h_out) / (h0_rel_out - h_out_s)
        ma_crit = 1.0

        # Exit flow angle (subsonic deviation model)
        beta_out = (-1) ** i * dm.get_subsonic_deviation(
            ma_out, ma_crit, {"A_throat": A_throat, "A_out": A_out}, deviation_model
        )

        # Mass flow rate
        mass_flow = d_out * w_out * math.cosd(beta_out) * A_out

        # Critical state at throat (for guess)
        w_throat_crit = a_out * ma_crit
        h_throat_crit = h0_rel_out - 0.5 * w_throat_crit**2
        s_throat_crit = s_out
        static_state_throat_crit = fluid.get_state(jxp.HmassSmass_INPUTS, h_throat_crit, s_throat_crit)
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
        if i != (num_casc - 1):
            A_next = geometry["A_in"][i + 1]
            radius_mean_next = geometry["radius_mean_in"][i + 1]
            velocity_triangle_out = evaluate_velocity_triangle_out(
                blade_speed_out, w_out, beta_out
            )
            v_m_in = velocity_triangle_out["v_m"] * A_out / A_next
            v_t_in = velocity_triangle_out["v_t"] * radius_mean_out / radius_mean_next
            v_in = np.sqrt(v_m_in**2 + v_t_in**2)
            alpha_in = math.arctand(v_t_in / v_m_in)
            blade_speed_in = angular_speed * ((i + 1) % 2) * radius_mean_next
            velocity_triangle_in = evaluate_velocity_triangle_in(
                blade_speed_in, v_in, alpha_in
            )
            h0_in = h_out + 0.5 * velocity_triangle_out["v"] ** 2
            h_in = h0_in - 0.5 * v_in**2
            rothalpy = (
                h_in + 0.5 * velocity_triangle_in["w"] ** 2 - 0.5 * blade_speed_in**2
            )
            s_in = s_out
            static_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)
            d_in = static_in["d"]

    # Inlet velocity from mass flow
    initial_guess["v_in"] = mass_flow / (
        d0_first * geometry["A_in"][0] * math.cosd(alpha_first)
    )

    return initial_guess

def latin_hypercube_sampling(bounds, n_samples):
    n_variables = len(bounds)
    sampler = qmc.LatinHypercube(d=n_variables, seed=1)
    unit_samples = sampler.random(n=n_samples)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, lower_bounds, upper_bounds)
