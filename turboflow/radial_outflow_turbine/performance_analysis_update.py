# performance_analysis.py  (mapping-based dispatch; component-wise geometry; tree_at updates)

from __future__ import annotations

import os
import yaml
import copy
import datetime
import itertools
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import equinox as eqx

from scipy.stats import qmc
from scipy import optimize

from .. import math
from .. import pysolver_view as psv
from .. import utilities as utils
from . import geometry_model_axial as geom_ax
from . import geometry_model_radial as geom_rad
from . import flow_model_update as flow
from . import deviation_model as dm
import jaxprop as jxp
import jaxprop.perfect_gas as pg

import turboflow as tf

from .blade_row import (
    evaluate_velocity_triangle_out,
    evaluate_velocity_triangle_in,
)
from .blade_row import BladeRow
from .vaneless_channel import VanelessChannel

jax.config.update("jax_enable_x64", True)

SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}
NUMERIC = (int, float, np.floating)


# =========================== mappings ===========================

# Component type → class
COMPONENT_CLASSES = {
    "axial_cascade": BladeRow,
    "radial_cascade": BladeRow,
    "vaneless_channel": VanelessChannel,
}

GEOMETRY_BUILDERS = {
    # For axial cascades: pass [comp] and take the first (and only) geometry dict.
    "axial_cascade":  lambda comp: geom_ax.calculate_full_geometry([comp])[0],

    # For radial cascades: same idea – calculate_full_geometry expects a list.
    "radial_cascade": lambda comp: geom_rad.calculate_full_geometry([comp])[0],

    # Vaneless channel: geometry is already “ready” in the YAML.
    "vaneless_channel": lambda comp: copy.deepcopy(comp.get("geometry", {})),
}


# ====================== small helpers ======================

def _is_num(x): return isinstance(x, NUMERIC)

def assert_numeric_operation_point(op):
    for k, v in op.items():
        if k == "fluid_name":
            if not isinstance(v, str):
                raise TypeError(f"operation_point['fluid_name'] must be str, got {type(v)}")
            continue
        if not _is_num(v):
            raise TypeError(f"operation_point['{k}'] must be numeric, got {v!r} ({type(v)})")

def _looks_like_expr(s: str) -> bool:
    return any(t in s for t in ("np.", "(", ")", "[", "]", "*", "/", "+", "-", "**"))

def _eval_item_if_str(x, ctx):
    if isinstance(x, str):
        if not _looks_like_expr(x):
            return x
        try:
            return eval(x, {"__builtins__": {}, "np": np}, ctx)
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
    ctx.update({k: v for k, v in pm.items()
                if isinstance(v, list) and all(isinstance(e, (int, float)) for e in v)})
    for k, v in list(pm.items()):
        evaluated = _eval_item_if_str(v, ctx)
        if hasattr(evaluated, "tolist"):
            evaluated = evaluated.tolist()
        pm[k] = evaluated
        if isinstance(evaluated, (int, float)):
            ctx[k] = evaluated
        elif isinstance(evaluated, list) and all(isinstance(e, (int, float)) for e in evaluated):
            ctx[k] = evaluated
    return pm

def _numpy_to_native(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    elif isinstance(x, (list, tuple)):
        return type(x)(_numpy_to_native(e) for e in x)
    else:
        return x

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
    REQUIRED_FIELDS = {"fluid_name", "p0_in", "T0_in", "p_out", "alpha_in", "omega"}
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
        if key == "T0_in": return value - 273.15
        if key == "omega": return (value * 60) / (2 * np.pi)
        if key == "alpha_in": return np.degrees(value)
        if key in ["p0_in", "p_out"]: return value / 1.0e3
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
    lines.append(f" {'Fluid: ':<{column_width}} {BC['fluid_name']:<}")
    lines.append(f" {'Flow angle in: ':<{column_width}} {BC['alpha_in']:<.2f} deg")
    lines.append(f" {'Total temperature in: ':<{column_width}} {BC['T0_in'] - 273.15:<.2f} degC")
    lines.append(f" {'Total pressure in: ':<{column_width}} {BC['p0_in'] / 1e5:<.3f} bar")
    lines.append(f" {'Static pressure out: ':<{column_width}} {BC['p_out'] / 1e5:<.3f} bar")
    lines.append(f" {'Angular speed: ':<{column_width}} {BC['omega'] * 60 / 2 / np.pi:<.1f} RPM")
    lines.append("-" * 80)
    lines.append("")
    return "\n".join(lines)

def print_simulation_summary(solvers):
    width = 80
    sep = "-" * width
    times, failed_points = [], []
    for i, solver in enumerate(solvers):
        if solver is None:
            failed_points.append(i); continue
        ok = getattr(solver, "success", False)
        if not ok: failed_points.append(i)
        t = getattr(solver, "elapsed_time", None)
        if t is not None:
            try: times.append(float(t))
            except Exception: pass
    total_points = len(solvers)
    lines = ["", sep, "Final summary of performance analysis calculations".center(width), sep,
             f" Simulation successful for {total_points - len(failed_points)} out of {total_points} points"]
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
    lines.append(sep); lines.append("")
    return lines

def latin_hypercube_sampling(bounds, n_samples):
    n_variables = len(bounds)
    sampler = qmc.LatinHypercube(d=n_variables, seed=1)
    unit_samples = sampler.random(n=n_samples)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, lower_bounds, upper_bounds)


# ===================== component-wise geometry =====================

def _build_component_geometry(components):
    geoms = []
    for comp in components:
        ctype = str(comp.get("component_type", "")).lower()
        builder = GEOMETRY_BUILDERS.get(ctype)
        if builder is None:
            geoms.append(copy.deepcopy(comp.get("geometry", {})))
        else:
            geoms.append(builder(comp))
    return geoms


# =================== tiny dynamic state pytree ===================

class TurbineState(eqx.Module):
    # per-component inlet state
    h0_in: jnp.ndarray
    s_in: jnp.ndarray
    v_in: jnp.ndarray
    alpha_in: jnp.ndarray  # degrees

    # per-row unknowns
    w_out: jnp.ndarray
    s_out: jnp.ndarray
    beta_out: jnp.ndarray
    w_crit_throat: jnp.ndarray
    s_crit_throat: jnp.ndarray
    v_crit_in: jnp.ndarray


# ============================ public API ============================

def compute_performance(
    operation_points,
    config,
    out_filename=None,
    out_dir="output",
    stop_on_failure=False,
    export_results=True,
    logger=None
):
    if not config.get("components"):
        raise ValueError("No 'components' found in config. Provide a list of components.")

    if isinstance(operation_points, dict):
        operation_points = generate_operation_points(operation_points)
    elif not isinstance(operation_points, (list, np.ndarray)):
        raise TypeError("operation_points must be either list of dicts or a dict with ranges.")

    for op in operation_points:
        validate_operation_point(op)
        assert_numeric_operation_point(op)

    operation_point_data, overall_data = [], []
    plane_data, cascade_data, stage_data = [], [], []
    solver_data, solution_data, geometry_data = [], [], []
    solver_container = []

    message = print_operation_points(operation_points)
    for line in message.splitlines():
        logger.info(line)

    for i, operation_point in enumerate(operation_points):
        logger.info("")
        logger.info(f" Computing operation point {i+1} of {len(operation_points)}")
        for line in print_boundary_conditions(operation_point).splitlines():
            logger.info(line)

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

        solver, results = compute_single_operation_point(
            operation_point,
            initial_guess_cfg,
            config["components"],
            config.get("simulation_options", {}),
            config["performance_analysis"]["solver_options"],
            logger=logger
        )

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

        operation_point_data.append(pd.DataFrame([operation_point]))
        overall_data.append(pd.DataFrame.from_dict(results["overall"], orient="index").T)
        plane_data.append(utils.flatten_dataframe(pd.DataFrame(results["planes"])))
        cascade_data.append(utils.flatten_dataframe(pd.DataFrame(results["cascades"])))
        stage_data.append(utils.flatten_dataframe(pd.DataFrame(results["stage"])))
        geom_rows_df = pd.DataFrame(results["geometry_components"])
        geometry_data.append(utils.flatten_dataframe(geom_rows_df))
        solver_data.append(pd.DataFrame([solver_status]))
        solution_data.append(solver.problem.vars_real)
        solver_container.append(solver)

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
            config_data = {k: v for k, v in config.items() if v}
            config_data = utils.convert_numpy_to_python(config_data, precision=12)
            with open(os.path.join(out_dir, f"{fname}.yaml"), "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            filepath_xlsx = os.path.join(out_dir, f"{fname}.xlsx")
            with pd.ExcelWriter(filepath_xlsx, engine="openpyxl") as writer:
                for sheet_name, df in dfs.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)

            filepath_pkl = os.path.join(out_dir, f"{fname}.pkl")
            solver = solver_container[-1]
            solver.problem = None
            import dill
            with open(filepath_pkl, 'wb') as f:
                dill.dump(solver, f)

        logger.info(f" Performance data successfully written to {filepath_xlsx}")

    message = print_simulation_summary(solver_container)
    for line in message:
        logger.info(line)

    return solver_container


# ================= one operation point (solver) =================

def compute_single_operation_point(
    operating_point,
    initial_guess,
    components,
    simulation_options,
    solver_options,
    logger=None
):
    
    fluid_name = operating_point["fluid_name"]
    problem = AxialTurbineProblem(components, simulation_options)
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(solver_options)
    

    # ---- per-row guesses via BladeRow objects
    omega = problem.boundary_conditions["omega"]
    alpha_in = problem.boundary_conditions["alpha_in"]
    alpha_in_deg = np.degrees(alpha_in) if abs(alpha_in) <= np.pi*1.01 else alpha_in

    inlet_seed = {
        "h0": problem.boundary_conditions["h0_in"],
        "s":  problem.boundary_conditions["s_in"],
        "alpha": alpha_in_deg,
        "v": 0.5 * problem.reference_values["v0"],
    }

    row_guess_dict = {}
    row_counter = 0
    for obj in problem.comp_objects:
        if not isinstance(obj, BladeRow):
            continue
        row_counter += 1
        tag = f"_{row_counter}"
        is_rotor = ("rotor" in str(obj.cascade_type).lower())
        omega_i = omega if is_rotor else 0.0
        ig_row = obj.build_initial_guess(
            inlet_state=inlet_seed,
            omega=jnp.array(omega_i),
            choking_criterion=simulation_options.get("choking_criterion", "critical_mach_number"),
        )
        row_guess_dict[f"w_out{tag}"] = ig_row["w_out"]
        row_guess_dict[f"s_out{tag}"] = ig_row["s_out"]
        row_guess_dict[f"beta_out{tag}"] = ig_row["beta_out"]
        if "w_crit_throat" in ig_row: row_guess_dict[f"w_crit_throat{tag}"] = ig_row["w_crit_throat"]
        if "s_crit_throat" in ig_row: row_guess_dict[f"s_crit_throat{tag}"] = ig_row["s_crit_throat"]
        if "v_crit_in" in ig_row:     row_guess_dict[f"v_crit_in{tag}"] = ig_row["v_crit_in"]

    if "v_in" not in row_guess_dict:
        row_guess_dict["v_in"] = inlet_seed["v"]

    # prune to match choking (component-wise overrides beat global)
    per_cascade_choking = []
    for comp in components:
        ctype = str(comp.get("component_type","")).lower()
        if ctype not in ("axial_cascade", "radial_cascade"):
            continue
        comp_opts = (comp.get("model_options") or {})
        per_cascade_choking.append(
            comp_opts.get("choking_criterion", simulation_options.get("choking_criterion", "critical_mach_number"))
        )

    pruned = {"v_in": row_guess_dict["v_in"]}
    for i in range(len(per_cascade_choking)):
        tag = f"_{i+1}"
        pruned[f"w_out{tag}"] = row_guess_dict[f"w_out{tag}"]
        pruned[f"s_out{tag}"] = row_guess_dict[f"s_out{tag}"]
        pruned[f"beta_out{tag}"] = row_guess_dict[f"beta_out{tag}"]
        crit = str(per_cascade_choking[i]).lower()
        if crit == "critical_mach_number":
            if f"w_crit_throat{tag}" in row_guess_dict:
                pruned[f"w_crit_throat{tag}"] = row_guess_dict[f"w_crit_throat{tag}"]
            if f"s_crit_throat{tag}" in row_guess_dict:
                pruned[f"s_crit_throat{tag}"] = row_guess_dict[f"s_crit_throat{tag}"]
        elif crit == "critical_isentropic_throat":
            if f"w_crit_throat{tag}" in row_guess_dict:
                pruned[f"w_crit_throat{tag}"] = row_guess_dict[f"w_crit_throat{tag}"]
        elif crit == "critical_mass_flow_rate":
            if f"v_crit_in{tag}" in row_guess_dict:
                pruned[f"v_crit_in{tag}"] = row_guess_dict[f"v_crit_in{tag}"]
            if f"w_crit_throat{tag}" in row_guess_dict:
                pruned[f"w_crit_throat{tag}"] = row_guess_dict[f"w_crit_throat{tag}"]
            if f"s_crit_throat{tag}" in row_guess_dict:
                pruned[f"s_crit_throat{tag}"] = row_guess_dict[f"s_crit_throat{tag}"]
    row_guess_dict = pruned

    # pack & scale for solver
    initial_guess_scaled = problem.scale_values(row_guess_dict)
    x0 = np.array(list(initial_guess_scaled.values()), dtype=float)
    problem.keys = list(initial_guess_scaled.keys())
    if not np.all(np.isfinite(x0)):
        bad = {k: v for k, v in zip(problem.keys, x0) if not np.isfinite(v)}
        raise ValueError(f"Initial guess contains non-finite values: {bad}")

    solver_methods = [solver_options["method"]] + [m for m in SOLVER_MAP.keys() if m != solver_options["method"]]
    for method in solver_methods:
        solver_options["method"] = method
        solver = psv.NonlinearSystemSolver(problem, logger=logger, **solver_options)
        try:
            solver.solve(x0)
        except Exception as e:
            if solver.func_count == 0:
                raise e
            if logger:
                logger.info(f" Error during solving: {e}")
            solver.success = False
        if solver.success:
            break

    if not solver.success and logger:
        logger.info("WARNING: All attempts failed to converge")

    return solver, problem.results


# =================== problem (mapping-based) ===================

class AxialTurbineProblem(psv.NonlinearSystemProblem):
    """
    Component-wise turbine analysis.

    Responsibilities:
      - Build per-component geometry once (via _build_component_geometry).
      - Instantiate BladeRow / VanelessChannel objects once from the YAML components + geometry.
      - Inject the fluid into each component using eqx.tree_at in update_boundary_conditions.
      - Provide residual(x) that delegates to flow.evaluate_axial_turbine_componentwise.
      - Provide scale_values(...) and gradient(...) for the solver.
    """

    def __init__(self, components, simulation_options):
        self.components = components             # raw YAML component dicts
        self.model_options = simulation_options  # global sim options (e.g. choking criterion)
        self.keys = []                           # filled by compute_single_operation_point

        # 1) Build geometry per component (component-wise geometry, same length as components)
        self.geometry_components = _build_component_geometry(self.components)

        # 2) Instantiate component objects
        #
        #    - For cascades: BladeRow objects (fluid injected later).
        #    - For vaneless_channel: VanelessChannel objects (fluid injected later).
        #    - Other component types: kept as-is (must be handled explicitly in flow_model).
        comp_objs: list[Any] = []
        cascade_geoms: list[Dict[str, Any]] = []
        cascade_indices: list[int] = []

        for idx, (comp, geom) in enumerate(zip(self.components, self.geometry_components)):
            ctype = str(comp.get("component_type", "")).lower()

            # ------------------------------
            # Axial or radial cascade → BladeRow
            # ------------------------------
            if ctype in ("axial_cascade", "radial_cascade"):
                cfg_row = {
                    "name": comp.get("name", f"row_{idx+1}"),
                    "cascade_type": geom["cascade_type"],
                    "geometry": geom,
                    "model_options": comp.get("model_options", {}),
                    "initial_guess": comp.get("initial_guess", {}),
                }
                # fluid=None for now; we inject it later in update_boundary_conditions
                row = BladeRow.from_dict(
                    cfg_row,
                    fluid=None,
                    model_options_global=self.model_options,
                )
                comp_objs.append(row)
                cascade_geoms.append(geom)
                cascade_indices.append(idx)
                continue

            # ------------------------------
            # Vaneless channel → VanelessChannel
            # ------------------------------
            if ctype == "vaneless_channel":
                cfg_ch = {
                    "name": comp.get("name", f"channel_{idx+1}"),
                    "geometry": geom,
                    "model_options": comp.get("model_options", {}),
                    # Dummy operating conditions: required by from_dict, but not actually
                    # used by the turbine component-wise framework. We'll drive the
                    # channel from the upstream cascade instead.
                    "operating_conditions": {
                        "p_in": 1.0e5,    # arbitrary but valid
                        "h_in": 1.0e5,    # arbitrary but valid
                        "v_in": 1.0,      # non-zero so nothing divides by zero
                        "alpha_in": 0.0,  # degrees
                    },
                }
                # fluid=None for now; injected later
                ch = VanelessChannel.from_dict(
                    cfg_ch,
                    fluid=None,
                )
                comp_objs.append(ch)
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
        self.fluid = None

        # Optional: arrayview for cascades (legacy helpers / diagnostics)
        self.geometry_components_arrayview = self._to_array_geometry(self.geometry_components_cascades)

    # ------------------------------------------------------------------
    # Helper: convert list-of-dicts geometry → dict-of-lists "arrayview"
    # ------------------------------------------------------------------
    def _to_array_geometry(self, rows):
        """
        Utility to keep your old 'arrayview' style geometry if needed:
        geometry_components_arrayview["A_in"][i] == rows[i]["A_in"], etc.
        """
        if not rows:
            return {"number_of_cascades": 0, "number_of_stages": 0}

        all_keys = set().union(*[row.keys() for row in rows])
        array_geom = {
            "number_of_cascades": len(rows),
            "number_of_stages": max(0, len(rows) // 2),
        }
        for k in all_keys:
            array_geom[k] = [row.get(k) for row in rows]
        return array_geom

    # ------------------------------------------------------------------
    # Boundary conditions + fluid injection
    # ------------------------------------------------------------------
    def update_boundary_conditions(self, operation_point):
        """
        Set boundary conditions, reference values, and inject fluid into
        all component objects (BladeRow, VanelessChannel, ...).
        """
        assert_numeric_operation_point(operation_point)
        self.boundary_conditions = operation_point

        # ---- Create fluid ONCE ----
        fluid = jxp.FluidJAX(operation_point["fluid_name"])
        self.fluid = fluid

        # ---- Inject fluid into all components that have a `.fluid` field ----
        new_list = []
        for comp in self.comp_objects:
            if hasattr(comp, "fluid"):
                comp = eqx.tree_at(lambda c: c.fluid, comp, fluid)
            new_list.append(comp)
        self.comp_objects = new_list

        # ---- Compute inlet stagnation state ----
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        p_out = operation_point["p_out"]

        st_in = fluid.get_state(jxp.PT_INPUTS, p0_in, T0_in)
        h0_in = st_in["h"]
        s_in  = st_in["s"]

        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"]  = s_in

        # ---- Isentropic outlet ----
        st_out_s = fluid.get_state(jxp.PSmass_INPUTS, p_out, s_in)
        h_out_s = st_out_s["h"]
        d_out_s = st_out_s["d"]

        # ---- Reference velocity ----
        v0 = np.sqrt(2 * (h0_in - h_out_s))

        # ---- Reference mass flow (use last component with A_out) ----
        A_out = None
        for g in reversed(self.geometry_components):
            if "A_out" in g:
                A_out = g["A_out"]
                break

        if A_out is None:
            raise ValueError("Cannot determine reference A_out (no component with 'A_out' found).")

        mass_flow_ref = A_out * d_out_s * v0

        # ---- Store reference values ----
        self.reference_values = {
            "v0": v0,
            "h_out_s": h_out_s,
            "d_out_s": d_out_s,
            "mass_flow_ref": mass_flow_ref,
            "s_min": s_in,
            "s_range": fluid.get_state(jxp.HmassP_INPUTS, h0_in, p_out)["s"] - s_in,
            "angle_min": -90.0,
            "angle_range": 180.0,
        }

        # ---- Inlet angle in degrees ----
        alpha = operation_point["alpha_in"]
        alpha_deg = np.degrees(alpha) if abs(alpha) < np.pi * 1.1 else alpha
        self.boundary_conditions["alpha_deg"] = float(alpha_deg)

    # ------------------------------------------------------------------
    # Scaling utilities
    # ------------------------------------------------------------------
    def scale_values(self, variables, to_normalized=True):
        """
        Convert values between normalized and real values using reference_values.
        Keys:
          - starting with "v" or "w": scale by v0
          - starting with "s":        scale by s_range/s_min
          - starting with "b":        scale by angle_range/angle_min
        """
        v0          = self.reference_values["v0"]
        s_range     = self.reference_values["s_range"]
        s_min       = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min   = self.reference_values["angle_min"]

        scaled_variables = {}
        for key, val in variables.items():
            if key.startswith(("v", "w")):
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

    # ------------------------------------------------------------------
    # Residual & gradient
    # ------------------------------------------------------------------
    def residual(self, x):
        """
        Map the solver vector x → dict of scaled vars → call
        flow.evaluate_axial_turbine_componentwise.
        """
        try:
            # 1) unpack x into a dict with the keys determined in compute_single_operation_point
            self.vars_scaled = dict(zip(self.keys, x))

            # 2) Call the flow model with *scaled* variables; it will unscale internally
            result = flow.evaluate_axial_turbine_componentwise(
                variables=self.vars_scaled,
                boundary_conditions=self.boundary_conditions,
                comp_objects=self.comp_objects,
                geometry_components=self.geometry_components,   # NOTE: all components
                fluid=self.fluid,
                reference_values=self.reference_values,
            )

            # 3) stash results for export
            self.results = result

            # 4) flatten residuals dict into the vector the solver expects
            return jnp.array(list(result["residuals"].values()))
        except Exception as e:
            bc_types = {k: type(v).__name__ for k, v in getattr(self, "boundary_conditions", {}).items()}
            raise TypeError(
                f"Residual failed: {e}\n"
                f"  OP types: {bc_types}\n"
                f"  Keys in vars_scaled: {list(getattr(self, 'vars_scaled', {}).keys())}"
            ) from e

    def gradient(self, x):
        """
        Jacobian of the residual w.r.t. x, used by the nonlinear solver.
        """
        return jax.jacfwd(self.residual, argnums=0)(x)


# ================= IG & distance utilities (unchanged) =================

def extract_initial_guess_from_components(components):
    ig, eff_tt, eff_ke, ma_list = {}, None, None, []
    for comp in components:
        ctype = str(comp.get("component_type", "")).lower()
        if ctype not in ("axial_cascade", "radial_cascade"):
            continue
        ig_c = comp.get("initial_guess", {}) or {}
        if eff_tt is None and "efficiency_tt" in ig_c:
            eff_tt = ig_c["efficiency_tt"]
        if eff_ke is None and "efficiency_ke" in ig_c:
            eff_ke = ig_c["efficiency_ke"]
        ma = (ig_c.get("ma") or ig_c.get("ma_out") or ig_c.get("ma_rel_out")
              or ig_c.get("ma_exit") or ig_c.get("ma_2") or ig_c.get("ma_1"))
        ma_list.append(ma if isinstance(ma, (int, float)) else None)
    marker = {}
    if eff_tt is not None: marker["efficiency_tt"] = eff_tt
    if eff_ke is not None: marker["efficiency_ke"] = eff_ke
    if ma_list:
        for i, m in enumerate(ma_list):
            marker[f"ma_{i+1}"] = 0.8 if m is None else m
    return marker if marker else {"_empty_": True}

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
