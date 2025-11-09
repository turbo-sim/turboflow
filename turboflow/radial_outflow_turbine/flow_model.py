# flow_model.py — updated to use BladeRow for per-row evaluation

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import turboflow as tf
import jaxprop as jxp

from .. import math
from .. import utilities as utils
from . import loss_model as lm
from . import choking_criterion as cm

from .blade_row import BladeRow
from .blade_row import evaluate_cascade_throat as _blade_throat
from .vaneless_channel import VanelessChannel 



# ============================================================
# Valid options for blockage model (kept for compatibility)
# ============================================================
BLOCKAGE_MODELS = ["flat_plate_turbulent"]


# ============================================================
# Public API: component-wise axial turbine evaluation (unchanged signature)
# ============================================================
def evaluate_axial_turbine_componentwise(
    variables: Dict[str, Any],
    boundary_conditions: Dict[str, Any],
    geom_list: List[Dict[str, Any]],                 # list[dict], one per component
    fluid: Any,
    reference_values: Dict[str, Any],
    components: List[Dict[str, Any]] | None = None,  # original YAML components (to read per-component model_options)
    model_options_global: Dict[str, Any] | None = None,  # global fallbacks
    benner_requires_throat: bool = True,  # kept for compatibility (now enforced inside BladeRow.from_dict)
):
    """
    Orchestrates the stator→interspace→rotor sequence using BladeRow objects.
    Returns the same structure as the original implementation.
    """

    n_comp = len(geom_list)
    names  = [c.get("name", f"component_{i+1}") for i, c in enumerate(geom_list)]
    ctypes = [str(c["cascade_type"]).lower() for c in geom_list]

    # Inlet (from BCs)
    h0_in    = boundary_conditions["h0_in"]
    s_in     = boundary_conditions["s_in"]
    alpha_in = boundary_conditions["alpha_in"]
    omega_bc = boundary_conditions["omega"]

    # Reference scales
    v0          = reference_values["v0"]

    planes_seq: List[Dict[str, Any]] = []
    cascades_seq: List[Dict[str, Any]] = []
    residuals: Dict[str, Any] = {}

    # inlet absolute speed (scaled → unscaled)
    v_in = variables["v_in"] * v0

    # Prepare initial inlet payload
    inlet = {"h0": h0_in, "s": s_in, "alpha": alpha_in, "v": v_in}

    # Build BladeRow objects per component (merges model options & validates geometry)
    rows: List[BladeRow] = []
    for i, g in enumerate(geom_list):
        per_opts = (components[i].get("model_options", {}) if (components and i < len(components)) else {})
        config_i = {
            "name": names[i],
            "cascade_type": ctypes[i],
            "geometry": g,
            "model_options": per_opts,
        }
        row = BladeRow.from_dict(config_i, fluid=fluid, model_options_global=model_options_global)
        rows.append(row)

    # Component-wise loop (now delegated to BladeRow.evaluate)
    for i, row in enumerate(rows):
        tag = f"_{i+1}"
        is_rotor = ("rotor" in row.cascade_type)
        omega_i = jnp.array(omega_bc if is_rotor else 0.0)

        # Unscale per-row unknowns + extract choking variables for this row
        row_vars, choking_vars = _extract_row_vars_and_choking(variables, i + 1, reference_values)

        # Evaluate this blade row (inlet→exit, +losses, +choking)
        planes, cascade, res_i, handoff = row.evaluate(
            inlet_state=inlet,
            row_vars=row_vars,
            omega=omega_i,
            reference_values=reference_values,
            choking_vars=choking_vars,
        )

        planes_seq += planes
        cascades_seq.append(cascade)
        residuals.update(_suffix_keys(res_i, tag))

        # Interspace mapping if current is stator and next is rotor, else pass-through
        if i < n_comp - 1:
            next_ctype = rows[i + 1].cascade_type
            if ("stator" in row.cascade_type) and ("rotor" in next_ctype):
                h0_in, s_in, alpha_in, v_in = _evaluate_vaneless_interspace(
                fluid=fluid,
                handoff=handoff,
                row_out_geom=row.geometry,                   # current row geom (dict)
                row_in_geom_next=rows[i+1].geometry,         # next row geom (dict)
                boundary_conditions=boundary_conditions,
                interspace_options=model_options_global.get("vaneless", {}) if model_options_global else None,)
                inlet = {"h0": h0_in, "s": s_in, "alpha": alpha_in, "v": v_in}
            else:
                exit_plane = planes[-1]
                inlet = {
                    "h0":    exit_plane["enthalpy0"],
                    "s":     exit_plane["entropy0"],
                    "alpha": exit_plane["alpha"],
                    "v":     exit_plane["v"],
                }

    # Collect arrays (unchanged)
    planes   = tf.combine_to_dict_of_arrays(planes_seq)
    cascades = tf.combine_to_dict_of_arrays(cascades_seq)

    # Outlet pressure residual (unchanged)
    p_calc = planes_seq[-1]["pressure"]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # Stage & overall KPIs (unchanged)
    stage   = compute_stage_performance_componentwise(planes, ctypes)
    overall = compute_overall_performance_componentwise(planes, boundary_conditions, reference_values, geom_list[-1])

    return {
        "planes": planes,
        "cascades": cascades,
        "stage": stage,
        "overall": overall,
        "residuals": residuals,
        "independent_variables": variables,
        "component_names": names,
        "component_types": ctypes,
        "geometry_components": geom_list,
    }


# ============================================================
# Helpers for the top-level orchestrator
# ============================================================

def _extract_row_vars_and_choking(
    variables: Dict[str, Any],
    index_1based: int,
    reference_values: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Unscale per-row unknowns for row 'index_1based' and gather choking vars.
    """
    tag = f"_{index_1based}"
    v0          = reference_values["v0"]
    s_range     = reference_values["s_range"]
    s_min       = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min   = reference_values["angle_min"]

    row_vars = {
        "w_out":    variables["w_out"   + tag] * v0,
        "s_out":    variables["s_out"   + tag] * s_range + s_min,
        "beta_out": variables["beta_out"+ tag] * angle_range + angle_min,
    }

    # Choking keys: include any variable with "crit" and this row's tag, plus global "v_crit_in" if present
    choking_vars = {}
    for key, val in variables.items():
        if ("crit" in key) and (tag in key):
            choking_vars[key.replace(tag, "")] = val
    if "v_crit_in" in variables:
        choking_vars["v_crit_in"] = variables["v_crit_in"]
    return row_vars, choking_vars


def _suffix_keys(d: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Append a component index suffix to residual keys, matching the original interface."""
    return {f"{k}{suffix}": v for k, v in d.items()}

def _evaluate_vaneless_interspace(
    *,
    fluid,
    handoff,            # dict from BladeRow.evaluate (exit of current row)
    row_out_geom,       # current row geometry dict (outlet side)
    row_in_geom_next,   # next row geometry dict (inlet side)
    boundary_conditions,
    interspace_options=None,  # optional dict of friction/heat/solver opts
):
    """
    Solve the vaneless interspace with VanelessChannel and return (h0_in_next, s_in_next, alpha_in_next, v_in_next).
    Expects VanelessChannel to accept:
      - geometry: r_in, r_out, A_in, A_out   (b_in/out will be computed)
      - operating_conditions: p_in, h0_in, v_m_in, v_t_in
    """

    # -------- Geometry (effective area at inlet includes blockage) --------
    A_in_eff  = row_out_geom["A_out"] * (1.0 - handoff["blockage_out"])
    r_in      = row_out_geom["radius_mean_out"]
    r_out     = row_in_geom_next["radius_mean_in"]
    A_out_tgt = row_in_geom_next["A_in"]

    geometry_cfg = {
        "r_in":  r_in,
        "r_out": r_out,
        "A_in":  A_in_eff,
        "A_out": A_out_tgt,
        # Optional: if your channel wants a specific axial span/angles, you can add:
        # "z_in": 0.0, "z_out": 0.05, "phi_in": 0.0, "phi_out": 0.0, "td_in": 0.01, "td_out": 0.01
        # but the class now fills safe defaults.
    }

    # -------- Operating conditions at interspace inlet (absolute frame) --------
    v_m_in = handoff["v_m_out"]
    v_t_in = handoff["v_t_out"]
    h0_in  = handoff["enthalpy0_out"]

    # Prefer a direct exit static pressure from the blade row if present;
    # otherwise reconstruct p_in from (h, s) if available; else fall back to BC s_in.
    p_in = handoff.get("p_out", None)
    if p_in is None:
        # reconstruct static enthalpy at row exit
        v_mag_exit = jnp.sqrt(v_m_in**2 + v_t_in**2)
        h_exit = h0_in - 0.5 * v_mag_exit**2
        s_exit = handoff.get("s_out", handoff.get("entropy_out", boundary_conditions["s_in"]))
        state_exit = fluid.get_state(jxp.HmassSmass_INPUTS, h_exit, s_exit)
        p_in = state_exit["p"]

    operating_conditions_cfg = {
        "p_in":   p_in,
        "h0_in":  h0_in,
        "v_m_in": v_m_in,
        "v_t_in": v_t_in,
        # omega = 0 in a vaneless, non-rotating interspace
        "omega":  0.0,
    }

    # -------- Options (friction/heat + solver) --------
    interspace_options = interspace_options or {}
    model_options_cfg  = interspace_options.get("model_options", {
        "friction_model": {"type": "aungier", "roughness": 1.0e-6, "Re_transition": 2300.0, "Re_width": 500.0},
        "heat_model":     {"type": "adiabatic"},
    })
    solver_options_cfg = interspace_options.get("solver_options", {
        "solver_name": "Dopri5",
        "adjoint_name": "DirectAdjoint",
        "rtol": 1.0e-6,
        "atol": 1.0e-6,
        "n_points": 50,
        "max_steps": 200,
        "throw": True,
    })

    config = {
        "name": "interspace",
        "geometry": geometry_cfg,
        "model_options": model_options_cfg,
        "solver_options": solver_options_cfg,
        "operating_conditions": operating_conditions_cfg,
    }

    # -------- Solve channel --------
    channel  = VanelessChannel.from_dict(config, fluid)
    solution = channel.evaluate()  # dict of arrays along meridional coordinate

    # -------- Map channel outlet → next-row inlet --------
    # Use the last point in the arrays returned by the solver
    v_m_out = solution["v_m"][-1]
    v_t_out = solution["v_t"][-1]
    v_out   = jnp.sqrt(v_m_out**2 + v_t_out**2)
    alpha   = math.arctand(v_t_out / v_m_out)

    # Prefer direct s/h0 if present, else reconstruct
    s_out = solution.get("s", None)
    if s_out is not None:
        s_out = s_out[-1]
    else:
        # reconstruct from (p,T) if available
        if ("p" in solution) and ("T" in solution):
            st = fluid.get_state(jxp.PT_INPUTS, solution["p"][-1], solution["T"][-1])
            s_out = st["s"]
        else:
            s_out = boundary_conditions["s_in"]

    if "h0" in solution:
        h0_out = solution["h0"][-1]
    else:
        if "h" in solution:
            h0_out = solution["h"][-1] + 0.5 * v_out**2
        else:
            # conservative fallback
            h0_out = h0_in

    return h0_out, s_out, alpha, v_out


# ============================================================
# Interspace mapping (unchanged)
# ============================================================

def evaluate_cascade_interspace(
    h0_exit,
    v_m_exit,
    v_t_exit,
    rho_exit,
    radius_exit,
    area_exit,
    blockage_exit,
    radius_inlet,
    area_inlet,
    fluid,
):
    """Propagate exit of i to inlet of i+1 (interspace)."""
    h0_in = h0_exit
    v_t_in = v_t_exit * radius_exit / radius_inlet
    v_m_in = v_m_exit * area_exit / area_inlet * (1 - blockage_exit)
    v_in = jnp.sqrt(v_t_in**2 + v_m_in**2)
    alpha_in = math.arctand(v_t_in / v_m_in)

    h_in = h0_in - 0.5 * v_in**2
    rho_in = rho_exit
    stagnation_properties = fluid.get_state(jxp.DmassHmass_INPUTS, rho_in, h_in)
    s_in = stagnation_properties["s"]

    return h0_in, s_in, alpha_in, v_in


def evaluate_cascade_throat(*args, **kwargs):
    return _blade_throat(*args, **kwargs)


# ============================================================
# Stage & overall KPIs (unchanged)
# ============================================================

def compute_stage_performance_componentwise(planes, component_types):
    """
    Stage reaction based on plane ordering: for each stage,
      stator_in, stator_out, rotor_in, rotor_out  (4 planes)
    """
    # derive number of stages by counting stator→rotor pairs
    pairs = 0
    for i in range(0, len(component_types) - 1):
        if ("stator" in component_types[i].lower()) and ("rotor" in component_types[i+1].lower()):
            pairs += 1
    number_of_stages = pairs
    if number_of_stages == 0:
        return {}

    h = planes["enthalpy"]
    # planes list is [stator_in, stator_out, rotor_in, rotor_out, ...]
    R = jnp.array(
        [
            (h[i * 4 + 1] - h[i * 4 + 3]) / (h[i * 4] - h[i * 4 + 3])
            for i in range(number_of_stages)
        ]
    )
    return {"reaction": R}


def compute_overall_performance_componentwise(planes, boundary_conditions, reference_values, last_geom):
    """Overall KPIs using last component’s geometry for blade-jet ratios."""
    angular_speed = boundary_conditions["omega"]
    v0 = reference_values["v0"]
    h_out_s = reference_values["h_out_s"]

    p  = planes["pressure"]
    p0 = planes["pressure0"]
    h0 = planes["enthalpy0"]
    v_out = planes["v"][-1]
    u_out = planes["blade_speed"][-1]
    mass_flow = planes["mass_flow"][-1]
    exit_flow_angle = planes["alpha"][-1]

    PR_tt = p0[0] / p0[-1]
    PR_ts = p0[0] / p[-1]
    h0_in = h0[0]
    h0_out = h0[-1]
    efficiency_tt = (h0_in - h0_out) / (h0_in - h_out_s - 0.5 * v_out**2) * 100
    efficiency_ts = (h0_in - h0_out) / (h0_in - h_out_s) * 100
    efficiency_ts_drop_kinetic = 0.5 * v_out**2 / (h0_in - h_out_s)
    efficiency_ts_drop_losses  = 1.0 - efficiency_ts - efficiency_ts_drop_kinetic
    power = mass_flow * (h0_in - h0_out)
    torque = power / angular_speed

    overall = {
        "PR_tt": PR_tt,
        "PR_ts": PR_ts,
        "mass_flow_rate": mass_flow,
        "efficiency_tt": efficiency_tt,
        "efficiency_ts": efficiency_ts,
        "efficiency_ts_drop_kinetic": efficiency_ts_drop_kinetic,
        "efficiency_ts_drop_losses": efficiency_ts_drop_losses,
        "power": power,
        "torque": torque,
        "angular_speed": jnp.array(angular_speed),
        "exit_flow_angle": exit_flow_angle,
        "exit_velocity": v_out,
        "spouting_velocity": jnp.array(v0),
        "last_blade_velocity": u_out,
        "blade_jet_ratio": u_out / v0,
        "h0_in": h0_in,
        "h0_out": h0_out,
        "h_out_s": jnp.array(h_out_s),
        "specific_speed": angular_speed * (mass_flow / reference_values["d_out_s"]) ** 0.5 / ((h0_in - h_out_s) ** 0.75),
        # blade-jet ratios using last component’s radii if available
        "blade_jet_ratio_mean": jnp.array(angular_speed * last_geom["radius_mean_out"] / v0),
        "blade_jet_ratio_hub": jnp.array(angular_speed * last_geom.get("radius_hub_out", jnp.nan) / v0) if "radius_hub_out" in last_geom else jnp.nan,
        "blade_jet_ratio_tip": jnp.array(angular_speed * last_geom.get("radius_tip_out", jnp.nan) / v0) if "radius_tip_out" in last_geom else jnp.nan,
    }
    return overall
