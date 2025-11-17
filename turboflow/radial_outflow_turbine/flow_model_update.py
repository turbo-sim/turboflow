# flow_model.py — component-object dispatcher for performance_analysis

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
# Helpers
# ============================================================


def _alpha_deg(a: jnp.ndarray) -> jnp.ndarray:
    """Ensure angle is in degrees (convert from radians if it looks like radians)."""
    return jnp.degrees(a) if jnp.abs(a) <= (jnp.pi * 1.01) else a


def _suffix_keys(d: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Append suffix (e.g. '_1', '_2') to all keys in dict."""
    return {f"{k}{suffix}": v for k, v in d.items()}


def _extract_row_vars_and_choking(
    variables: Dict[str, Any],
    index_1based: int,
    reference_values: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Unscale per-row unknowns for row `index_1based` from solver variables.

    Expected normalized keys:
        w_out_i, s_out_i, beta_out_i, (optional) w_crit_throat_i, s_crit_throat_i, v_crit_in

    Returns
    -------
    row_vars : dict
        Unscaled row unknowns: {w_out, s_out, beta_out}
    choking_vars : dict
        Unscaled choking-related unknowns (subset, if present):
        {w_crit_throat, s_crit_throat, v_crit_in}
    """
    tag = f"_{index_1based}"

    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    a_range = reference_values["angle_range"]
    a_min = reference_values["angle_min"]

    row_vars = {
        "w_out": variables[f"w_out{tag}"] * v0,
        "s_out": variables[f"s_out{tag}"] * s_range + s_min,
        "beta_out": variables[f"beta_out{tag}"] * a_range + a_min,
    }

    choking_vars: Dict[str, Any] = {}
    if f"w_crit_throat{tag}" in variables:
        choking_vars["w_crit_throat"] = variables[f"w_crit_throat{tag}"] * v0
    if f"s_crit_throat{tag}" in variables:
        choking_vars["s_crit_throat"] = (
            variables[f"s_crit_throat{tag}"] * s_range + s_min
        )
    if "v_crit_in" in variables:
        choking_vars["v_crit_in"] = variables["v_crit_in"] * v0

    return row_vars, choking_vars


# ============================================================
# Public API: evaluate an already-instantiated component list
# ============================================================


def evaluate_axial_turbine_componentwise(
    variables: Dict[str, Any],  # solver vars (normalized)
    boundary_conditions: Dict[str, Any],
    comp_objects: List[Any],  # BladeRow / VanelessChannel objects (in order)
    geometry_components: List[
        Dict[str, Any]
    ],  # component-wise geometry (same order as comp_objects)
    fluid: Any,
    reference_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Walk the component objects in order, evaluate each, chain inlet→outlet, and
    accumulate residuals and performance metrics.

    Notes
    -----
    - For BladeRow: we unscale per-row unknowns from `variables` using row order
      tags _1, _2, ...
    - For VanelessChannel: we call obj.evaluate(...) and map its outlet to the
      next inlet, but don't treat it as a cascade for stage/overall KPIs.
    """

    # ---------- inlet from boundary conditions ----------
    h0_in = boundary_conditions["h0_in"]
    s_in = boundary_conditions["s_in"]
    alpha_in_deg = _alpha_deg(boundary_conditions["alpha_in"])

    # inlet velocity from normalized var if present
    v_in = reference_values["v0"] * variables.get("v_in", 0.0)
    inlet = {"h0": h0_in, "s": s_in, "alpha": alpha_in_deg, "v": v_in}

    omega_bc = boundary_conditions["omega"]

    # ---------- collectors ----------
    planes_seq: List[Dict[str, Any]] = []
    cascades_seq: List[Dict[str, Any]] = []
    residuals: Dict[str, Any] = {}

    row_counter = 0
    cascade_geoms: List[Dict[str, Any]] = []  # cascades only (for stage KPIs etc.)

    # ---------- main component loop ----------
    for gi, (obj, geom) in enumerate(zip(comp_objects, geometry_components)):
        # =====================================================
        # Blade row (axial or radial cascade)
        # =====================================================
        if isinstance(obj, BladeRow):
            row_counter += 1
            cascade_geoms.append(geom)

            # unscale row vars + choking vars for this row
            row_vars, choking_vars = _extract_row_vars_and_choking(
                variables=variables,
                index_1based=row_counter,
                reference_values=reference_values,
            )

            # rotor rows rotate; stators do not
            is_rotor = "rotor" in str(obj.cascade_type).lower()
            omega_i = jnp.asarray(omega_bc if is_rotor else 0.0, dtype=jnp.float64)

            planes, cascade, res_i, handoff = obj.evaluate(
                inlet_state=inlet,
                row_vars=row_vars,
                omega=omega_i,
                reference_values=reference_values,
                choking_vars=choking_vars,
            )

            planes_seq += planes
            cascades_seq.append(cascade)
            residuals.update(_suffix_keys(res_i, f"_{row_counter}"))

            # chain inlet for next component (absolute frame)
            exit_plane = planes[-1]
            inlet = {
                "h0": exit_plane["enthalpy0"],
                "s": exit_plane["entropy"],
                "alpha": exit_plane["alpha"],  # already degrees
                "v": exit_plane["v"],
            }
            continue

        # =====================================================
        # Vaneless channel
        # =====================================================
        if isinstance(obj, VanelessChannel):
            # Map BladeRow-style inlet to channel operating conditions (static)
            v_mag = inlet["v"]
            alphaD = _alpha_deg(inlet["alpha"])
            h_in = inlet["h0"] - 0.5 * v_mag**2
            st = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, inlet["s"])
            p_in = st["p"]

            # Call channel; expected to return either:
            #   (planes_ch, res_ch)  or  {"planes": ..., "residuals": ...}
            result = obj.evaluate(
                inlet_state={
                    "p_in": p_in,
                    "h_in": h_in,
                    "v_in": v_mag,
                    "alpha_in": alphaD,  # degrees
                },
                fluid=fluid,
                reference_values=reference_values,
            )

            if isinstance(result, tuple) and len(result) == 2:
                planes_ch, res_ch = result
            elif isinstance(result, dict):
                planes_ch = result.get("planes")
                res_ch = result.get("residuals", {})
            else:
                raise TypeError("VanelessChannel.evaluate returned unsupported type.")

            # optional residual from channel (e.g., mass error)
            if isinstance(res_ch, dict) and "mass_error_exit" in res_ch:
                residuals[f"vaneless_mass_error_{gi+1}"] = res_ch["mass_error_exit"]

            # Update inlet to next component
            v_m_out = planes_ch[1]["v_m"]
            v_t_out = planes_ch[1]["v_t"]
            v_out = jnp.sqrt(v_m_out**2 + v_t_out**2)
            alpha = jnp.degrees(jnp.arctan2(v_t_out, v_m_out))
            h0_out = planes_ch[1].get(
                "enthalpy0", planes_ch[1]["enthalpy"] + 0.5 * v_out**2
            )
            s_out = planes_ch[1].get("entropy", inlet["s"])

            inlet = {"h0": h0_out, "s": s_out, "alpha": alpha, "v": v_out}
            continue

        # =====================================================
        # Unsupported component type
        # =====================================================
        raise ValueError(
            f"Unsupported component object type at index {gi}: {type(obj).__name__}"
        )

    # ---------- aggregate & KPIs ----------
    if not planes_seq:
        raise RuntimeError(
            "No cascades evaluated; cannot compute outlet residuals/overall KPIs."
        )

    planes = tf.combine_to_dict_of_arrays(planes_seq)
    cascades = tf.combine_to_dict_of_arrays(cascades_seq)

    # outlet pressure residual uses last cascade exit static pressure
    p_calc = planes_seq[-1]["pressure"]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # stage & overall KPIs (using cascade-only geometries)
    stage = compute_stage_performance_componentwise(
        planes, [g["cascade_type"] for g in cascade_geoms]
    )
    overall = compute_overall_performance_componentwise(
        planes,
        boundary_conditions,
        reference_values,
        cascade_geoms[-1],
    )

    return {
        "planes": planes,
        "cascades": cascades,
        "stage": stage,
        "overall": overall,
        "residuals": residuals,
        "independent_variables": variables,
        "component_names": [
            getattr(o, "name", f"component_{i+1}") for i, o in enumerate(comp_objects)
        ],
        "component_types": [
            (o.cascade_type if isinstance(o, BladeRow) else "vaneless_channel")
            for o in comp_objects
        ],
        "geometry_components": cascade_geoms,  # cascades-only list for reporting
    }


# ============================================================
# Thin re-export for compatibility where needed
# ============================================================


def evaluate_cascade_throat(*args, **kwargs):
    return _blade_throat(*args, **kwargs)


# ============================================================
# Stage & overall KPIs
# ============================================================


def compute_stage_performance_componentwise(planes, component_types):
    """
    Stage reaction based on plane ordering: per stage we expect
      [stator_in, stator_out, rotor_in, rotor_out] → 4 planes.
    """
    # count stator→rotor pairs
    pairs = 0
    for i in range(0, len(component_types) - 1):
        if ("stator" in component_types[i].lower()) and (
            "rotor" in component_types[i + 1].lower()
        ):
            pairs += 1
    number_of_stages = pairs
    if number_of_stages == 0:
        return {}

    h = planes["enthalpy"]
    R = jnp.array(
        [
            (h[i * 4 + 1] - h[i * 4 + 3]) / (h[i * 4] - h[i * 4 + 3])
            for i in range(number_of_stages)
        ]
    )
    return {"reaction": R}


def compute_overall_performance_componentwise(
    planes, boundary_conditions, reference_values, last_geom
):
    """Overall KPIs using last cascade’s geometry for blade-jet ratios."""
    angular_speed = boundary_conditions["omega"]
    v0 = reference_values["v0"]
    h_out_s = reference_values["h_out_s"]

    p = planes["pressure"]
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
    efficiency_ts_drop_losses = 1.0 - efficiency_ts - efficiency_ts_drop_kinetic
    power = mass_flow * (h0_in - h0_out)
    torque = power / angular_speed

    return {
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
        "specific_speed": angular_speed
        * (mass_flow / reference_values["d_out_s"]) ** 0.5
        / ((h0_in - h_out_s) ** 0.75),
        "blade_jet_ratio_mean": jnp.array(
            angular_speed * last_geom["radius_mean_out"] / v0
        ),
        "blade_jet_ratio_hub": (
            jnp.array(angular_speed * last_geom.get("radius_hub_out", jnp.nan) / v0)
            if "radius_hub_out" in last_geom
            else jnp.nan
        ),
        "blade_jet_ratio_tip": (
            jnp.array(angular_speed * last_geom.get("radius_tip_out", jnp.nan) / v0)
            if "radius_tip_out" in last_geom
            else jnp.nan
        ),
    }
