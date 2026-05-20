# flow_model.py — component-object dispatcher for performance_analysis

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import time
import jax
import equinox as eqx

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
# from .interspace_model import Interspace
from .interspace_model_update import Interspace

# ============================================================
# Helpers
# ============================================================


def _alpha_deg(a: jnp.ndarray) -> jnp.ndarray:
    """Ensure angle is in degrees (convert from radians if it looks like radians)."""
    return jnp.degrees(a) if jnp.abs(a) <= (jnp.pi * 1.01) else a


def _suffix_keys(d: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Append suffix (e.g. '_1', '_2') to all keys in dict."""
    return {f"{k}{suffix}": v for k, v in d.items()}


def _geom_get(geom: Any, key: str) -> Any:
    """Read geometry value from either a dict-based or object-based geometry."""
    if isinstance(geom, dict):
        if key in geom:
            return geom[key]
    elif hasattr(geom, key):
        return getattr(geom, key)
    raise KeyError(f"Geometry key '{key}' not available for {type(geom).__name__}.")


def _geom_inlet_area(geom: Any) -> Any:
    """
    Return inlet area from geometry.
    Priority:
      1) explicit `A_in` if available
      2) reconstructed annulus area from radius_mean_in and b_in
    """
    if isinstance(geom, dict):
        if "A_in" in geom:
            return geom["A_in"]
        if "radius_mean_in" in geom and "b_in" in geom:
            return 2.0 * jnp.pi * geom["radius_mean_in"] * geom["b_in"]
    else:
        if hasattr(geom, "A_in"):
            return getattr(geom, "A_in")
        if hasattr(geom, "radius_mean_in") and hasattr(geom, "b_in"):
            return 2.0 * jnp.pi * getattr(geom, "radius_mean_in") * getattr(geom, "b_in")

    raise KeyError(
        "Unable to infer inlet area: expected `A_in` or (`radius_mean_in`, `b_in`) "
        f"on geometry type {type(geom).__name__}."
    )


# ============================================================
# Public API: evaluate an already-instantiated component list
# ============================================================

# @eqx.filter_jit
def evaluate_turbomachine(
    variables: Dict[str, Any],          # solver vars (normalized)
    boundary_conditions: Dict[str, Any],
    comp_objects: List[Any],            # BladeRow / VanelessChannel / Interspace objects (in order)
    fluid: Any,
    reference_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Walk the component objects in order, evaluate each, chain inlet→outlet, and
    accumulate residuals and performance metrics.

    Notes
    -----
    - BladeRow: we unscale per-row unknowns from `variables` using row order
      tags _1, _2, ...
    - VanelessChannel: we call obj.evaluate() and map its outlet to the next
      inlet, but don't treat it as a cascade for stage/overall KPIs.
    - Interspace: simple algebraic mapping between two blade rows; also does
      not contribute planes or cascades, only updates the inlet for the next row.
    """

    # t_eval_start = time.perf_counter()

    # ---------- inlet from boundary conditions ----------
    h0_in = boundary_conditions["h0_in"]
    s_in = boundary_conditions["s_in"]
    alpha_in_deg = boundary_conditions["alpha_in"]

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
    for gi, obj in enumerate(comp_objects):
        # =====================================================
        # Blade row (axial or radial cascade)
        # =====================================================
        if isinstance(obj, BladeRow):

            # --- timing for this blade row ---
            # t_row0 = time.perf_counter()

            row_counter += 1
            geom = obj.geometry
            cascade_geoms.append(geom)

            # unscale row vars + choking vars for this row via BladeRow helper
            row_vars, choking_vars = BladeRow.unscale_row_vars_and_choking(
                variables=variables,
                index_1based=row_counter,
                reference_values=reference_values,
            )

            # t_row1 = time.perf_counter()

            # rotor rows rotate; stators do not
            is_rotor = "rotor" in str(obj.cascade_type).lower()
            omega_i = jnp.asarray(omega_bc if is_rotor else 0.0, dtype=jnp.float64)

            # BladeRow.evaluate returns a dict
            row_result = obj.evaluate(
                inlet_state=inlet,
                row_vars=row_vars,
                omega=omega_i,
                reference_values=reference_values,
                choking_vars=choking_vars,
            )

            # force JAX to finish compute for this row before timing
            exit_plane_for_timing = row_result["planes"][-1]
            jax.block_until_ready(exit_plane_for_timing["pressure"])

            # t_row2 = time.perf_counter()

            planes = row_result["planes"]
            cascade = row_result["cascade_summary"]
            res_i = row_result["residuals"]
            handoff = row_result["handoff"]  # kept for future use if needed

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
            # t_row3 = time.perf_counter()

            # print(
            #     f"[flow] BladeRow {row_counter}: "
            #     f"unscale={t_row1 - t_row0:.3e} s, "
            #     f"eval={t_row2 - t_row1:.3e} s, "
            #     f"post={t_row3 - t_row2:.3e} s, "
            #     f"total={t_row3 - t_row0:.3e} s"
            # )

            continue

        # =====================================================
        # Vaneless channel (ODE-based model)
        # =====================================================
        if isinstance(obj, VanelessChannel):

            # t_vc0 = time.perf_counter()

            # Map BladeRow-style inlet to channel operating conditions (static)
            # v_key = f"v_vaneless_{obj.name}"
            # if v_key in variables:
            #     v_mag = reference_values["v0"] * variables[v_key]
            # else:
            #     v_mag = inlet["v"]

            # v_cap = 0.98 * jnp.sqrt(jnp.maximum(2.0 * inlet["h0"], 1.0))
            # v_mag = jnp.clip(v_mag, 1.0, v_cap)

            v_mag = inlet["v"]
            alphaD = inlet["alpha"]
            h_in = inlet["h0"] - 0.5 * v_mag**2
            st = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, inlet["s"])
            p_in = st["p"]

            # Build a new OperatingConditions instance with updated inlet
            oc_cls = obj.operating_conditions.__class__
            oc_new = oc_cls(
                p_in=jnp.asarray(p_in),
                h_in=jnp.asarray(h_in),
                v_in=jnp.asarray(v_mag),
                alpha_in=jnp.asarray(alphaD),
            )

            # Replace operating_conditions in the channel object
            obj = eqx.tree_at(lambda c: c.operating_conditions, obj, oc_new)

            # Solve the channel; returns dict-of-arrays
            result = obj.evaluate()

            # # --- vaneless mass continuity residual ---
            # A_in = 2.0 * jnp.pi * obj.geometry.radius_mean_in * obj.geometry.b_in
            # v_m_in = v_mag * jnp.cos(jnp.deg2rad(alphaD))
            # rho_in = st["d"]
            # m_dot_in = rho_in * v_m_in * A_in

            # if "mass_flow" in result:
            #     m_dot_out = result["mass_flow"][-1]
            # else:
            #     m_dot_out = result["d"][-1] * result["v_m"][-1] * result["A"][-1]

            # mass_error_exit = m_dot_in - m_dot_out
            # residuals[f"mass_error_exit_{obj.name}"] = (
            #     mass_error_exit / reference_values["mass_flow_ref"]
            # )

            # force JAX to finish this solve before timing
            jax.block_until_ready(result["p"][-1])

            # t_vc1 = time.perf_counter()

            # Update inlet to next component
            v_m_out = result["v_m"][-1]
            v_t_out = result["v_t"][-1]
            v_out = jnp.sqrt(v_m_out**2 + v_t_out**2)
            alpha = jnp.degrees(jnp.arctan2(v_t_out, v_m_out))
            h0_out = result["h0"][-1]
            s_out = result["s"][-1]

            inlet = {"h0": h0_out, "s": s_out, "alpha": alpha, "v": v_out}

            # t_vc2 = time.perf_counter()

            # print(
            #     f"[flow] VanelessChannel {gi + 1}: "
            #     f"eval={t_vc1 - t_vc0:.3e} s, "
            #     f"post={t_vc2 - t_vc1:.3e} s, "
            #     f"total={t_vc2 - t_vc0:.3e} s"
            # )

            continue

        # =====================================================
        # Interspace (algebraic mapping between blade rows)
        # =====================================================
        if isinstance(obj, Interspace):

            v_key = f"v_out_is_{obj.name}"
            v_out_is = reference_values["v0"] * variables[v_key]
            v_cap = 0.98 * jnp.sqrt(jnp.maximum(2.0 * inlet["h0"], 1.0))
            v_out_is = jnp.clip(v_out_is, 1.0, v_cap)

            # t_is0 = time.perf_counter()

            # We require an upstream BladeRow already evaluated.
            if not planes_seq or not cascade_geoms:
                raise RuntimeError(
                    "Interspace must follow at least one BladeRow; "
                    "no previous cascade exit state available."
                )

            # # We also require that the NEXT component is a BladeRow
            # if gi + 1 >= len(comp_objects) or not isinstance(comp_objects[gi + 1], BladeRow):
            #     raise RuntimeError(
            #         "Interspace must be followed by a BladeRow to define the next inlet geometry."
            #     )

            prev_exit = planes_seq[-1]
            prev_geom = cascade_geoms[-1]

            if gi + 1 >= len(comp_objects):
                raise RuntimeError(
                    f"Interspace '{obj.name}' is the last component, so downstream inlet geometry is undefined."
                )

            next_comp = comp_objects[gi + 1]
            if not hasattr(next_comp, "geometry"):
                raise RuntimeError(
                    f"Interspace '{obj.name}' is followed by '{type(next_comp).__name__}', "
                    "which does not expose a geometry object."
                )

            next_geom = next_comp.geometry
            radius_inlet = _geom_get(next_geom, "radius_mean_in")
            area_inlet = _geom_inlet_area(next_geom)

            alpha_target_deg = None
            # if obj.name == "interspace_8" and getattr(next_comp, "name", "") == "stator_5":
            #     alpha_target_deg = jnp.asarray(
            #         _geom_get(next_geom, "leading_edge_angle"),
            #         dtype=jnp.float64,
            #     )

            h0_in_new, s_in_new, alpha_in_new, v_in_new, mass_res = obj.evaluate(
                h0_exit=prev_exit["enthalpy0"],
                v_m_exit=prev_exit["v_m"],
                v_t_exit=prev_exit["v_t"],
                rho_exit=prev_exit["density"],
                radius_exit=prev_geom["radius_mean_out"],
                area_exit=prev_geom["A_out"],
                blockage_exit=prev_exit["blockage"],
                radius_inlet=radius_inlet,
                area_inlet=area_inlet,
                s_exit=prev_exit["entropy"],
                mass_flow_exit=prev_exit["mass_flow"],
                v_out_is=v_out_is,
                mass_flow_ref=reference_values["mass_flow_ref"],
                alpha_target_deg=alpha_target_deg,   
                
                )

            inlet = {
                "h0": h0_in_new,
                "s": s_in_new,
                "alpha": alpha_in_new,
                "v": v_in_new,
            }

            residuals[f"mass_error_exit_{obj.name}"] = mass_res

            # t_is1 = time.perf_counter()
            # print(
            #     f"[flow] Interspace {gi + 1}: "
            #     f"total={t_is1 - t_is0:.3e} s"
            # )

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

    # --- timing for final aggregation & KPIs ---
    # t_agg0 = time.perf_counter()
    # print(planes_seq)
    planes = tf.combine_to_dict_of_arrays(planes_seq)
    cascades = tf.combine_to_dict_of_arrays(cascades_seq)

    # outlet pressure residual uses last cascade exit static pressure
    p_calc = planes_seq[-1]["pressure"]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # ## Debug
    # print("[flow] cascade component types:", [g["cascade_type"] for g in cascade_geoms])
    # ## Debug

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

    # force JAX to finish KPI computations before timing
    jax.block_until_ready(overall["power"])

    # t_agg1 = time.perf_counter()
    # print(f"[flow] aggregation+KPIs took {t_agg1 - t_agg0:.3e} s")

    # t_eval_end = time.perf_counter()
    # print(f"[flow] evaluate_turbomachine total {t_eval_end - t_eval_start:.3e} s")

    # Component-type tags in output, distinguishing interspace and vaneless
    def _component_type_tag(o: Any) -> str:
        if isinstance(o, BladeRow):
            return o.cascade_type
        if isinstance(o, VanelessChannel):
            return "vaneless_channel"
        if isinstance(o, Interspace):
            return "interspace"
        return type(o).__name__

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
        "component_types": [_component_type_tag(o) for o in comp_objects],
        # cascades-only list of geometry dicts (used in KPIs / reporting)
        "geometry_components": cascade_geoms,
    }



# ============================================================
# Thin re-export for compatibility where needed
# ============================================================


def evaluate_cascade_throat(*args, **kwargs):
    return _blade_throat(*args, **kwargs)


# ============================================================
# Stage & overall KPIs
# ============================================================

# @eqx.filter_jit
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

    # # DEBUG BLOCK START
    # h0 = planes["enthalpy0"]
    # p  = planes["pressure"]
    # v  = planes["v"]
    # Ma = planes["Ma_rel"]

    # # Optional fields (present in your key list)
    # has_vt = "v_t" in planes
    # has_u  = "blade_speed" in planes

    # v_t = planes["v_t"] if has_vt else None
    # u   = planes["blade_speed"] if has_u else None

    # print("=== Stage performance debug ===")
    # print("component_types:", component_types)
    # print("number_of_stages:", number_of_stages)

    # for i in range(number_of_stages):
    #     i0 = i * 4
    #     idx_st_in  = i0 + 0
    #     idx_st_out = i0 + 1
    #     idx_ro_in  = i0 + 2
    #     idx_ro_out = i0 + 3

    #     h_st_in  = h[idx_st_in]
    #     h_st_out = h[idx_st_out]
    #     h_ro_in  = h[idx_ro_in]
    #     h_ro_out = h[idx_ro_out]

    #     h0_st_in  = h0[idx_st_in]
    #     h0_st_out = h0[idx_st_out]
    #     h0_ro_in  = h0[idx_ro_in]
    #     h0_ro_out = h0[idx_ro_out]

    #     p_st_in  = p[idx_st_in]
    #     p_st_out = p[idx_st_out]
    #     p_ro_in  = p[idx_ro_in]
    #     p_ro_out = p[idx_ro_out]

    #     v_st_in  = v[idx_st_in]
    #     v_st_out = v[idx_st_out]
    #     v_ro_in  = v[idx_ro_in]
    #     v_ro_out = v[idx_ro_out]

    #     Ma_st_in  = Ma[idx_st_in]
    #     Ma_st_out = Ma[idx_st_out]
    #     Ma_ro_in  = Ma[idx_ro_in]
    #     Ma_ro_out = Ma[idx_ro_out]   

    #     print(f" Stage {i+1}:")
    #     print("   h  (st_in, st_out, ro_in, ro_out) =",
    #           h_st_in, h_st_out, h_ro_in, h_ro_out)
    #     print("   h0 (st_in, st_out, ro_in, ro_out) =",
    #           h0_st_in, h0_st_out, h0_ro_in, h0_ro_out)
    #     print("   p  (st_in, st_out, ro_in, ro_out) =",
    #           p_st_in, p_st_out, p_ro_in, p_ro_out)
    #     print("   v  (st_in, st_out, ro_in, ro_out) =",
    #           v_st_in, v_st_out, v_ro_in, v_ro_out)
    #     print("   Ma  (st_in, st_out, ro_in, ro_out) =",
    #           Ma_st_in, Ma_st_out, Ma_ro_in, Ma_ro_out)

    #     if has_vt and has_u:
    #         vt_st_in  = v_t[idx_st_in]
    #         vt_st_out = v_t[idx_st_out]
    #         vt_ro_in  = v_t[idx_ro_in]
    #         vt_ro_out = v_t[idx_ro_out]

    #         u_st_in  = u[idx_st_in]
    #         u_st_out = u[idx_st_out]
    #         u_ro_in  = u[idx_ro_in]
    #         u_ro_out = u[idx_ro_out]

    #         print("   v_t (st_in, st_out, ro_in, ro_out) =",
    #               vt_st_in, vt_st_out, vt_ro_in, vt_ro_out)
    #         print("   u   (st_in, st_out, ro_in, ro_out) =",
    #               u_st_in, u_st_out, u_ro_in, u_ro_out)

    #         # Euler turbine work vs. total enthalpy change in rotor
    #         dh0_rotor = h0_ro_out - h0_ro_in
    #         euler_rotor = u_ro_in * vt_ro_in - u_ro_out * vt_ro_out
    #         print("   Δh0_rotor =", dh0_rotor,
    #               "; Euler (uVθ in - uVθ out) =", euler_rotor)

    #     # Same reaction definition as before
    #     R_i = (h_ro_in - h_ro_out) / (h_st_in - h_ro_out)
    #     print("   R_i (debug) =", R_i)

    # beta = planes["beta"]
    # print("Rotor beta_in, beta_out =", beta[2], beta[3])
    # print("Rotor incidence =", beta[2] - 30.0)

    # print("=== End stage debug ===")
    # # DEBUG BLOCK END

    R = jnp.array(
        [
            (h[i * 4 + 1] - h[i * 4 + 3]) / (h[i * 4] - h[i * 4 + 3])
            for i in range(number_of_stages)
        ]
    )
    return {"reaction": R}

# @eqx.filter_jit
def compute_overall_performance_componentwise(
    planes, boundary_conditions, reference_values, last_geom
):
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
