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
    geom_list: List[Dict[str, Any]],                 # geometry for *cascades only* (in cascade order)
    fluid: Any,
    reference_values: Dict[str, Any],
    components: List[Dict[str, Any]] | None = None,  # full YAML components (mixed types)
    model_options_global: Dict[str, Any] | None = None,
    benner_requires_throat: bool = True,
):
    """
    Iterate the full component list (axial_cascade and vaneless_channel).
    - For axial_cascade: build a BladeRow from geom_list[cascade_idx] and evaluate
    - For vaneless_channel: solve VanelessChannel using its own geometry + options and
      update the inlet state, but do not create planes (keeps 4-plane-per-stage pattern)
    """
    assert components is not None, "components list (mixed types) must be provided"

    # Inlet (from BCs)
    h0_in    = boundary_conditions["h0_in"]
    s_in     = boundary_conditions["s_in"]
    alpha_in = boundary_conditions["alpha_in"]   
    omega_bc = boundary_conditions["omega"]

    # Reference scales
    v0 = reference_values["v0"]

    planes_seq:   List[Dict[str, Any]] = []
    cascades_seq: List[Dict[str, Any]] = []
    residuals:    Dict[str, Any] = {}

    # inlet absolute speed (scaled → unscaled)
    v_in = variables["v_in"] * v0

    # Prepare initial inlet payload
    inlet = {"h0": h0_in, "s": s_in, "alpha": alpha_in, "v": v_in}

    # --- helper to normalize alpha to degrees for the channel (if needed)
    def _alpha_deg(a):
        # if looks like radians (|a| <= ~pi), convert to degrees
        return jnp.degrees(a) if jnp.abs(a) <= (jnp.pi * 1.01) else a

    # Iterate mixed components; track a *cascade* index separately
    cascade_idx = 0
    cascade_types: List[str] = []
    cascade_geoms_for_rows: List[BladeRow] = []

    for comp_i, comp in enumerate(components):
        ctype = str(comp.get("component_type", "")).lower()
        name  = comp.get("name", f"component_{comp_i+1}")

        # ---------- Axial cascade ----------
        if ctype == "axial_cascade":
            # geometry for this cascade comes from geom_list[cascade_idx]
            if cascade_idx >= len(geom_list):
                raise IndexError("Provided 'geom_list' has fewer cascade geometries than axial_cascade components.")

            geom_cascade = geom_list[cascade_idx]
            cascade_types.append(str(geom_cascade["cascade_type"]).lower())

            # Build BladeRow using per-component model_options
            per_opts = comp.get("model_options", {}) or {}
            config_row = {
                "name": name,
                "cascade_type": geom_cascade["cascade_type"],
                "geometry": geom_cascade,
                "model_options": per_opts,
            }
            row = BladeRow.from_dict(config_row, fluid=fluid, model_options_global=model_options_global)

            # Unscale decision variables *by cascade index* (tags _1, _2, ...)
            tag_idx = cascade_idx + 1
            row_vars, choking_vars = _extract_row_vars_and_choking(variables, tag_idx, reference_values)

            # Rotor rows rotate; stators do not
            is_rotor = ("rotor" in row.cascade_type)
            omega_i  = jnp.array(omega_bc if is_rotor else 0.0)

            # Evaluate row
            planes, cascade, res_i, handoff = row.evaluate(
                inlet_state=inlet,
                row_vars=row_vars,
                omega=omega_i,
                reference_values=reference_values,
                choking_vars=choking_vars,
            )

            # Collect
            planes_seq += planes
            cascades_seq.append(cascade)
            residuals.update(_suffix_keys(res_i, f"_{tag_idx}"))

            # Next inlet = last plane state (absolute)
            exit_plane = planes[-1]
            inlet = {
                "h0":    exit_plane["enthalpy0"],
                "s":     exit_plane["entropy0"],
                "alpha": exit_plane["alpha"],   # (BladeRow uses degrees; keep as-is)
                "v":     exit_plane["v"],
            }

            cascade_idx += 1
            continue

        # ---------- Vaneless channel ----------
        if ctype == "vaneless_channel":
            # Build a config for VanelessChannel using the component's own geometry/options
            # + our current inlet state mapped to *static* OC
            geometry_cfg = comp.get("geometry", {})
            model_opts   = (comp.get("model_options") or
                            (model_options_global or {}).get("vaneless", {}).get("model_options") or {})
            solver_opts  = (comp.get("solver_options") or
                            (model_options_global or {}).get("vaneless", {}).get("solver_options") or {})

            # Convert inlet to static: h_in, p_in; v_in magnitude + alpha_in (deg)
            v_mag  = inlet["v"]
            alphaD = _alpha_deg(inlet["alpha"])
            h_in   = inlet["h0"] - 0.5 * v_mag**2
            st     = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, inlet["s"])
            p_in   = st["p"]

            # print(p_in, h_in, v_mag, alphaD)

            config_ch = {
                "name": name,
                "geometry": geometry_cfg,
                "model_options": (
                    model_opts if model_opts else
                    {"friction_model": {"type": "aungier", "roughness": 1.0e-6, "Re_transition": 2300.0, "Re_width": 500.0},
                     "heat_model": {"type": "adiabatic"}}
                ),
                "solver_options": (
                    solver_opts if solver_opts else
                    {"solver_name": "Dopri5", "adjoint_name": "DirectAdjoint",
                     "rtol": 1e-6, "atol": 1e-6, "n_points": 50, "max_steps": 200, "throw": True}
                ),
                "operating_conditions": {
                    "p_in": p_in,
                    "h_in": h_in,
                    "v_in": v_mag,
                    "alpha_in": alphaD,   # degrees, as required by VanelessChannel
                },
            }

            channel  = VanelessChannel.from_dict(config_ch, fluid)
            sol      = channel.evaluate()

            # Map channel outlet → updated inlet (no planes added)
            v_m_out = sol["v_m"][-1]
            v_t_out = sol["v_t"][-1]
            v_out   = jnp.sqrt(v_m_out**2 + v_t_out**2)
            alpha   = math.arctand(v_t_out / v_m_out)  # returns degrees in your math helpers

            # h0/s: prefer direct if present; else reconstruct
            if "h0" in sol:
                h0_out = sol["h0"][-1]
            elif "h" in sol:
                h0_out = sol["h"][-1] + 0.5 * v_out**2
            else:
                h0_out = inlet["h0"]

            if "s" in sol:
                s_out = sol["s"][-1]
            elif ("p" in sol) and ("T" in sol):
                st_out = fluid.get_state(jxp.PT_INPUTS, sol["p"][-1], sol["T"][-1])
                s_out  = st_out["s"]
            else:
                s_out = inlet["s"]

            inlet = {"h0": h0_out, "s": s_out, "alpha": alpha, "v": v_out}
            continue

        # ---------- Unknown component ----------
        raise ValueError(f"Unsupported component_type: {ctype}")

    # Collect arrays (unchanged)
    planes   = tf.combine_to_dict_of_arrays(planes_seq)
    cascades = tf.combine_to_dict_of_arrays(cascades_seq)

    # Outlet pressure residual uses the *last cascade* exit static pressure
    if not planes_seq:
        raise RuntimeError("No cascades evaluated; cannot compute outlet residuals/overall KPIs.")
    p_calc = planes_seq[-1]["pressure"]
    # p_calc = planes_seq[-1]["p"]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # Stage & overall KPIs (unchanged; pass *cascade* types only)
    stage   = compute_stage_performance_componentwise(planes, [c["cascade_type"] for c in geom_list])
    overall = compute_overall_performance_componentwise(planes, boundary_conditions, reference_values, geom_list[-1])

    return {
        "planes": planes,
        "cascades": cascades,
        "stage": stage,
        "overall": overall,
        "residuals": residuals,
        "independent_variables": variables,
        "component_names": [c.get("name", f"component_{i+1}") for i, c in enumerate(components)],
        "component_types": [str(c.get("component_type", "")).lower() for c in components],
        "geometry_components": geom_list,  # cascades-only geometry list
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

# def _evaluate_vaneless_interspace(
#     *,
#     fluid,
#     handoff,            # dict from BladeRow.evaluate (exit of current row)
#     row_out_geom,       # current row geometry dict (outlet side)
#     row_in_geom_next,   # next row geometry dict (inlet side)
#     boundary_conditions,
#     interspace_options=None,  # optional dict of friction/heat/solver opts
# ):
#     """
#     Solve the vaneless interspace with VanelessChannel and return (h0_in_next, s_in_next, alpha_in_next, v_in_next).
#     Expects VanelessChannel to accept:
#       - geometry: r_in, r_out, A_in, A_out   (b_in/out will be computed)
#       - operating_conditions: p_in, h0_in, v_m_in, v_t_in
#     """

#     # -------- Geometry (effective area at inlet includes blockage) --------
#     A_in_eff  = row_out_geom["A_out"] * (1.0 - handoff["blockage_out"])
#     r_in      = row_out_geom["radius_mean_out"]
#     r_out     = row_in_geom_next["radius_mean_in"]
#     A_out_tgt = row_in_geom_next["A_in"]

#     geometry_cfg = {
#         "r_in":  r_in,
#         "r_out": r_out,
#         "A_in":  A_in_eff,
#         "A_out": A_out_tgt,
#         # Optional: if your channel wants a specific axial span/angles, you can add:
#         # "z_in": 0.0, "z_out": 0.05, "phi_in": 0.0, "phi_out": 0.0, "td_in": 0.01, "td_out": 0.01
#         # but the class now fills safe defaults.
#     }

#     # -------- Operating conditions at interspace inlet (absolute frame) --------
#     v_m_in = handoff["v_m_out"]
#     v_t_in = handoff["v_t_out"]
#     h0_in  = handoff["enthalpy0_out"]

#     # Prefer a direct exit static pressure from the blade row if present;
#     # otherwise reconstruct p_in from (h, s) if available; else fall back to BC s_in.
#     p_in = handoff.get("p_out", None)
#     if p_in is None:
#         # reconstruct static enthalpy at row exit
#         v_mag_exit = jnp.sqrt(v_m_in**2 + v_t_in**2)
#         h_exit = h0_in - 0.5 * v_mag_exit**2
#         s_exit = handoff.get("s_out", handoff.get("entropy_out", boundary_conditions["s_in"]))
#         state_exit = fluid.get_state(jxp.HmassSmass_INPUTS, h_exit, s_exit)
#         p_in = state_exit["p"]

#     operating_conditions_cfg = {
#         "p_in":   p_in,
#         "h0_in":  h0_in,
#         "v_m_in": v_m_in,
#         "v_t_in": v_t_in,
#         # omega = 0 in a vaneless, non-rotating interspace
#         "omega":  0.0,
#     }

#     # -------- Options (friction/heat + solver) --------
#     interspace_options = interspace_options or {}
#     model_options_cfg  = interspace_options.get("model_options", {
#         "friction_model": {"type": "aungier", "roughness": 1.0e-6, "Re_transition": 2300.0, "Re_width": 500.0},
#         "heat_model":     {"type": "adiabatic"},
#     })
#     solver_options_cfg = interspace_options.get("solver_options", {
#         "solver_name": "Dopri5",
#         "adjoint_name": "DirectAdjoint",
#         "rtol": 1.0e-6,
#         "atol": 1.0e-6,
#         "n_points": 50,
#         "max_steps": 200,
#         "throw": True,
#     })

#     config = {
#         "name": "interspace",
#         "geometry": geometry_cfg,
#         "model_options": model_options_cfg,
#         "solver_options": solver_options_cfg,
#         "operating_conditions": operating_conditions_cfg,
#     }

#     # -------- Solve channel --------
#     channel  = VanelessChannel.from_dict(config, fluid)
#     solution = channel.evaluate()  # dict of arrays along meridional coordinate

#     # -------- Map channel outlet → next-row inlet --------
#     # Use the last point in the arrays returned by the solver
#     v_m_out = solution["v_m"][-1]
#     v_t_out = solution["v_t"][-1]
#     v_out   = jnp.sqrt(v_m_out**2 + v_t_out**2)
#     alpha   = math.arctand(v_t_out / v_m_out)

#     # Prefer direct s/h0 if present, else reconstruct
#     s_out = solution.get("s", None)
#     if s_out is not None:
#         s_out = s_out[-1]
#     else:
#         # reconstruct from (p,T) if available
#         if ("p" in solution) and ("T" in solution):
#             st = fluid.get_state(jxp.PT_INPUTS, solution["p"][-1], solution["T"][-1])
#             s_out = st["s"]
#         else:
#             s_out = boundary_conditions["s_in"]

#     if "h0" in solution:
#         h0_out = solution["h0"][-1]
#     else:
#         if "h" in solution:
#             h0_out = solution["h"][-1] + 0.5 * v_out**2
#         else:
#             # conservative fallback
#             h0_out = h0_in

#     return h0_out, s_out, alpha, v_out


# ============================================================
# Interspace mapping (unchanged)
# ============================================================

# def evaluate_cascade_interspace(
#     h0_exit,
#     v_m_exit,
#     v_t_exit,
#     rho_exit,
#     radius_exit,
#     area_exit,
#     blockage_exit,
#     radius_inlet,
#     area_inlet,
#     fluid,
# ):
#     """Propagate exit of i to inlet of i+1 (interspace)."""
#     h0_in = h0_exit
#     v_t_in = v_t_exit * radius_exit / radius_inlet
#     v_m_in = v_m_exit * area_exit / area_inlet * (1 - blockage_exit)
#     v_in = jnp.sqrt(v_t_in**2 + v_m_in**2)
#     alpha_in = math.arctand(v_t_in / v_m_in)

#     h_in = h0_in - 0.5 * v_in**2
#     rho_in = rho_exit
#     stagnation_properties = fluid.get_state(jxp.DmassHmass_INPUTS, rho_in, h_in)
#     s_in = stagnation_properties["s"]

#     return h0_in, s_in, alpha_in, v_in


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
    # p  = planes["p"]
    p0 = planes["pressure0"]
    # p0 = planes["p0"]
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
