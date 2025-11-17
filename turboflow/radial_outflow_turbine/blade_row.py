# blade_row.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from scipy import optimize
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxprop as jxp

# Project-local deps (same as your flow model)
from .. import math
from .. import utilities as utils
from . import loss_model as lm
from . import choking_criterion as cm


from . import geometry_model_axial, geometry_model_radial


# ============================================================
# Constants
# ============================================================

BLOCKAGE_MODELS = ["flat_plate_turbulent"]


# ============================================================
# Tiny helpers (pure utilities)
# ============================================================


def _to_jax_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert plain numerics to jnp.array; leave others untouched."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out[k] = jnp.array(v, dtype=jnp.float64)
        else:
            out[k] = v
    return out


def _normalize_model_options(
    global_opts: Dict[str, Any] | None,
    comp_opts: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Merge and apply defaults. Raises if 'loss_model' is missing."""
    mo: Dict[str, Any] = {}
    if global_opts:
        mo.update(global_opts)
    if comp_opts:
        mo.update(comp_opts)

    if "loss_model" not in mo:
        raise ValueError("model_options must include 'loss_model'.")

    mo.setdefault("loss_coefficient", "stagnation_pressure")
    mo.setdefault("deviation_model", "aungier")
    mo.setdefault("choking_criterion", "critical_mach_number")
    mo.setdefault("blockage_model", None)
    mo.setdefault("inlet_displacement_thickness_height_ratio", 0.011)
    return mo


# ============================================================
# Local definitions (kept here for clarity; no imports)
# ============================================================


def _validate_geometry_component(name: str, g: Dict[str, Any], require_throat: bool):
    required = [
        "radius_mean_in",
        "radius_mean_out",
        "A_in",
        "A_out",
        "chord",
        "stagger_angle",
        "pitch",
        "height",
        "cascade_type",
        "leading_edge_angle",
    ]
    missing = [k for k in required if k not in g]
    if missing:
        raise ValueError(f"[{name}] Missing required geometry keys: {missing}")

    if require_throat:
        throat_req = ["A_throat", "leading_edge_wedge_angle", "leading_edge_diameter"]
        throat_missing = [k for k in throat_req if k not in g]
        if throat_missing:
            raise ValueError(
                f"[{name}] Benner incidence requires: {throat_req}. Missing: {throat_missing}"
            )


# @jax.jit
def evaluate_velocity_triangle_in(blade_speed, v, alpha):
    # Promote to JAX arrays (float64 for consistency with the rest of the model)
    blade_speed = jnp.asarray(blade_speed, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)

    # Absolute (v) → tangential/meridional components
    v_t = v * math.sind(alpha)
    v_m = v * math.cosd(alpha)

    # Relative components (rotating frame)
    w_t = v_t - blade_speed
    w_m = v_m

    # Magnitude (hypot is stable and differentiable)
    w = jnp.sqrt(w_t**2 + w_m**2)
    # w = jnp.hypot(w_t, w_m)

    # Flow angle in degrees; atan2 avoids the explicit division
    # Note: jnp.arctan2 takes (y, x). Here beta = atan2(w_t, w_m) in degrees.
    beta = jnp.degrees(jnp.arctan2(w_t, w_m))

    return {
        "blade_speed": blade_speed,
        "v": v,
        "v_m": v_m,
        "v_t": v_t,
        "alpha": alpha,
        "w": w,
        "w_m": w_m,
        "w_t": w_t,
        "beta": beta,
    }


# @jax.jit
def evaluate_velocity_triangle_out(blade_speed, w, beta):
    # Promote to JAX arrays (float64 for consistency)
    blade_speed = jnp.asarray(blade_speed, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)

    # Relative components from magnitude/angle (beta in degrees)
    w_t = w * math.sind(beta)
    w_m = w * math.cosd(beta)

    # Absolute components
    v_t = w_t + blade_speed
    v_m = w_m

    # Magnitudes (hypot is stable and differentiable)
    v = jnp.sqrt(v_t**2 + v_m**2)
    # v = jnp.hypot(v_t, v_m)
    # Angle α in degrees; atan2 handles quadrants and avoids division
    alpha = jnp.degrees(jnp.arctan2(v_t, v_m))

    return {
        "blade_speed": blade_speed,
        "v": v,
        "v_m": v_m,
        "v_t": v_t,
        "alpha": alpha,
        "w": w,
        "w_m": w_m,
        "w_t": w_t,
        "beta": beta,
    }


# def compute_blockage_boundary_layer(blockage_model, Re, chord, opening):
#     """Boundary-layer blockage. If `opening` is None/inf → 0."""
#     if opening is None or (jnp.isinf(opening)):
#         return 0.0
#     opening_safe = jnp.maximum(jnp.asarray(opening, dtype=jnp.float64), 1e-9)

#     if blockage_model == BLOCKAGE_MODELS[0]:
#         displacement_thickness = 0.048 / Re ** (1 / 5) * 0.9 * chord
#         blockage_factor = 2 * displacement_thickness / opening_safe
#     elif isinstance(blockage_model, (float, int)) and 0 <= blockage_model <= 1:
#         blockage_factor = float(blockage_model)
#     elif blockage_model is None:
#         blockage_factor = 0.0
#     else:
#         raise ValueError(
#             f"Invalid throat blockage option: '{blockage_model}'. "
#             "Valid: 'flat_plate_turbulent', numeric in [0,1], or None."
#         )
#     return blockage_factor


def compute_blockage_boundary_layer(blockage_model, Re, chord, opening):
    """Boundary-layer blockage. If opening is None/inf → 0 (JIT-safe)."""
    # opening may be None (static) or a traced scalar/array
    if opening is None:
        return jnp.array(0.0, dtype=jnp.float64)

    opening_arr = jnp.asarray(opening, dtype=jnp.float64)
    opening_safe = jnp.maximum(opening_arr, 1e-9)

    # Base blockage_factor depending on model (blockage_model is static under filter_jit)
    if blockage_model == BLOCKAGE_MODELS[0]:
        displacement_thickness = 0.048 * chord * 0.9 / (Re ** (1.0 / 5.0))
        blockage_factor = 2.0 * displacement_thickness / opening_safe
    elif (
        isinstance(blockage_model, (float, int)) and 0.0 <= float(blockage_model) <= 1.0
    ):
        blockage_factor = jnp.asarray(blockage_model, dtype=jnp.float64)
    elif blockage_model is None:
        blockage_factor = jnp.array(0.0, dtype=jnp.float64)
    else:
        # This branch is evaluated outside tracing since blockage_model is static.
        raise ValueError(
            f"Invalid throat blockage option: '{blockage_model}'. "
            "Valid: 'flat_plate_turbulent', numeric in [0,1], or None."
        )

    # If opening is +inf → 0 blockage (handled with JAX control flow)
    blockage_factor = jnp.where(jnp.isinf(opening_arr), 0.0, blockage_factor)
    return blockage_factor


# @eqx.filter_jit
def evaluate_cascade_inlet(cascade_inlet_input, fluid, geometry, angular_speed):
    # ---- inputs → float64 JAX arrays ----
    h0 = jnp.asarray(cascade_inlet_input["h0"], dtype=jnp.float64)
    s = jnp.asarray(cascade_inlet_input["s"], dtype=jnp.float64)
    v = jnp.asarray(cascade_inlet_input["v"], dtype=jnp.float64)
    alpha = jnp.asarray(cascade_inlet_input["alpha"], dtype=jnp.float64)

    radius = jnp.asarray(geometry["radius_mean_in"], dtype=jnp.float64)
    chord = jnp.asarray(geometry["chord"], dtype=jnp.float64)
    area = jnp.asarray(geometry["A_in"], dtype=jnp.float64)
    omega = jnp.asarray(angular_speed, dtype=jnp.float64)

    # ---- velocity triangle (abs → rel) ----
    blade_speed = radius * omega
    vt = evaluate_velocity_triangle_in(blade_speed, v, alpha)
    w = vt["w"]
    w_m = vt["w_m"]

    # ---- thermodynamic states ----
    h = h0 - 0.5 * v**2
    sp = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)  # static
    rho = sp["d"]
    mu = sp["mu"]
    a = sp["a"]

    sg0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)  # stagnation (abs)
    sg0 = utils.add_string_to_keys(sg0, "0")

    h0_rel = h + 0.5 * w**2
    sg0r = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)  # stagnation (rel)
    sg0r = utils.add_string_to_keys(sg0r, "0_rel")

    # ---- groups & flow quantities ----
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    m_dot = rho * w_m * area
    rothalpy = h0_rel - 0.5 * blade_speed**2

    zero = jnp.array(0.0, dtype=jnp.float64)
    loss_dict = {
        "loss_error": zero,
        "loss_profile": zero,
        "loss_incidence": zero,
        "loss_trailing": zero,
        "loss_secondary": zero,
        "loss_clearance": zero,
        "loss_total": zero,
    }

    # ---- assemble outlet ----
    plane = {
        **vt,
        **sp,
        **sg0,
        **sg0r,
        **loss_dict,
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": m_dot,
        "rothalpy": rothalpy,
        "blockage": zero,
        "h_is": sp["h"],
    }
    return plane


# def evaluate_cascade_inlet(cascade_inlet_input, fluid, geometry, angular_speed):
#     """Inlet plane: velocity triangles, thermo state, Re/Ma, mass flow."""
#     h0 = cascade_inlet_input["h0"]
#     s  = cascade_inlet_input["s"]
#     v  = cascade_inlet_input["v"]
#     alpha = cascade_inlet_input["alpha"]

#     radius = geometry["radius_mean_in"]
#     chord  = geometry["chord"]
#     area   = geometry["A_in"]

#     blade_speed = radius * angular_speed
#     velocity_triangle = evaluate_velocity_triangle_in(blade_speed, v, alpha)
#     w  = velocity_triangle["w"]
#     w_m = velocity_triangle["w_m"]

#     h = h0 - 0.5 * v**2
#     static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

#     rho = static_properties["d"]
#     mu  = static_properties["mu"]
#     a   = static_properties["a"]

#     stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
#     stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

#     h0_rel = h + 0.5 * w**2
#     relative_stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)
#     relative_stagnation_properties = utils.add_string_to_keys(relative_stagnation_properties, "0_rel")

#     Ma     = v / a
#     Ma_rel = w / a
#     Re     = rho * w * chord / mu
#     m      = rho * w_m * area
#     rothalpy = h0_rel - 0.5 * blade_speed**2

#     loss_dict = {
#         "loss_error": 0.0,
#         "loss_profile": 0.0,
#         "loss_incidence": 0.0,
#         "loss_trailing": 0.0,
#         "loss_secondary": 0.0,
#         "loss_clearance": 0.0,
#         "loss_total": 0.0,
#     }

#     plane = {
#         **velocity_triangle,
#         **static_properties,
#         **stagnation_properties,
#         **relative_stagnation_properties,
#         **loss_dict,
#         "Ma": Ma,
#         "Ma_rel": Ma_rel,
#         "Re": Re,
#         "mass_flow": m,
#         "rothalpy": rothalpy,
#         "blockage": 0.0,
#         "h_is": static_properties["h"],
#     }
#     return plane


# @eqx.filter_jit
def evaluate_cascade_exit(
    cascade_exit_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    # ---- inputs → float64 JAX arrays ----
    w = jnp.asarray(cascade_exit_input["w"], dtype=jnp.float64)
    beta = jnp.asarray(cascade_exit_input["beta"], dtype=jnp.float64)
    s = jnp.asarray(cascade_exit_input["s"], dtype=jnp.float64)
    rothalpy = jnp.asarray(cascade_exit_input["rothalpy"], dtype=jnp.float64)

    chord = jnp.asarray(geometry["chord"], dtype=jnp.float64)
    area = jnp.asarray(geometry["A_out"], dtype=jnp.float64)
    radius = jnp.asarray(geometry["radius_mean_out"], dtype=jnp.float64)

    omega = jnp.asarray(angular_speed, dtype=jnp.float64)

    # ---- kinematics (relative → absolute) ----
    blade_speed = omega * radius
    vt = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = vt["v"]
    w_m = vt["w_m"]

    # ---- thermodynamics ----
    # rothalpy = h + 0.5 w^2 - 0.5 U^2  →  h = rothalpy + 0.5 U^2 - 0.5 w^2
    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    sp = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)  # static properties

    rho = sp["d"]
    mu = sp["mu"]
    a = sp["a"]

    h0 = h + 0.5 * v**2
    sg0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)  # abs. stagnation
    sg0 = utils.add_string_to_keys(sg0, "0")

    h0_rel = h + 0.5 * w**2
    sg0r = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)  # rel. stagnation
    sg0r = utils.add_string_to_keys(sg0r, "0_rel")

    # nondim groups
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    rothalpy_out = h0_rel - 0.5 * blade_speed**2  # (matches inlet def)

    # isentropic references (relative and static)
    rs_is_rel = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"])
    rs_is_stat = fluid.get_state(jxp.PSmass_INPUTS, sp["p"], inlet_plane["entropy"])

    # ---- blockage & mass flow ----
    # If opening is None → treat as ∞ (no BL blockage)
    opening_eff = geometry.get("opening", None)  # may be None
    blk = compute_blockage_boundary_layer(blockage, Re, chord, opening_eff)
    blk = jnp.asarray(blk, dtype=jnp.float64)

    mass_flow = rho * w_m * area * (1.0 - blk)

    # ---- losses ----
    min_val = jnp.array(1e-3, dtype=jnp.float64)

    loss_dict = lm.evaluate_loss_model(
        loss_model,
        {
            "geometry": geometry,
            "flow": {
                "p0_rel_in": inlet_plane["pressure0_rel"],
                "p0_rel_out": sg0r["pressure0_rel"],
                "p_in": inlet_plane["pressure"],
                "p_out": sp["p"],
                "h_out": sp["h"],
                "beta_out": beta,
                "w_out": w,
                "beta_in": inlet_plane["beta"],
                "Ma_rel_in": jnp.maximum(min_val, inlet_plane["Ma_rel"]),
                "Ma_rel_out": jnp.maximum(min_val, Ma_rel),
                "Re_in": jnp.maximum(min_val, inlet_plane["Re"]),
                "Re_out": jnp.maximum(min_val, Re),
                "gamma_out": sp["gamma"],
                "p0_rel_is": rs_is_rel["p"],
                "h_is": rs_is_stat["h"],
            },
        },
    )

    # ---- assemble plane ----
    plane = {
        **vt,
        **sp,
        **sg0,
        **sg0r,
        **loss_dict,
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": mass_flow,
        "rothalpy": rothalpy_out,
        "blockage": blk,
        "h_is": rs_is_stat["h"],
    }
    return plane, loss_dict


# def evaluate_cascade_exit(
#     cascade_exit_input,
#     fluid,
#     geometry,
#     inlet_plane,
#     angular_speed,
#     blockage,
#     loss_model,
# ):
#     """Exit plane: velocity triangles, thermo state, Re/Ma, mass flow, losses."""
#     w     = cascade_exit_input["w"]
#     beta  = cascade_exit_input["beta"]
#     s     = cascade_exit_input["s"]
#     rothalpy = cascade_exit_input["rothalpy"]

#     chord   = geometry["chord"]
#     opening = geometry.get("opening", None)
#     area    = geometry["A_out"]
#     radius  = geometry["radius_mean_out"]

#     blade_speed = angular_speed * radius
#     velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
#     v   = velocity_triangle["v"]
#     w_m = velocity_triangle["w_m"]

#     h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
#     static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

#     rho = static_properties["d"]
#     mu  = static_properties["mu"]
#     a   = static_properties["a"]

#     h0 = h + 0.5 * v**2
#     stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
#     stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

#     h0_rel = h + 0.5 * w**2
#     relative_stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)
#     relative_stagnation_properties = utils.add_string_to_keys(relative_stagnation_properties, "0_rel")

#     Ma     = v / a
#     Ma_rel = w / a
#     Re     = rho * w * chord / mu
#     rothalpy = h0_rel - 0.5 * blade_speed**2

#     relative_stagnation_isentropic_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"])
#     relative_static_isentropic_properties     = fluid.get_state(jxp.PSmass_INPUTS, static_properties["p"], inlet_plane["entropy"])

#     blockage_factor = compute_blockage_boundary_layer(
#         blockage,
#         Re,
#         chord,
#         opening if (opening is not None) else jnp.inf
#     )
#     mass_flow = rho * w_m * area * (1 - blockage_factor)

#     min_val = 1e-3

#     loss_dict = lm.evaluate_loss_model(
#         loss_model,
#         {
#             "geometry": geometry,
#             "flow": {
#                 "p0_rel_in": inlet_plane["pressure0_rel"],
#                 # "p0_rel_in": inlet_plane["p0_rel"],
#                 "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
#                 # "p0_rel_out": relative_stagnation_properties["p0_rel"],
#                 "p_in": inlet_plane["pressure"],
#                 # "p_in": inlet_plane["p"],
#                 "p_out": static_properties["p"],
#                 "h_out": static_properties["h"],
#                 "beta_out": beta,
#                 "w_out" : w,
#                 "beta_in": inlet_plane["beta"],
#                 "Ma_rel_in": jnp.maximum(min_val, inlet_plane["Ma_rel"]),
#                 "Ma_rel_out": jnp.maximum(min_val, Ma_rel),
#                 "Re_in": jnp.maximum(min_val, inlet_plane["Re"]),
#                 "Re_out": jnp.maximum(min_val, Re),
#                 "gamma_out": static_properties["gamma"],
#                 "p0_rel_is" : relative_stagnation_isentropic_properties["p"],
#                 "h_is" : relative_static_isentropic_properties["h"],
#             },
#         }
#     )

#     plane = {
#         **velocity_triangle,
#         **static_properties,
#         **stagnation_properties,
#         **relative_stagnation_properties,
#         **loss_dict,
#         "Ma": Ma,
#         "Ma_rel": Ma_rel,
#         "Re": Re,
#         "mass_flow": mass_flow,
#         "rothalpy": rothalpy,
#         "blockage": blockage_factor,
#         "h_is" : relative_static_isentropic_properties["h"],
#     }
#     return plane, loss_dict


def evaluate_cascade_throat(
    cascade_throat_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    """Throat plane evaluation (optional path); consistent with exit."""
    w = cascade_throat_input["w"]
    beta = cascade_throat_input["beta"]
    s = cascade_throat_input["s"]
    rothalpy = cascade_throat_input["rothalpy"]

    chord = geometry["chord"]
    opening = geometry.get("opening", None)
    area = geometry["A_throat"]
    radius = geometry["radius_mean_throat"]

    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]

    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    # print(h, rothalpy,blade_speed,w)
    static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

    h0 = h + 0.5 * v**2
    stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(
        relative_stagnation_properties, "0_rel"
    )

    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    rothalpy = h0_rel - 0.5 * blade_speed**2

    relative_stagnation_isentropic_properties = fluid.get_state(
        jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"]
    )
    relative_static_isentropic_properties = fluid.get_state(
        jxp.PSmass_INPUTS, static_properties["pressure"], inlet_plane["entropy"]
    )

    blockage_factor = compute_blockage_boundary_layer(
        blockage, Re, chord, opening if (opening is not None) else jnp.inf
    )
    mass_flow = rho * w * area * (1 - blockage_factor)

    min_val = 1e-3

    loss_dict = lm.evaluate_loss_model(
        loss_model,
        {
            "geometry": geometry,
            "flow": {
                "p0_rel_in": inlet_plane["pressure0_rel"],
                # "p0_rel_in": inlet_plane["p0_rel"],
                "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
                # "p0_rel_out": relative_stagnation_properties["p0_rel"],
                "p_in": inlet_plane["pressure"],
                # "p_in": inlet_plane["p"],
                "p_out": static_properties["p"],
                "h_out": static_properties["h"],
                "beta_out": beta,
                "w_out": w,
                "beta_in": inlet_plane["beta"],
                "Ma_rel_in": jnp.maximum(min_val, inlet_plane["Ma_rel"]),
                "Ma_rel_out": jnp.maximum(min_val, Ma_rel),
                "Re_in": jnp.maximum(min_val, inlet_plane["Re"]),
                "Re_out": jnp.maximum(min_val, Re),
                "gamma_out": static_properties["gamma"],
                "p0_rel_is": relative_stagnation_isentropic_properties["p"],
                "h_is": relative_static_isentropic_properties["h"],
            },
        },
    )

    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
        **loss_dict,
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": mass_flow,
        "rothalpy": rothalpy,
        "blockage": blockage_factor,
        "h_is": relative_static_isentropic_properties["h"],
    }
    return plane, loss_dict


# ============================================================
# BladeRow (Equinox module)
# ============================================================


class BladeRow(eqx.Module):
    """
    Equinox-compatible blade row wrapper around the local inlet/exit primitives.

    Static (non-JAX) fields are marked with eqx.field(static=True).
    """

    # Identity / type
    name: str = eqx.field(static=True)
    cascade_type: str = eqx.field(static=True)  # "stator" or "rotor"

    # Geometry kept as a mapping of arrays (pytree-friendly and compatible)
    geometry: Dict[str, Any]

    # Model options (static selectors + one dynamic numeric tunable as example)
    loss_model: Any = eqx.field(static=True)
    loss_coefficient: str = eqx.field(static=True)
    deviation_model: str = eqx.field(static=True)
    choking_criterion: str = eqx.field(static=True)
    blockage_model: Any = eqx.field(static=True)

    inlet_displacement_thickness_height_ratio: jnp.ndarray

    # External fluid handle (non-JAX)
    fluid: Any = eqx.field(static=True)

    initial_guess_spec: Dict[str, Any] = eqx.field(static=True, default_factory=dict)

    # -----------------
    # Constructors
    # -----------------

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        *,
        fluid: Any,
        model_options_global: Dict[str, Any] | None = None,
    ) -> "BladeRow":
        """
        Build a BladeRow from a nested component dict.
        Requires: config["cascade_type"], config["geometry"].
        """
        name = config.get("name", "blade_row")
        ctype = str(config["cascade_type"]).lower()
        geom_raw = config["geometry"]
        per_opts = config.get("model_options", {})
        ig_spec = config.get("initial_guess", {})
        component_type = config["component_type"]

        # ---------------------------------------------------------
        # TODO: compute the full geometry here
        # New Roberto 17.11.2025
        if component_type == "axial_cascade":
            full_geom = geometry_model_axial.calculate_full_geometry(geom_raw)

        elif component_type == "radial_cascade":
            full_geom = geometry_model_radial.calculate_full_geometry(geom_raw)
        else:
            raise ValueError("Invalid component type")
        # ---------------------------------------------------------

        # Merge options + validate geometry (Benner throat if needed)
        mo = _normalize_model_options(model_options_global, per_opts)
        require_throat = "benner" in str(mo["loss_model"]).lower()
        _validate_geometry_component(name, geom_raw, require_throat=require_throat)

        geometry = _to_jax_dict(geom_raw)

        return cls(
            name=name,
            cascade_type=ctype,
            geometry=full_geom,
            loss_model=mo["loss_model"],
            loss_coefficient=mo["loss_coefficient"],
            deviation_model=mo["deviation_model"],
            choking_criterion=mo["choking_criterion"],
            blockage_model=mo["blockage_model"],
            inlet_displacement_thickness_height_ratio=jnp.array(
                mo["inlet_displacement_thickness_height_ratio"], dtype=jnp.float64
            ),
            fluid=fluid,
            initial_guess_spec=ig_spec,
        )

    # -----------------
    # Evaluation
    # -----------------

    def evaluate(
        self,
        inlet_state: Dict[str, Any],  # {"h0","s","alpha","v"}
        row_vars: Dict[str, Any],  # {"w_out","s_out","beta_out"}
        omega: jnp.ndarray,  # scalar (0 for stator; ω for rotor)
        reference_values: Dict[str, Any],  # must include "mass_flow_ref"
        choking_vars: Dict[str, Any],  # per-row "*crit*" variables
    ) -> Tuple[list[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Run the per-row evaluation using the local inlet/exit primitives and choking model.

        Returns:
          planes_list: [inlet_plane, exit_plane]
          cascade_summary: dict
          residuals: dict
          handoff: dict (for evaluate_cascade_interspace)
        """
        fluid = self.fluid
        g = self.geometry

        # 1) Inlet plane
        inlet_plane = evaluate_cascade_inlet(inlet_state, fluid, g, omega)

        # 2) Exit plane (+ losses & blockage)
        exit_input = {
            "w": row_vars["w_out"],
            "beta": row_vars["beta_out"],
            "s": row_vars["s_out"],
            "rothalpy": inlet_plane["rothalpy"],
        }
        exit_plane, loss_dict = evaluate_cascade_exit(
            exit_input,
            fluid,
            g,
            inlet_plane,
            omega,
            self.blockage_model,
            self.loss_model,
        )

        # 3) Isentropic drop to same p and inlet s
        props_is = fluid.get_state(
            jxp.PSmass_INPUTS, exit_plane["pressure"], inlet_plane["entropy"]
        )
        dh_is = exit_plane["enthalpy"] - props_is["enthalpy"]

        # 4) Choking model options dict (kept minimal & compatible)
        mo_dict = {
            "loss_model": self.loss_model,
            "loss_coefficient": self.loss_coefficient,
            "deviation_model": self.deviation_model,
            "choking_criterion": self.choking_criterion,
            "blockage_model": self.blockage_model,
            "inlet_displacement_thickness_height_ratio": self.inlet_displacement_thickness_height_ratio,
        }
        residuals_critical, critical_state = cm.evaluate_choking(
            choking_vars,
            inlet_plane,
            exit_plane,
            fluid,
            g,
            omega,
            mo_dict,
            reference_values,
        )

        # 5) Residuals
        mass_error_exit = inlet_plane["mass_flow"] - exit_plane["mass_flow"]
        residuals = {
            "loss_error_exit": exit_plane["loss_error"],
            "mass_error_exit": mass_error_exit / reference_values["mass_flow_ref"],
            **residuals_critical,
        }

        # 6) Per-row summary
        cascade_summary = {
            **loss_dict,
            **critical_state,
            "dh_s": dh_is,
            "incidence": inlet_plane["beta"] - g["leading_edge_angle"],
            "name": self.name,
            "cascade_type": self.cascade_type,
        }

        # 7) Handoff payload for interspace
        handoff = {
            "enthalpy0_out": exit_plane["enthalpy0"],
            "v_m_out": exit_plane["v_m"],
            "v_t_out": exit_plane["v_t"],
            "rho_out": exit_plane["density"],
            "r_out": g["radius_mean_out"],
            "A_out": g["A_out"],
            "blockage_out": exit_plane["blockage"],
        }

        planes_list = [inlet_plane, exit_plane]
        return planes_list, cascade_summary, residuals, handoff

    # ---------- NEW: per-row initial guess synthesizer ----------
    def build_initial_guess(
        self,
        inlet_state: Dict[str, Any],  # {"h0","s","alpha","v"}; alpha in degrees
        omega: jnp.ndarray,  # scalar; 0 for stator, ω for rotor
        *,
        use_eta_for_entropy: bool = True,
        eta_blend: float = 0.6,  # 0 -> pure Mach seed; 1 -> pure eta seed
        beta_fallback_deg: float = 30.0,
        choking_criterion: str | None = None,
    ) -> Dict[str, Any]:
        """
        Return an initial-guess dict for THIS blade row only:
          {"w_out","s_out","beta_out","w_crit_throat","s_crit_throat","v_crit_in"}.

        Strategy:
          1) Mach-seed: enforce target Ma_rel at s_out ≈ s_in to get p_out and (h_out_M, a_out_M).
          2) Eta-seed: at same p_out, compute h_is(p_out, s_in), then set
               h_out_eta = h_is + (1 - eta_tt_row) * (h0_rel_out - h_is)
             → get s_out_eta from (p_out, h_out_eta).
          3) Blend: h_out = (1-eta_blend)*h_out_M + eta_blend*h_out_eta, then recompute state.
        """
        g = self.geometry
        fluid = self.fluid

        # ---- Hints from YAML ----
        ig = self.initial_guess_spec or {}
        Ma_t = (
            ig.get("ma_1")
            or ig.get("ma_out")
            or ig.get("ma_rel_out")
            or ig.get("ma")
            or 0.8
        )
        eta_tt_row = float(ig.get("efficiency_tt", 0.90))
        # Clamp to reasonable bounds for stability
        eta_tt_row = float(jnp.clip(eta_tt_row, 0.5, 1.0))

        # ---- Inlet kinematics & rothalpy (absolute α in degrees) ----
        alpha_in = float(inlet_state["alpha"])
        v_in = float(inlet_state["v"])
        h0_in = float(inlet_state["h0"])
        s_in = float(inlet_state["s"])

        U_in = float(omega) * float(g["radius_mean_in"])
        U_out = float(omega) * float(g["radius_mean_out"])

        vt_in = v_in * math.sind(alpha_in)
        vm_in = v_in * math.cosd(alpha_in)
        wt_in = vt_in - U_in
        wm_in = vm_in
        w_in = jnp.sqrt(wt_in**2 + wm_in**2)

        h_in = h0_in - 0.5 * v_in**2
        rothalpy_in = h_in + 0.5 * float(w_in) ** 2 - 0.5 * U_in**2
        h0_rel_out = rothalpy_in + 0.5 * U_out**2

        # ---- (1) Mach-seed at s_out ≈ s_in ----
        s_out_seed = s_in

        def f_pressure(p):
            st = fluid.get_state(jxp.PSmass_INPUTS, p, s_out_seed)
            h = st["h"]
            a = st["speed_sound"]
            return float(h - h0_rel_out + 0.5 * Ma_t**2 * a**2)

        # robust bracket around inlet static pressure
        p_ref = float(fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)["p"])
        p_lo = max(1.0e3, 0.1 * p_ref)
        p_hi = 5.0 * p_ref
        try:
            root = optimize.root_scalar(
                f_pressure, method="bisect", bracket=(p_lo, p_hi), xtol=1e-6
            )
            p_out = float(root.root)
        except Exception:
            root = optimize.root_scalar(
                f_pressure, method="secant", x0=p_ref, x1=0.8 * p_ref
            )
            p_out = float(root.root)

        st_M = fluid.get_state(jxp.PSmass_INPUTS, p_out, s_out_seed)
        h_out_M = float(st_M["h"])
        a_out_M = float(st_M["speed_sound"])
        rho_out_M = float(st_M["d"])

        w_out_M = float(jnp.sqrt(jnp.maximum(0.0, h0_rel_out - h_out_M) * 2.0))

        # ---- (2) Eta-seed at same p_out (uses eta_tt_row) ----
        # Isentropic static enthalpy at this pressure from inlet entropy:
        st_is = fluid.get_state(jxp.PSmass_INPUTS, p_out, s_in)
        h_is = float(st_is["h"])

        # Interpret eta_tt_row as total-to-static (relative) for a seed:
        #   eta_ts_row ≈ eta_tt_row  (seed-level equivalence)
        #   h_out_eta = h_is + (1 - eta_ts_row) * (h0_rel_out - h_is)
        h_out_eta = h_is + (1.0 - eta_tt_row) * (h0_rel_out - h_is)

        # Convert (p_out, h_out_eta) to full state → entropy s_out_eta, sound speed a_out_eta
        st_eta = fluid.get_state(jxp.HmassP_INPUTS, h_out_eta, p_out)
        s_out_eta = float(st_eta["s"])
        a_out_eta = float(st_eta["speed_sound"])
        rho_out_eta = float(st_eta["d"])
        w_out_eta = float(jnp.sqrt(jnp.maximum(0.0, h0_rel_out - h_out_eta) * 2.0))

        # ---- (3) Blend Mach- and eta-seeds for robustness ----
        eta_blend = float(jnp.clip(eta_blend, 0.0, 1.0))
        h_out = (1.0 - eta_blend) * h_out_M + eta_blend * h_out_eta

        # Recompute the final exit state at (p_out, h_out)
        st_final = fluid.get_state(jxp.HmassP_INPUTS, h_out, p_out)
        s_out = float(st_final["s"])
        a_out = float(st_final["speed_sound"])
        rho_out = float(st_final["d"])
        w_out = float(jnp.sqrt(jnp.maximum(0.0, h0_rel_out - h_out) * 2.0))

        # ---- Exit angle seed (geometric gauge; sign by cascade type) ----
        beta_sign = -1.0 if ("rotor" in self.cascade_type) else +1.0
        A_out = float(g["A_out"])
        A_th = float(g.get("A_throat") or 0.0)
        if A_th > 0.0 and A_out > 0.0 and (A_th <= A_out):
            beta_mag = math.arccosd(A_th / A_out)
        else:
            beta_mag = beta_fallback_deg
        beta_out = float(beta_sign * beta_mag)

        # ---- Critical throat seeds (M_rel = 1) ----
        w_crit = a_out
        h_th_crit = h0_rel_out - 0.5 * w_crit**2
        st_th_crit = fluid.get_state(jxp.HmassSmass_INPUTS, h_th_crit, s_out)
        rho_th_crit = float(st_th_crit["d"])
        A_throat = float(A_th if A_th > 0.0 else A_out * 0.95)
        m_dot_crit = rho_th_crit * w_crit * A_throat

        # ---- Inlet-crit proxy for NEXT row (used by some choking criteria) ----
        st_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)
        rho_in = float(st_in["d"])
        A_in = float(g["A_in"])
        vm_in_crit_next = m_dot_crit / max(1e-9, rho_in * A_in)
        v_in_crit_next = vm_in_crit_next / max(1e-6, math.cosd(alpha_in))

        return {
            "w_out": w_out,
            "s_out": s_out,
            "beta_out": beta_out,
            "w_crit_throat": w_crit,
            "s_crit_throat": s_out,
            "v_crit_in": v_in_crit_next,
        }
