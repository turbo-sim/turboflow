# blade_row.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from scipy import optimize
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
import jaxprop as jxp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Project-local deps (same as your flow model)
from .. import math
from .. import utilities as utils
from . import loss_model as lm
from . import choking_criterion as cm
from matplotlib.patches import Arc 


# from turboflow.radial_outflow_turbine import blade_parametrization as bp  
from turboflow.radial_outflow_turbine import blade_parametrization_update as bp 
from . import geometry_model_axial, geometry_model_radial




# ============================================================
# Constants
# ============================================================

BLOCKAGE_MODELS = ["flat_plate_turbulent"]

REQUIRED_MODEL_OPTION_KEYS = {
    "loss_model",
    "loss_coefficient",
    "deviation_model",
    "choking_criterion",
    "blockage_model",
    "inlet_displacement_thickness_height_ratio",
}

dispatch = {
    "axial_cascade": 0,
    "radial_cascade": 1,
}
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
    """
    Merge global and per-component model options WITHOUT applying defaults.

    Rules:
      - global_opts applied first
      - comp_opts override globals
      - all REQUIRED_MODEL_OPTION_KEYS must be present after merging
    """
    mo: Dict[str, Any] = {}

    if global_opts:
        mo.update(global_opts)
    if comp_opts:
        mo.update(comp_opts)

    # Require all keys
    missing = [k for k in REQUIRED_MODEL_OPTION_KEYS if k not in mo]
    if missing:
        raise ValueError(
            f"model_options is missing required keys: {missing}. "
            "No defaults are applied; please specify them explicitly."
        )

    # Optional: you can enforce some minimal type checks if you want
    if mo["loss_model"] is None:
        raise ValueError("model_options['loss_model'] must not be None.")

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

@eqx.filter_jit
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

@eqx.filter_jit
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

@eqx.filter_jit
def compute_blockage_boundary_layer(blockage_model, Re, chord, opening):
    """Boundary-layer blockage. If opening is None/inf → 0 (JIT-safe)."""


    opening_arr = jnp.asarray(opening, dtype=jnp.float64)
    opening_safe = jnp.maximum(opening_arr, 1e-9)

    # Base blockage_factor depending on model (blockage_model is static under filter_jit)
    if blockage_model == BLOCKAGE_MODELS[0]:
        displacement_thickness = 0.048 / Re ** (1 / 5) * 0.9 * chord
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
    # blockage_factor = jnp.where(jnp.isinf(opening_arr), 0.0, blockage_factor)
    return blockage_factor


@eqx.filter_jit
def _cascade_exit_numeric_core(
    w,
    beta,
    blade_speed,
    chord,
    area,
    blockage_factor,
    rho,
    mu,
    a,
):
    """JIT-friendly numeric core for cascade exit kinematics/dimensionless groups."""
    vt = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = vt["v"]
    w_mag = vt["w"]
    w_m = vt["w_m"]

    Ma = v / a
    Ma_rel = w_mag / a
    Re = rho * jnp.abs(w_mag) * chord / mu
    mass_flow = rho * w_m * area * (1.0 - blockage_factor)

    return vt, Ma, Ma_rel, Re, mass_flow

@eqx.filter_jit
def _cascade_throat_numeric_core(
    w,
    beta,
    blade_speed,
    chord,
    area,
    rho,
    mu,
    a,
):
    """JIT-friendly numeric core for throat kinematics/dimensionless groups (uses w, not w_m, for mass flow as in original)."""
    vt = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = vt["v"]
    w_mag = vt["w"]

    Ma = v / a
    Ma_rel = w_mag / a
    Re = rho * jnp.abs(w_mag) * chord / mu
    mass_flow_no_blk = rho * w_mag * area  # original throat mass-flow uses w

    return vt, Ma, Ma_rel, Re, mass_flow_no_blk


@eqx.filter_jit
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
    Re = rho * jnp.abs(w) * chord / mu
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
        "loss_definition": zero,
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


@eqx.filter_jit
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

    # jax.debug.print(
    #     " s={s},  s_in={s_in}\n",
    #     s=s,
    #     s_in=inlet_plane["entropy"],
    # )

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

    # nondim groups and refreshed kinematics using true properties (no blockage)
    vt, Ma, Ma_rel, Re, mass_flow_no_blk = _cascade_exit_numeric_core(
        w=w,
        beta=beta,
        blade_speed=blade_speed,
        chord=chord,
        area=area,
        blockage_factor=jnp.array(0.0),  # mass_flow handled below with actual blockage
        rho=rho,
        mu=mu,
        a=a,
    )
    rothalpy_out = h0_rel - 0.5 * blade_speed**2  # (matches inlet def)

    # isentropic references (relative and static)
    rs_is_rel = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"])
    rs_is_stat = fluid.get_state(jxp.PSmass_INPUTS, sp["p"], inlet_plane["entropy"])

    # ---- blockage & mass flow ----
    # If opening is None → treat as ∞ (no BL blockage)
    opening_eff = geometry.get("opening", None)  # may be None
    blk = compute_blockage_boundary_layer(blockage, Re, chord, opening_eff)
    blk = jnp.asarray(blk, dtype=jnp.float64)

    # Full mass flow using blockage factor
    mass_flow = mass_flow_no_blk * (1.0 - blk)

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

@eqx.filter_jit
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
    opening = geometry["opening"]
    area = geometry["A_throat"]
    radius = geometry["radius_mean_throat"]

    blade_speed = angular_speed * radius

    # # ---------- DEBUG: throat inputs ----------
    # jax.debug.print(
    #     "\n[THROAT INPUT]\n"
    #     "  w={w}\n"
    #     "  beta={beta}\n"
    #     "  s={s}\n"
    #     "  rothalpy_in={H0}\n"
    #     "  radius={r}, blade_speed={U}\n"
    #     "  chord={chord}, opening={opening}, area={area}",
    #     w=w,
    #     beta=beta,
    #     s=s,
    #     H0=rothalpy,
    #     r=radius,
    #     U=blade_speed,
    #     chord=chord,
    #     opening=opening,
    #     area=area,
    # )

    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]

    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
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

    # nondim and mass flow using throat-specific numeric core (matches original throat mass-flow definition)
    vt_core, Ma, Ma_rel, Re, mass_flow_no_blk = _cascade_throat_numeric_core(
        w=w,
        beta=beta,
        blade_speed=blade_speed,
        chord=chord,
        area=area,
        rho=rho,
        mu=mu,
        a=a,
    )
    # overwrite velocity_triangle components with core output for consistency
    velocity_triangle = {**velocity_triangle, **vt_core}
    rothalpy = h0_rel - 0.5 * blade_speed**2

    # # ---------- DEBUG: throat state ----------
    # jax.debug.print(
    #     "[THROAT STATE]\n"
    #     "  h={h}\n"
    #     "  h0={h0}, h0_rel={h0_rel}\n"
    #     "  p={p}, T={T}\n"
    #     "  rho={rho}, mu={mu}, a={a}\n"
    #     "  v={v}, w={w}, Ma={Ma}, Ma_rel={Ma_rel}\n"
    #     "  Re={Re}",
    #     h=h,
    #     h0=h0,
    #     h0_rel=h0_rel,
    #     p=static_properties["p"],
    #     T=static_properties["T"],
    #     rho=rho,
    #     mu=mu,
    #     a=a,
    #     v=v,
    #     w=w,
    #     Ma=Ma,
    #     Ma_rel=Ma_rel,
    #     Re=Re,
    # )

    relative_stagnation_isentropic_properties = fluid.get_state(
        jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"]
    )
    relative_static_isentropic_properties = fluid.get_state(
        jxp.PSmass_INPUTS, static_properties["pressure"], inlet_plane["entropy"]
    )

    blockage_factor = compute_blockage_boundary_layer(
        blockage, Re, chord, opening
    )
    mass_flow = mass_flow_no_blk * (1 - blockage_factor)

    # # ---------- DEBUG: throat mass flow / blockage ----------
    # jax.debug.print(
    #     "[THROAT FLOW]\n"
    #     "  blockage_factor={blk}\n"
    #     "  mass_flow={m_dot}",
    #     blk=blockage_factor,
    #     m_dot=mass_flow,
    # )

    min_val = 1e-3

    loss_dict = lm.evaluate_loss_model(
        loss_model,
        {
            "geometry": geometry,
            "flow": {
                "p0_rel_in": inlet_plane["pressure0_rel"],
                "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
                "p_in": inlet_plane["pressure"],
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

    # # ---------- DEBUG: throat losses ----------
    # jax.debug.print(
    #     "[THROAT LOSSES]\n"
    #     "  Y_tot={Y_tot}\n"
    #     "  Y_def={Y_def}\n"
    #     "  loss_error (Y*)={Y_err}",
    #     Y_tot=loss_dict["loss_total"],
    #     Y_def=loss_dict["loss_definition"],
    #     Y_err=loss_dict["loss_total"] - loss_dict["loss_definition"],
    # )

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
    component_type: str = eqx.field(static=True)

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
    fluid: Any 

    initial_guess_spec: Dict[str, Any] = eqx.field(static=True, default_factory=dict)

    # @staticmethod
    # def unscale_row_vars_and_choking(
    #     variables: Dict[str, Any],
    #     index_1based: int,
    #     reference_values: Dict[str, Any],
    # ) -> tuple[Dict[str, Any], Dict[str, Any]]:
    #     """
    #     Convert normalized solver variables for a given row index into
    #     physical row_vars and choking_vars.

    #     Expects normalized keys:
    #         w_out_i, s_out_i, beta_out_i,
    #         optional: w_crit_throat_i, s_crit_throat_i, v_crit_in_i
    #     """
    #     tag = f"_{index_1based}"

    #     v0 = reference_values["v0"]
    #     s_range = reference_values["s_range"]
    #     s_min = reference_values["s_min"]
    #     a_range = reference_values["angle_range"]
    #     a_min = reference_values["angle_min"]

    #     # -------- main row unknowns (always present) --------
    #     row_vars = {
    #         "w_out": jnp.maximum(1.0, variables[f"w_out{tag}"] * v0),
    #         # "s_out": jnp.clip(variables[f"s_out{tag}"], 0.0, 1.0) * s_range + s_min,
    #         "s_out": variables[f"s_out{tag}"] * s_range + s_min,
    #         "beta_out": jnp.clip(variables[f"beta_out{tag}"] * a_range + a_min, -89.0, 89.0)
    #     }


    #     # -------- choking-related unknowns (optional) --------
    #     choking_vars: Dict[str, Any] = {}

    #     # Reparameterized w_crit_throat: solver variable is z (dimensionless),
    #     # mapped via tanh into a bounded physical velocity range.
    #     # if f"w_crit_throat{tag}" in variables:
    #     #     z = variables[f"w_crit_throat{tag}"]  # normalized solver variable (O(1))

    #     #     # Choose a reference speed scale; v0 is a natural choice
    #     #     a_ref = v0

    #     #     # Max allowed factor
    #     #     w_max_fac = 0.50

    #     #     # Define a physical interval for w_throat
    #     #     w_min = 0.01 * a_ref      # small but non-zero velocity
    #     #     w_max = w_max_fac * a_ref   # upper bound on throat relative speed

    #     #     # Map z ∈ R → w_throat ∈ [w_min, w_max]
    #     #     w_throat = w_min + 0.5 * (w_max - w_min) * (jnp.tanh(z) + 1.0)

    #     #     choking_vars["w_crit_throat"] = w_throat


    #     if f"w_crit_throat{tag}" in variables:
    #         choking_vars["w_crit_throat"] = jnp.maximum(1.0, variables[f"w_crit_throat{tag}"] * v0)

    #     if f"s_crit_throat{tag}" in variables:
    #         choking_vars["s_crit_throat"] = (
    #             # jnp.clip(variables[f"s_crit_throat{tag}"], 0.0, 1.0) * s_range + s_min
    #             variables[f"s_crit_throat{tag}"] * s_range + s_min
    #         )

    #     # # IMPORTANT: use the per-row key v_crit_in_i, not a global "v_crit_in"
    #     # if f"v_crit_in{tag}" in variables:
    #     #     choking_vars["v_crit_in"] = variables[f"v_crit_in{tag}"] * v0

    #     return row_vars, choking_vars
    
    @staticmethod
    def unscale_row_vars_and_choking(
        variables: Dict[str, Any],
        index_1based: int,
        reference_values: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Mu/sigma unscale to physical row_vars and choking_vars.

        Scaling:
          v*/w*: mu=0, sigma=v0
          s*   : mu=s_min+0.5*s_range, sigma=max(0.5*s_range, s_floor)
          beta*: mu=angle_min+0.5*angle_range, sigma=0.5*angle_range
        """
        tag = f"_{index_1based}"

        v0 = reference_values["v0"]
        s_range = reference_values["s_range"]
        s_min = reference_values["s_min"]
        a_range = reference_values["angle_range"]
        a_min = reference_values["angle_min"]

        s_floor = jnp.maximum(
            jnp.array(1e-6, dtype=jnp.float64),
            0.01 * jnp.maximum(jnp.abs(s_min), 1.0),
        )
        mu_s = s_min + 0.5 * s_range
        sigma_s = jnp.maximum(0.5 * s_range, s_floor)

        mu_b = a_min + 0.5 * a_range
        sigma_b = 0.5 * a_range

        w_out = variables[f"w_out{tag}"] * v0
        s_out = variables[f"s_out{tag}"] * sigma_s + mu_s
        beta_out = variables[f"beta_out{tag}"] * sigma_b + mu_b
        beta_out = jnp.clip(beta_out, -89.0, 89.0)

        row_vars = {
            "w_out": jnp.maximum(1.0, w_out),
            "s_out": s_out,
            "beta_out": beta_out,
        }

        choking_vars: Dict[str, Any] = {}

        if f"w_crit_throat{tag}" in variables:
            w_ct = variables[f"w_crit_throat{tag}"] * v0
            choking_vars["w_crit_throat"] = jnp.maximum(1.0, w_ct)

        if f"s_crit_throat{tag}" in variables:
            s_ct = variables[f"s_crit_throat{tag}"] * sigma_s + mu_s
            choking_vars["s_crit_throat"] = s_ct

        return row_vars, choking_vars

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
        component_type = config["component_type"]
        partial_geom = config["geometry"]
        ctype = str(partial_geom["cascade_type"]).lower()
        per_opts = config.get("model_options", {})
        ig_spec = config.get("initial_guess", {})
        

        # ---------------------------------------------------------
        # TODO: Replace if by jax.lax.switch for JIT compatibility
        # New Roberto 17.11.2025
        if component_type == "axial_cascade":
            full_geom = geometry_model_axial.calculate_full_geometry_for_axial_cascade(partial_geom)

        elif component_type == "radial_cascade":
            full_geom = geometry_model_radial.calculate_full_geometry_for_radial_cascade(partial_geom)
        else:
            raise ValueError("Invalid component type")
        
        # idx = dispatch.get(component_type, -1)
        # if idx == -1:
        #     raise ValueError(f"Invalid component type: {component_type!r}")

        # branches = (
        #     lambda geom: geometry_model_axial.calculate_full_geometry_for_axial_cascade(geom),
        #     lambda geom: geometry_model_radial.calculate_full_geometry_for_radial_cascade(geom),
        # )

        # full_geom = lax.switch(idx, branches, partial_geom)
        # ---------------------------------------------------------

        # Merge options + validate geometry (Benner throat if needed)
        mo = _normalize_model_options(model_options_global, per_opts)
        require_throat = "benner" in str(mo["loss_model"]).lower()
        # _validate_geometry_component(name, full_geom, require_throat=require_throat)

        geometry = _to_jax_dict(full_geom)

        return cls(
            name=name,
            cascade_type=ctype,
            component_type=component_type, 
            geometry=geometry,
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
        row_vars: Dict[str, Any],     # {"w_out","s_out","beta_out"}
        omega: jnp.ndarray,           # scalar (0 for stator; ω for rotor)
        reference_values: Dict[str, Any],  # must include "mass_flow_ref"
        choking_vars: Dict[str, Any],      # per-row "*crit*" variables
    ) -> Dict[str, Any]:
        """
        Run the per-row evaluation using the local inlet/exit primitives and choking model.

        Returns a dictionary with:
          {
            "planes": [inlet_plane, exit_plane],
            "cascade_summary": { ... },
            "residuals": { ... },
            "handoff": { ... },
          }
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
        # print(f"Inlet Plane: {inlet_plane}")
        # print(f"Exit Plane: {exit_plane}")
        
        mass_error_exit = inlet_plane["mass_flow"] - exit_plane["mass_flow"]
        residuals = {
            "loss_error_exit": exit_plane["loss_error"],
            "mass_error_exit": mass_error_exit / reference_values["mass_flow_ref"],
            **residuals_critical,
        }

        # TODO: Mach residual to have least square formulation

        # 5b) Optional subsonic constraint at row exit (inequality as residual)
        # Residual is zero when Ma_rel_out <= Ma_rel_out_max, and grows linearly otherwise.
        ma_lim = reference_values.get("Ma_rel_out_max", None)
        if ma_lim is not None:
            ma_scale = reference_values.get("Ma_rel_out_scale", 0.05)
            ma_lim = jnp.asarray(ma_lim, dtype=jnp.float64)
            ma_scale = jnp.asarray(ma_scale, dtype=jnp.float64)
            ma_scale = jnp.maximum(jnp.array(1e-6, dtype=jnp.float64), ma_scale)
            residuals["Ma_rel_out_max"] = jnp.maximum(
                0.0, exit_plane["Ma_rel"] - ma_lim
            ) / ma_scale

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

        return {
            "planes": planes_list,
            "cascade_summary": cascade_summary,
            "residuals": residuals,
            "handoff": handoff,
        }

    def build_initial_guess(
        self,
        inlet_state: Dict[str, Any],  # {"h0","s","alpha","v"}; alpha in degrees
        omega,                        # scalar; 0 for stator, ω for rotor
        row_index: int,
    ) -> Dict[str, Any]:
        """
        Initial guess for THIS blade row only.

        Returns a dict whose keys already carry the _{row_index} tag, e.g.:
        w_out_1, s_out_1, beta_out_1, w_crit_throat_1, s_crit_throat_1, v_crit_in_1

        BUT the set of keys depends strictly on the choking_criterion:

        - critical_mach_number:
            w_out_i, s_out_i, beta_out_i, w_crit_throat_i, s_crit_throat_i
        - critical_isentropic_throat:
            w_out_i, s_out_i, beta_out_i, w_crit_throat_i
        - critical_mass_flow_rate:
            w_out_i, s_out_i, beta_out_i,
            w_crit_throat_i, s_crit_throat_i, v_crit_in_i

        No hidden fallbacks. If required info is missing, this raises.
        """

        g = self.geometry
        fluid = self.fluid
        ig = self.initial_guess_spec or {}

        # ---------- 1) Required hints from YAML  ----------
        # Mach target
        
        Ma_exit_rel = jnp.asarray(ig["Ma_exit"], dtype=jnp.float64)

        # Row efficiency
        if "efficiency_tt" not in ig:
            raise KeyError(
                f"BladeRow '{self.name}': initial_guess_spec must contain 'efficiency_tt'."
            )
        eta_tt_row = jnp.asarray(ig["efficiency_tt"], dtype=jnp.float64)
        eta_tt_row = jnp.clip(eta_tt_row, 0.5, 1.0)

        # ---------- 2) Inlet kinematics & rothalpy ----------
        alpha_in = jnp.asarray(inlet_state["alpha_in"], dtype=jnp.float64)
        v_in    = jnp.asarray(inlet_state["v_in"],     dtype=jnp.float64)
        h0_in   = jnp.asarray(inlet_state["h0_in"],    dtype=jnp.float64)
        s_in    = jnp.asarray(inlet_state["s_in"],     dtype=jnp.float64)

        omega = jnp.asarray(omega, dtype=jnp.float64)
        r_in  = jnp.asarray(g["radius_mean_in"],  dtype=jnp.float64)
        r_throat = jnp.asarray(g["radius_mean_throat"], dtype=jnp.float64)
        r_out = jnp.asarray(g["radius_mean_out"], dtype=jnp.float64)

        U_in  = omega * r_in
        U_throat = omega * r_throat
        U_out = omega * r_out

        vt_in = v_in * math.sind(alpha_in)
        vm_in = v_in * math.cosd(alpha_in)
        wt_in = vt_in - U_in
        wm_in = vm_in
        w_in  = jnp.sqrt(wt_in**2 + wm_in**2)

        h_in        = h0_in - 0.5 * v_in**2
        rothalpy_in = h_in + 0.5 * w_in**2 - 0.5 * U_in**2
        h0_rel_out  = rothalpy_in + 0.5 * U_out**2
        h0_rel_throat  = rothalpy_in + 0.5 * U_throat**2

        # ---------- 3) Mach-based seed at s_out ≈ s_in ----------
        s_out_seed = s_in

        def f_pressure_scalar(p_scalar: float) -> float:
            p_arr = jnp.asarray(p_scalar, dtype=jnp.float64)
            st = fluid.get_state(jxp.PSmass_INPUTS, p_arr, s_out_seed)
            h = st["h"]
            a = st["a"]
            val = h - h0_rel_out + 0.5 * Ma_exit_rel**2 * a**2 ## This is enthalpy residual
            return float(val)

        # static inlet pressure for bracket
        st_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)
        p_ref = st_in["p"]
        p_lo = float(jnp.maximum(1.0e3, 0.1 * p_ref))
        p_hi = float(5.0 * p_ref)

        root = optimize.root_scalar(
            f_pressure_scalar, method="bisect", bracket=(p_lo, p_hi), xtol=1e-6
        )
        p_out = root.root
        p_out_arr = jnp.asarray(p_out, dtype=jnp.float64)

        # st_M = fluid.get_state(jxp.PSmass_INPUTS, p_out_arr, s_out_seed)
        # h_out_M = st_M["h"]

        # ---------- 4) Eta-based seed at same p_out ----------
        st_is = fluid.get_state(jxp.PSmass_INPUTS, p_out_arr, s_in)
        h_is = st_is["h"]

        # use eta_tt_row as TT row efficiency → TS seed
        h_out_eta = h_is + (1.0 - eta_tt_row) * (h0_rel_out - h_is)

        # final exit state at (p_out, h_out_eta)
        st_final = fluid.get_state(jxp.HmassP_INPUTS, h_out_eta, p_out_arr)
        s_out = st_final["s"]
        a_out = st_final["a"]
        w_out = jnp.sqrt(jnp.maximum(0.0, h0_rel_out - h_out_eta) * 2.0)
        w_m_out = w_out*jnp.cos(jnp.deg2rad(g["metal_angle_out"])) 
        w_t_out = w_out*jnp.sin(jnp.deg2rad(g["metal_angle_out"]))
        v_m_out = w_m_out
        v_t_out = w_t_out + U_out
        v_out = jnp.sqrt(v_m_out**2 + v_t_out**2)
        # build_initial_guess expects/stores alpha_in in degrees for the next row
        alpha_out_deg = jnp.degrees(jnp.atan2(v_t_out, v_m_out))

        # ---------- 5) Exit angle from throat area (no arbitrary fallback) ----------
        # beta_sign = -1.0 if ("rotor" in self.cascade_type) else +1.0
        # A_out = jnp.asarray(g["A_out"], dtype=jnp.float64)
        # A_th  = jnp.asarray(g.get("A_throat", None) or 0.0, dtype=jnp.float64)

        # if (A_th <= 0.0) or (A_out <= 0.0) or (A_th > A_out):
        #     raise ValueError(
        #         f"BladeRow '{self.name}': invalid throat geometry; need 0 < A_throat <= A_out."
        #     )

        # beta_mag  = math.arccosd(A_th / A_out)
        # beta_out  = beta_sign * beta_mag

        beta_out = g["metal_angle_out"]
        A_th  = jnp.asarray(g["A_throat"])

        # ---------- 6) Critical throat (M_rel = 1) ----------

        def f_pressure_scalar(p_scalar: float) -> float:
            p_arr = jnp.asarray(p_scalar, dtype=jnp.float64)
            st = fluid.get_state(jxp.PSmass_INPUTS, p_arr, s_in)
            h = st["h"]
            a = st["a"]
            val = h - h0_rel_throat + 0.5 * 1.0**2 * a**2
            return float(val)

        root = optimize.root_scalar(
            f_pressure_scalar, method="bisect", bracket=(p_lo, p_hi), xtol=1e-6
        )
        p_crit = root.root
        p_crit = jnp.asarray(p_crit, dtype=jnp.float64)

        st_crit = fluid.get_state(jxp.PSmass_INPUTS, p_crit, s_in)

        w_crit_throat = st_crit["a"]
        w_crit = st_final["a"]

        ######
        # print(f"w_crit_exit: {w_crit}, w_crit_throat: {w_crit_throat}")
        ######

        w_crit = w_crit_throat
        
        # rho_crit = st_crit["d"]
        # A_throat    = A_th
        # m_dot_crit  = rho_crit * w_crit * A_throat

        # # ---------- 7) Inlet-crit proxy for NEXT row ----------
        # rho_in = st_in["d"]
        # A_in   = jnp.asarray(g["A_in"], dtype=jnp.float64)
        # vm_in_crit_next = m_dot_crit / (rho_in * A_in)
        # v_in_crit_next  = vm_in_crit_next / math.cosd(alpha_in)

        # ---------- 8) Tag and choking-based key selection ----------
        tag = f"_{row_index}" if row_index is not None and row_index > 0 else ""
        crit = str(self.choking_criterion).lower()

        guess: Dict[str, Any] = {
            f"w_out{tag}": w_out,
            f"s_out{tag}": s_out,
            f"beta_out{tag}": beta_out,
        }

        if crit == "critical_mach_number":
            guess[f"w_crit_throat{tag}"] = w_crit
            guess[f"s_crit_throat{tag}"] = st_crit["s"]

        elif crit == "critical_isentropic_throat":
            guess[f"w_crit_throat{tag}"] = w_crit

        # elif crit == "critical_mass_flow_rate":
        #     guess[f"w_crit_throat{tag}"] = w_crit
        #     guess[f"s_crit_throat{tag}"] = s_out
        #     guess[f"v_crit_in{tag}"] = v_in_crit_next

        else:
            raise ValueError(
                f"BladeRow '{self.name}': unsupported choking_criterion "
                f"'{self.choking_criterion}'."
            )
        
        inlet_seed = {
            "h0_in": st_final["h"] + v_out**2/2,
            "s_in": st_final["s"],
            "alpha_in": alpha_out_deg,
            "v_in": v_out,}

        return guess, inlet_seed

    def build_initial_guess_pr_zeta(
        self,
        inlet_state: Dict[str, Any],  # {"h0","s","alpha","v"}; alpha in degrees
        omega,                        # scalar; 0 for stator, ω for rotor
        row_index: int,
    ) -> Dict[str, Any]:
        """
        Initial guess using total-to-static pressure ratio (PR_ts) and
        enthalpy-based loss coefficient (zeta_h).

        Expects in self.initial_guess_spec:
            PR_ts  : p0_in / p_out (total-to-static)
            zeta_h : fraction of ideal total-to-static enthalpy drop that is lost
        """
        g = self.geometry
        fluid = self.fluid
        ig = self.initial_guess_spec or {}

        if "PR_ts" not in ig or "zeta_h" not in ig:
            raise KeyError(
                f"BladeRow '{self.name}': initial_guess_spec must contain 'PR_ts' and 'zeta_h'."
            )

        PR_ts = jnp.asarray(ig["PR_ts"], dtype=jnp.float64)
        zeta_h = jnp.asarray(ig["zeta_h"], dtype=jnp.float64)
        eta_row = 1.0 - zeta_h

        alpha_in = jnp.asarray(inlet_state["alpha_in"], dtype=jnp.float64)
        v_in = jnp.asarray(inlet_state["v_in"], dtype=jnp.float64)
        h0_in = jnp.asarray(inlet_state["h0_in"], dtype=jnp.float64)
        s_in = jnp.asarray(inlet_state["s_in"], dtype=jnp.float64)

        # If p0_in is provided, use it; otherwise infer from (h0_in, s_in)
        p0_in = inlet_state.get("p0_in", None)
        if p0_in is None:
            st0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0_in, s_in)
            p0_in = st0["p"]
        p0_in = jnp.asarray(p0_in, dtype=jnp.float64)

        omega = jnp.asarray(omega, dtype=jnp.float64)
        r_in = jnp.asarray(g["radius_mean_in"], dtype=jnp.float64)
        r_throat = jnp.asarray(g["radius_mean_throat"], dtype=jnp.float64)
        r_out = jnp.asarray(g["radius_mean_out"], dtype=jnp.float64)

        U_in = omega * r_in
        U_throat = omega * r_throat
        U_out = omega * r_out

        vt_in = v_in * math.sind(alpha_in)
        vm_in = v_in * math.cosd(alpha_in)
        wt_in = vt_in - U_in
        wm_in = vm_in
        w_in = jnp.sqrt(wt_in**2 + wm_in**2)

        h_in = h0_in - 0.5 * v_in**2
        rothalpy_in = h_in + 0.5 * w_in**2 - 0.5 * U_in**2
        h0_rel_out = rothalpy_in + 0.5 * U_out**2
        h0_rel_throat = rothalpy_in + 0.5 * U_throat**2

        # Exit static pressure from PR_ts
        p_out = p0_in / PR_ts

        # Isentropic reference at exit
        st_is = fluid.get_state(jxp.PSmass_INPUTS, p_out, s_in)
        h_is = st_is["h"]

        # Apply enthalpy loss coefficient on total-to-static drop
        h_out = h_is + zeta_h * (h0_rel_out - h_is)  # = h0_rel_out - eta_row * (h0_rel_out - h_is)

        # Exit state and kinematics
        st_final = fluid.get_state(jxp.HmassP_INPUTS, h_out, p_out)
        s_out = st_final["s"]
        beta_out = g["metal_angle_out"]

        w_out = jnp.sqrt(jnp.maximum(0.0, 2.0 * (h0_rel_out - h_out)))
        w_m_out = w_out * jnp.cos(jnp.deg2rad(beta_out))
        w_t_out = w_out * jnp.sin(jnp.deg2rad(beta_out))
        v_m_out = w_m_out
        v_t_out = w_t_out + U_out
        v_out = jnp.sqrt(v_m_out**2 + v_t_out**2)
        # build_initial_guess_pr_zeta expects/stores alpha_in in degrees for the next row
        alpha_out_deg = jnp.degrees(jnp.atan2(v_t_out, v_m_out))
        # alpha_out_deg = jnp.atan2(v_t_out, v_m_out)
        # jax.debug.print(f"alpha_out_deg: {alpha_out_deg}, alpha_out_deg1: {alpha_out_deg1}")

        # Critical throat (Mach_rel = 1)
        def f_pressure_scalar(p_scalar: float) -> float:
            p_arr = jnp.asarray(p_scalar, dtype=jnp.float64)
            st = fluid.get_state(jxp.PSmass_INPUTS, p_arr, s_in)
            h = st["h"]
            a = st["a"]
            val = h - h0_rel_throat + 0.5 * 1.0**2 * a**2
            return float(val)

        st_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)
        p_ref = st_in["p"]
        p_lo = float(jnp.maximum(1.0e3, 0.1 * p_ref))
        p_hi = float(5.0 * p_ref)

        root = optimize.root_scalar(
            f_pressure_scalar, method="bisect", bracket=(p_lo, p_hi), xtol=1e-6
        )
        p_crit = jnp.asarray(root.root, dtype=jnp.float64)
        st_crit = fluid.get_state(jxp.PSmass_INPUTS, p_crit, s_in)
        w_crit_throat = st_crit["a"]

        tag = f"_{row_index}" if row_index is not None and row_index > 0 else ""
        crit = str(self.choking_criterion).lower()

        guess: Dict[str, Any] = {
            f"w_out{tag}": w_out,
            f"s_out{tag}": s_out,
            f"beta_out{tag}": beta_out,
        }

        if crit == "critical_mach_number":
            guess[f"w_crit_throat{tag}"] = w_crit_throat
            guess[f"s_crit_throat{tag}"] = st_crit["s"]
        elif crit == "critical_isentropic_throat":
            guess[f"w_crit_throat{tag}"] = w_crit_throat
        # elif crit == "critical_mass_flow_rate":
        #     guess[f"w_crit_throat{tag}"] = w_crit_throat
        #     guess[f"s_crit_throat{tag}"] = st_crit["s"]
        #     guess[f"v_crit_in{tag}"] = v_in_crit_next
        else:
            raise ValueError(
                f"BladeRow '{self.name}': unsupported choking_criterion '{self.choking_criterion}'."
            )

        inlet_seed = {
            "h0_in": st_final["h"] + 0.5 * v_out**2,
            "s_in": st_final["s"],
            "alpha_in": alpha_out_deg,
            "v_in": v_out,
        }

        return guess, inlet_seed
    
    def plot_meridional(
        self,
        fig=None,
        ax=None,
        n_points: int = 100,
        title: str | None = None,
        plot_meanline: bool = True,
        plot_hub: bool = True,
        plot_shroud: bool = True,
        plot_inlet: bool = True,
        plot_outlet: bool = True,
        label: str | None = None,
        z_center: float = 0.0,   # used only for radial cascades
    ):
        r"""
        Plot the meridional geometry of this blade row.

        RADIAL CASCADE (component_type == "radial_cascade")
        ---------------------------------------------------
        Uses:
            r_in, r_out,
            blade_height_in, blade_height_out

        Draws an annular band in (z, r):
            r ∈ [r_in, r_out],
            z ∈ [z_center - b(r)/2, z_center + b(r)/2].

        AXIAL CASCADE (component_type == "axial_cascade")
        -------------------------------------------------
        Uses:
            z_in,
            radius_hub_in, radius_hub_out,
            radius_shroud_in, radius_shroud_out,
            radius_mean_in, radius_mean_out,
            meridional_chord

        Draws an axial block in (z, r):
            z ∈ [z_in, z_in + meridional_chord],
            radius interpolated between inlet and outlet.
        """

        g = self.geometry
        comp_type = str(getattr(self, "component_type", "")).lower()

        # -------------------------------------------------
        # Create axes if needed (common setup)
        # -------------------------------------------------
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$z$ $-$ Axial coordinate  [m]")
            ax.set_ylabel(r"$r$ $-$ Radial coordinate [m]")

            fmt = mticker.FormatStrFormatter("%.3f")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

        # -------------------------------------------------
        # RADIAL CASCADE: annulus band centered at z_center
        # -------------------------------------------------
        if comp_type == "radial_cascade":
            r_in = g["radius_mean_in"]
            r_out = g["radius_mean_out"]
            b_in = g["blade_height_in"]
            b_out = g["blade_height_out"]

            s = jnp.linspace(0.0, 1.0, n_points)
            r_line = r_in + s * (r_out - r_in)
            b_line = b_in + s * (b_out - b_in)

            z_front = z_center - 0.5 * b_line
            z_back = z_center + 0.5 * b_line

            # Fill and outline the annular band
            ax.fill_betweenx(r_line, z_front, z_back, alpha=0.15, color="0.7")
            ax.plot(z_front, r_line, color="black", linestyle="-")
            ax.plot(z_back, r_line, color="black", linestyle="-")

            # Inlet / outlet arcs at r_in, r_out
            if plot_inlet:
                ax.plot(
                    jnp.array([z_center - 0.5 * b_in, z_center + 0.5 * b_in]),
                    jnp.array([r_in, r_in]),
                    color="black",
                    linestyle="-",
                )
            if plot_outlet:
                ax.plot(
                    jnp.array([z_center - 0.5 * b_out, z_center + 0.5 * b_out]),
                    jnp.array([r_out, r_out]),
                    color="black",
                    linestyle="-",
                )

            if label is None:
                label = self.name
            ax.text(
                z_center,
                0.5 * (r_in + r_out),
                label,
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.25",
                ),
            )

            fig.tight_layout(pad=1.0)
            return fig, ax

        # -------------------------------------------------
        # AXIAL CASCADE: meridional chord along z, starting at z_in
        # -------------------------------------------------
        if comp_type == "axial_cascade":
            # Required geometry keys (as in geometry_model_axial)
            z_in = g["z_in"]
            meridional_chord = g.get("meridional_chord", g.get("chord_axial", g.get("chord")))
            if meridional_chord is None:
                raise KeyError(
                    "Axial cascade requires one of 'meridional_chord', 'chord_axial', or 'chord'."
                )

            radius_hub_in = g["radius_hub_in"]
            radius_hub_out = g["radius_hub_out"]
            radius_shroud_in = g["radius_shroud_in"]
            radius_shroud_out = g["radius_shroud_out"]
            radius_mean_in = g["radius_mean_in"]
            radius_mean_out = g["radius_mean_out"]

            z_start = z_in
            z_end = z_in + meridional_chord

            z_line = jnp.linspace(z_start, z_end, n_points)
            radius_hub_line = jnp.linspace(radius_hub_in, radius_hub_out, n_points)
            radius_shroud_line = jnp.linspace(
                radius_shroud_in, radius_shroud_out, n_points
            )
            radius_mean_line = jnp.linspace(radius_mean_in, radius_mean_out, n_points)

            # Hub / shroud / meanline
            if plot_hub:
                ax.plot(z_line, radius_hub_line, color="black", linestyle="-")
            if plot_shroud:
                ax.plot(z_line, radius_shroud_line, color="black", linestyle="-")
            if plot_meanline:
                ax.plot(z_line, radius_mean_line, color="black", linestyle="--")

            # Inlet and outlet radial edges
            if plot_inlet:
                ax.plot(
                    jnp.array([z_start, z_start]),
                    jnp.array([radius_hub_in, radius_shroud_in]),
                    color="black",
                    linestyle="-",
                )
            if plot_outlet:
                ax.plot(
                    jnp.array([z_end, z_end]),
                    jnp.array([radius_hub_out, radius_shroud_out]),
                    color="black",
                    linestyle="-",
                )


            fig.tight_layout(pad=1.0)
            return fig, ax


       
        raise ValueError(
            f"BladeRow.plot_meridional_geometry: unsupported component_type='{self.component_type}'"
        )
    
    def plot_meridional_tangential(
        self,
        fig=None,
        ax=None,
        n_points: int = 800,
        title: str | None = None,
    ):
        """
        Blade-to-blade (meridional-tangential) view for this BladeRow.

        - component_type == "radial_cascade":
            Plots blades arranged on a circular ring (r-θ mapped to x-y).

        - component_type == "axial_cascade":
            Plots a blade in (x: axial, y: tangential) using:
                x ∈ [z_in, z_in + chord_axial]
            and repeats it periodically in y by pitch.

        Parameters
        ----------
        fig, ax : matplotlib Figure and Axes, optional
            Existing figure/axes. If None, a new one is created.
        n_points : int
            Number of points along the blade surface curve.
        title : str or None
            Optional title. If None, a default including the row name
            and component_type is used.

        Returns
        -------
        fig, ax
        """

        comp_type = str(getattr(self, "component_type", "")).lower()
        g = self.geometry

        def _normalize_thickness_model(model):
            tm = str(model).strip().lower()
            if tm in ("b_spline", "bspline", "quartic_bspline"):
                return "B_spline"
            if tm == "denton":
                return "Denton"
            return "NACA"

        def _resample_polyline(x, y, n_out):
            x = jnp.asarray(x, dtype=jnp.float64)
            y = jnp.asarray(y, dtype=jnp.float64)
            n_out = int(max(2, n_out))
            if x.size < 3 or n_out <= 2:
                return x, y
            ds = jnp.hypot(jnp.diff(x), jnp.diff(y))
            s = jnp.concatenate((jnp.array([0.0], dtype=jnp.float64), jnp.cumsum(ds)))
            s_end = float(s[-1])
            if s_end <= 1e-15:
                return x, y
            su = jnp.linspace(0.0, s_end, n_out)
            xu = jnp.interp(su, s, x)
            yu = jnp.interp(su, s, y)
            return xu, yu

        def _split_cartesian_contour_to_segments(x, y, n_lower):
            n_lower = int(n_lower)
            n_te = int(x.size - 2 * n_lower)
            if n_te < 0:
                raise ValueError(
                    f"Unexpected contour size for axial row: len={x.size}, n_lower={n_lower}."
                )
            x_lower = x[:n_lower]
            y_lower = y[:n_lower]
            x_te = x[n_lower : n_lower + n_te][::-1]
            y_te = y[n_lower : n_lower + n_te][::-1]
            x_upper = x[-n_lower:][::-1]
            y_upper = y[-n_lower:][::-1]
            return x_lower, y_lower, x_te, y_te, x_upper, y_upper

        # -------------------------------------------------
        # Create axes if needed
        # -------------------------------------------------
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("Meridional")
            ax.set_ylabel("Tangential")

        # -------------------------------------------------
        # RADIAL CASCADE: circular tiling (r–θ → x–y)
        # -------------------------------------------------
        if comp_type == "radial_cascade":
            camberline_type = g["camberline_type"]
            thickness_model = _normalize_thickness_model(g.get("thickness_model", "NACA"))
            r_in = g["radius_mean_in"]
            r_out = g["radius_mean_out"]
            N_blades = int(g["N_blades"])

            metal_angle_in = g["metal_angle_in"]
            metal_angle_out = g["metal_angle_out"]
            theta0 = g.get("theta0", 0.0)

            loc_max = g["maximum_thickness_location_fraction"]
            t_max = g["maximum_thickness"]
            t_te = g["trailing_edge_thickness"]
            wedge_trailing = jnp.deg2rad(g["trailing_edge_wedge_angle"])
            radius_leading = g["leading_edge_radius"]
            thickness_leading = float(g.get("leading_edge_thickness", 0.0))
            denton_shape_exp = float(
                g.get(
                    "denton_thickness_shape_exponent",
                    g.get("denton_tk_typ", 2.0),
                )
            )

            tm = str(thickness_model).strip().lower()
            use_tblade3_strategy = (
                str(camberline_type) == "curvature_based"
                and tm in ("b_spline", "bspline", "quartic_bspline")
            )
            radial_builder = (
                bp.compute_blade_coordinates_radial_segments_tblade3
                if use_tblade3_strategy
                else bp.compute_blade_coordinates_radial_segments
            )
            x_lower, y_lower, x_te, y_te, x_upper, y_upper, _, _ = (
                radial_builder(
                    camberline_type=camberline_type,
                    r1=r_in,
                    r2=r_out,
                    metal_angle1=jnp.deg2rad(metal_angle_in),
                    metal_angle2=jnp.deg2rad(metal_angle_out),
                    theta0=jnp.deg2rad(theta0),
                    loc_max=loc_max,
                    thickness_max=t_max,
                    thickness_trailing=t_te,
                    wedge_trailing=wedge_trailing,
                    radius_leading=radius_leading,
                    N_points=n_points,
                    thickness_model=thickness_model,
                    thickness_leading=thickness_leading,
                    thickness_shape_exponent=denton_shape_exp,
                )
            )

            # Plotting-only densification to avoid visible polyline kinks.
            x_lower, y_lower = _resample_polyline(x_lower, y_lower, n_points)
            x_upper, y_upper = _resample_polyline(x_upper, y_upper, n_points)
            x_te, y_te = _resample_polyline(x_te, y_te, max(80, n_points // 3))

            # Draw inlet and outlet circles
            theta = jnp.linspace(0.0, 2.0 * jnp.pi, 300)
            ax.plot(r_in * jnp.cos(theta), r_in * jnp.sin(theta), "k-", linewidth=0.5)
            ax.plot(r_out * jnp.cos(theta), r_out * jnp.sin(theta), "k-", linewidth=0.5)

            # Tile the blade around the circumference
            d_theta = 2.0 * jnp.pi / N_blades
            for i in range(N_blades):
                th = d_theta * i
                xl, yl = bp.rotate_counterclockwise_2D(x_lower, y_lower, th)
                xu, yu = bp.rotate_counterclockwise_2D(x_upper, y_upper, th)
                xte, yte = bp.rotate_counterclockwise_2D(x_te, y_te, th)
                ax.plot(
                    xl,
                    yl,
                    lw=1.1,
                    color="black",
                    label=self.name if i == 0 else None,
                )
                ax.plot(xu, yu, lw=1.1, color="black")
                ax.plot(xte, yte, lw=1.1, color="black")

            if title is None:
                title = f"Radial cascade - blade-to-blade"
            ax.set_title(title)

            fig.tight_layout(pad=1.0)
            return fig, ax

        # -------------------------------------------------
        # AXIAL CASCADE: x (axial) vs y (tangential), using z_in as start
        # -------------------------------------------------
        if comp_type == "axial_cascade":
            camberline_type = g["camberline_type"]
            thickness_model = _normalize_thickness_model(g.get("thickness_model", "NACA"))

            loc_max = g["maximum_thickness_location_fraction"]
            t_max = g["maximum_thickness"]
            t_te = g["trailing_edge_thickness"]
            wedge_trailing = jnp.deg2rad(g["trailing_edge_wedge_angle"])
            radius_leading = g["leading_edge_radius"]
            thickness_leading = float(g.get("leading_edge_thickness", 0.0))
            denton_shape_exp = float(
                g.get(
                    "denton_thickness_shape_exponent",
                    g.get("denton_tk_typ", 2.0),
                )
            )

            beta1_deg = g.get("metal_angle_in", g.get("leading_edge_angle"))
            beta2_deg = g.get("metal_angle_out", g.get("gauging_angle"))
            if beta1_deg is None or beta2_deg is None:
                raise KeyError(
                    "Axial cascade requires inlet/outlet metal angles. "
                    "Expected 'metal_angle_in/out' or 'leading_edge_angle/gauging_angle'."
                )
            beta1 = jnp.deg2rad(beta1_deg)
            beta2 = jnp.deg2rad(beta2_deg)

            chord_axial = g.get("chord_axial", g.get("meridional_chord", g.get("chord")))
            if chord_axial is None:
                raise KeyError(
                    "Axial cascade requires one of 'chord_axial', 'meridional_chord', or 'chord'."
                )
            pitch = g["pitch"]

            # Use the given z_in from geometry as axial start
            x1 = g["z_in"]
            y1 = 0.0

            tm = str(thickness_model).strip().lower()
            use_tblade3_strategy = (
                str(camberline_type) == "curvature_based"
                and tm in ("b_spline", "bspline", "quartic_bspline")
            )
            if use_tblade3_strategy:
                x_lower, y_lower, x_te, y_te, x_upper, y_upper, _, _ = (
                    bp.compute_blade_coordinates_cartesian_segments_tblade3(
                        camberline_type=camberline_type,
                        x1=x1,
                        y1=y1,
                        beta1=beta1,
                        beta2=beta2,
                        chord_ax=chord_axial,
                        loc_max=loc_max,
                        thickness_max=t_max,
                        thickness_trailing=t_te,
                        wedge_trailing=wedge_trailing,
                        radius_leading=radius_leading,
                        N_points=n_points,
                        thickness_model=thickness_model,
                        thickness_leading=thickness_leading,
                        thickness_shape_exponent=denton_shape_exp,
                    )
                )
            else:
                x_b, y_b, _, _ = bp.compute_blade_coordinates_cartesian(
                    camberline_type,
                    x1=x1,
                    y1=y1,
                    beta1=beta1,
                    beta2=beta2,
                    chord_ax=chord_axial,
                    loc_max=loc_max,
                    thickness_max=t_max,
                    thickness_trailing=t_te,
                    wedge_trailing=wedge_trailing,
                    radius_leading=radius_leading,
                    N_points=n_points,
                    thickness_model=thickness_model,
                    thickness_leading=thickness_leading,
                    thickness_shape_exponent=denton_shape_exp,
                )
                x_lower, y_lower, x_te, y_te, x_upper, y_upper = (
                    _split_cartesian_contour_to_segments(x_b, y_b, int(n_points))
                )

            x_lower, y_lower = _resample_polyline(x_lower, y_lower, n_points)
            x_upper, y_upper = _resample_polyline(x_upper, y_upper, n_points)
            x_te, y_te = _resample_polyline(x_te, y_te, max(80, n_points // 2))

            # Periodic copies in tangential direction
            for k in range(-1, 4):
                label = self.name if k == 0 else None
                ax.plot(
                    x_lower,
                    y_lower + k * pitch,
                    lw=1.2,
                    color="black",
                    label=label,
                )
                ax.plot(x_upper, y_upper + k * pitch, lw=1.2, color="black")
                ax.plot(x_te, y_te + k * pitch, lw=1.2, color="black")

            if title is None:
                title = f"Axial cascade - blade-to-blade"
            ax.set_title(title)

            fig.tight_layout(pad=1.0)
            return fig, ax

        raise ValueError(
            f"BladeRow.plot_meridional_tangential: unsupported component_type='{self.component_type}'"
        )
