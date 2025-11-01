# import numpy as np
import pandas as pd
import CoolProp as cp
from scipy.linalg import solve
from scipy.optimize._numdiff import approx_derivative

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .. import math
from .. import utilities as utils
from . import loss_model as lm
from . import deviation_model as dm
from . import choking_criterion as cm

import turboflow as tf
# from ..properties import perfect_gas_props_custom_jvp as perfect_gas_props
from ..properties import perfect_gas_props

import jaxprop as jxp
import jaxprop.perfect_gas as pg

# List of valid options
BLOCKAGE_MODELS = ["flat_plate_turbulent"]


# ---------------------------------------------------------------------
# NEW: Geometry adapter (per-row -> per-cascade arrays) and validation
# ---------------------------------------------------------------------

def build_array_geometry(full_by_name, row_order):
    """
    Convert {row_name: full_row_dict} (from geometry pipeline) to the
    array-shaped geometry dict expected by this flow model.

    - Sets `pitch = pitch_mean`, `height = height_mean`.
    - Sets `opening = abs(throat_opening)` per cascade (for blockage).
    - Passes through leading-edge fields and throat/areas.
    - Requires `row_order` to define cascade sequence (e.g. ["stator_1","rotor_1",...]).

    Returns:
        geometry_dict with list/array values for each cascade key plus:
            number_of_cascades, number_of_stages
    """
    rows = [full_by_name[name] for name in row_order]
    n = len(rows)
    n_stages = rows[0].get("number_of_stages", 0)

    def col(key, default=None, post=None):
        vals = [(r.get(key, default)) for r in rows]
        if post is not None:
            vals = [post(v) for v in vals]
        return vals

    geom = {
        # required top-level
        "number_of_cascades": n,
        "number_of_stages": n_stages,

        # radii & areas
        "radius_mean_in":     col("radius_mean_in"),
        "radius_mean_out":    col("radius_mean_out"),
        "radius_mean_throat": col("radius_mean_throat"),
        "A_in":               col("A_in"),
        "A_out":              col("A_out"),
        "A_throat":           col("A_throat"),

        # blade geometry
        "chord":              col("chord"),
        "meridional_chord":   col("meridional_chord"),
        "stagger_angle":      col("stagger_angle"),
        "pitch":              col("pitch"),            # == pitch_mean (provided by geometry module)
        "height":             col("height"),           # == height_mean (provided by geometry module)

        # throat opening (ensure positive for blockage model)
        "opening":            col("throat_opening", default=None, post=(lambda x: float(abs(x)) if x is not None else None)),

        # loss/deviation extras
        "leading_edge_angle":        col("leading_edge_angle"),
        "leading_edge_wedge_angle":  col("leading_edge_wedge_angle"),
        "leading_edge_diameter":     col("leading_edge_diameter"),
        "cascade_type":              col("cascade_type"),
    }
    return geom


def _is_array_geometry(geometry: dict) -> bool:
    """Heuristic: treat as array-geometry if it declares number_of_cascades."""
    return "number_of_cascades" in geometry and isinstance(geometry["number_of_cascades"], (int, jnp.ndarray))


def _validate_geometry_cascade(i, geom_i: dict, require_throat: bool):
    """Minimal checks to fail fast with a clear message."""
    required = [
        "radius_mean_in", "radius_mean_out", "A_in", "A_out",
        "chord", "meridional_chord", "pitch", "height",
        "stagger_angle", "cascade_type", "leading_edge_angle"
    ]
    missing = [k for k in required if geom_i.get(k, None) is None]
    if missing:
        raise ValueError(f"[cascade {i+1}] Missing required geometry keys: {missing}")

    # Benner incidence uses theta_out = arccos(A_throat/A_out) and needs leading edge wedge & diameter.
    if require_throat:
        throat_missing = [k for k in ["A_throat", "leading_edge_wedge_angle", "leading_edge_diameter"] if geom_i.get(k, None) is None]
        if throat_missing:
            raise ValueError(
                f"[cascade {i+1}] Benner/K–O incidence requires {throat_missing}. "
                f"Ensure YAML provides `throat_location_fraction`, `leading_edge_wedge_angle` (or `leading_edge_wedge`), "
                f"and `leading_edge_radius` (or `_fraction`)."
            )
    # opening used by blockage model
    if geom_i.get("opening", None) is None:
        # allow if blockage_model is None (handled later), but warn if using correlation
        pass


def _slice_geometry_for_cascade(geometry_arrays: dict, i: int, number_of_cascades: int) -> dict:
    """Take i-th scalar values from an array-geometry dict."""
    out = {}
    for key, values in geometry_arrays.items():
        if key in ["number_of_cascades", "number_of_stages", "interspace_area_ratio"]:
            continue
        # Assume list/array-like with length = number_of_cascades
        v = values[i] if isinstance(values, (list, tuple, jnp.ndarray)) else values
        out[key] = v
    return out


# ---------------------------------------------------------------------
# Flow model
# ---------------------------------------------------------------------

def evaluate_axial_turbine(
    variables,
    boundary_conditions,
    geometry,
    fluid,
    model_options,
    reference_values,
    row_order=None,          # NEW: optional sequence of row names if geometry is per-row dict
    benner_requires_throat=True,  # NEW: guard to enforce throat fields when using Benner/K–O incidence
):
    """
    Compute the performance of an axial (incl. radial-outflow cascade-by-cascade) turbine.

    Now supports:
      - geometry passed as *array-geometry* (original behavior), or
      - geometry passed as *per-row dict* {row_name: full_row} + row_order -> auto-adapted.
    """

    # --- Adapt geometry if needed ---
    if not _is_array_geometry(geometry):
        if row_order is None:
            raise ValueError(
                "Geometry appears to be per-row (no 'number_of_cascades'). "
                "Provide `row_order` (e.g., ['stator_1','rotor_1',...]) so it can be adapted, "
                "or pass array-geometry directly."
            )
        geometry = build_array_geometry(geometry, row_order)

    # Load geometry counters
    number_of_cascades = int(geometry["number_of_cascades"])
    number_of_stages = int(geometry.get("number_of_stages", 0))

    # Load boundary conditions
    h0_in = boundary_conditions["h0_in"]
    s_in = boundary_conditions["s_in"]
    alpha_in = boundary_conditions["alpha_in"]
    angular_speed = boundary_conditions["omega"]

    # Load reference_values
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Initialize results structure
    results = {
        "planes": [],
        "cascades": [],
        "stage": [],
        "overall": {},
        "geometry": geometry,
        "reference_values": reference_values,
        "boundary_conditions": boundary_conditions,
    }

    # initialize residual arrays
    residuals = {}

    # Rename turbine inlet velocity
    v_in = variables["v_in"] * v0

    # IMPORTANT: Loss model selection (to validate required fields)
    loss_model_name = model_options.get("loss_model", "")
    require_throat = benner_requires_throat and ("benner" in str(loss_model_name).lower())

    for i in range(number_of_cascades):
        # Pick rotor vs stator from geometry (instead of i % 2)
        ctype_i = str(geometry["cascade_type"][i]).lower() if isinstance(geometry["cascade_type"], (list, tuple, jnp.ndarray)) else str(geometry["cascade_type"]).lower()
        angular_speed_cascade = angular_speed if ("rotor" in ctype_i) else 0.0

        # Slice array-geometry to cascade scalars
        geometry_cascade = _slice_geometry_for_cascade(geometry, i, number_of_cascades)

        # Validate geometry fields early (fail-fast with actionable message)
        _validate_geometry_cascade(i, geometry_cascade, require_throat=require_throat)

        # Rename variables
        cascade = "_" + str(i + 1)
        w_out = variables["w_out" + cascade] * v0
        s_out = variables["s_out" + cascade] * s_range + s_min
        beta_out = variables["beta_out" + cascade] * angle_range + angle_min

        # Evaluate current cascade
        cascade_inlet_input = {
            "h0": h0_in,
            "s": s_in,
            "alpha": alpha_in,
            "v": v_in,
        }
        cascade_exit_input = {
            "w": w_out,
            "beta": beta_out,
            "s": s_out,
        }

        choking_input = {
            key.replace(cascade, ""): val
            for key, val in variables.items()
            if (("crit" in key) and (cascade in key)) or key == "v_crit_in"
        }

        cascade_residuals, inlet_plane, exit_plane, cascade_data = evaluate_cascade(
            cascade_inlet_input,
            cascade_exit_input,
            choking_input,
            fluid,
            geometry_cascade,
            angular_speed_cascade,
            model_options,
            reference_values,
        )

        # Add cascade residuals
        cascade_residuals = utils.add_string_to_keys(cascade_residuals, f"_{i+1}")
        residuals.update(cascade_residuals)

        # Calculate input of next cascade (interspace propagation)
        if i != number_of_cascades - 1:
            (
                h0_in,
                s_in,
                alpha_in,
                v_in,
            ) = evaluate_cascade_interspace(
                exit_plane["enthalpy0"],
                exit_plane["v_m"],
                exit_plane["v_t"],
                exit_plane["density"],
                geometry_cascade["radius_mean_out"],
                geometry_cascade["A_out"],
                exit_plane["blockage"],
                geometry["radius_mean_in"][i + 1],
                geometry["A_in"][i + 1],
                fluid,
            )

        # Store results
        results["planes"].append(inlet_plane)
        results["planes"].append(exit_plane)
        results["cascades"].append(cascade_data)

    # Merge lists of dicts into dict of arrays
    results["planes"] = tf.combine_to_dict_of_arrays(results["planes"])
    results["cascades"] = tf.combine_to_dict_of_arrays(results["cascades"])

    # Exit pressure residual
    p_calc = exit_plane["pressure"]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # Stage metrics
    results["stage"] = compute_stage_performance(results)

    # Overall metrics
    results["overall"] = compute_overall_performance(results, geometry)

    # Residuals & variables
    results["residuals"] = residuals
    results["independent_variables"] = variables

    # Retain only variables defined per cascade (array-geometry)
    geom_cascades = {
        key: value
        for key, value in geometry.items()
        if len(utils.ensure_iterable(value)) == number_of_cascades
    }
    results["geometry"] = geom_cascades

    return results


def evaluate_cascade(
    cascade_inlet_input,
    cascade_exit_input,
    choking_input,
    fluid,
    geometry,
    angular_speed,
    model_options,
    reference_values,
):
    """
    Evaluate the performance of a cascade (inlet → exit), compute choking, losses.
    """
    loss_model = model_options["loss_model"]
    deviation_model = model_options["deviation_model"]

    # Evaluate inlet plane
    inlet_plane = evaluate_cascade_inlet(
        cascade_inlet_input, fluid, geometry, angular_speed
    )

    # Evaluate exit plane
    cascade_exit_input["rothalpy"] = inlet_plane["rothalpy"]
    exit_plane, loss_dict = evaluate_cascade_exit(
        cascade_exit_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        model_options["blockage_model"],
        loss_model,
    )

    # Isentropic enthalpy drop to same pressure & inlet entropy
    props_is = fluid.get_state(jxp.PSmass_INPUTS, exit_plane["pressure"], inlet_plane["entropy"])
    dh_is = exit_plane["enthalpy"] - props_is["enthalpy"]

    # Critical (choking) evaluation
    residuals_critical, critical_state = cm.evaluate_choking(
        choking_input,
        inlet_plane,
        exit_plane,
        fluid,
        geometry,
        angular_speed,
        model_options,
        reference_values,
    )

    # Residuals
    mass_error_exit = inlet_plane["mass_flow"] - exit_plane["mass_flow"]
    residuals = {
        "loss_error_exit": exit_plane["loss_error"],
        "mass_error_exit": mass_error_exit / reference_values["mass_flow_ref"],
        **residuals_critical,
    }

    # Cascade summary
    cascade_data = {
        **loss_dict,
        **critical_state,
        "dh_s": dh_is,
        "incidence": inlet_plane["beta"] - geometry["leading_edge_angle"],
    }

    return residuals, inlet_plane, exit_plane, cascade_data


# @jax.jit
def evaluate_cascade_inlet(cascade_inlet_input, fluid, geometry, angular_speed):
    """
    Inlet plane: velocity triangles, thermo state, Re/Ma, mass flow.
    """
    # Inputs
    h0 = cascade_inlet_input["h0"]
    s  = cascade_inlet_input["s"]
    v  = cascade_inlet_input["v"]
    alpha = cascade_inlet_input["alpha"]

    # Geometry
    radius = geometry["radius_mean_in"]
    chord  = geometry["chord"]
    area   = geometry["A_in"]

    # Velocity triangles
    blade_speed = radius * angular_speed
    velocity_triangle = evaluate_velocity_triangle_in(blade_speed, v, alpha)
    w  = velocity_triangle["w"]
    w_m = velocity_triangle["w_m"]

    # Static state
    h = h0 - 0.5 * v**2
    static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

    rho = static_properties["d"]
    mu  = static_properties["mu"]
    a   = static_properties["a"]

    # Stagnation properties (absolute & relative)
    stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(relative_stagnation_properties, "0_rel")

    # Nondimensionals & flow rate
    Ma     = v / a
    Ma_rel = w / a
    Re     = rho * w * chord / mu
    m      = rho * w_m * area
    rothalpy = h0_rel - 0.5 * blade_speed**2

    loss_dict = {
        "loss_error": 0.0,
        "loss_profile": 0.0,
        "loss_incidence": 0.0,
        "loss_trailing": 0.0,
        "loss_secondary": 0.0,
        "loss_clearance": 0.0,
        "loss_total": 0.0,
    }

    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
        **loss_dict,
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": m,
        "rothalpy": rothalpy,
        "blockage": 0.0,
        "h_is": static_properties["h"],
    }
    return plane


# @jax.jit(static_argnums=(5,))
def evaluate_cascade_exit(
    cascade_exit_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    """
    Exit plane: velocity triangles, thermo state, Re/Ma, mass flow, losses.
    """
    # Exit inputs
    w     = cascade_exit_input["w"]
    beta  = cascade_exit_input["beta"]
    s     = cascade_exit_input["s"]
    rothalpy = cascade_exit_input["rothalpy"]

    # Geometry
    chord   = geometry["chord"]
    opening = geometry.get("opening", None)  # may be None (e.g., user omitted throat); handled below
    area    = geometry["A_out"]
    radius  = geometry["radius_mean_out"]

    # Velocity triangles
    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v   = velocity_triangle["v"]
    w_m = velocity_triangle["w_m"]

    # Static state
    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

    rho = static_properties["d"]
    mu  = static_properties["mu"]
    a   = static_properties["a"]

    # Stagnation properties (abs & rel)
    h0 = h + 0.5 * v**2
    stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(relative_stagnation_properties, "0_rel")

    # Nondimensionals
    Ma     = v / a
    Ma_rel = w / a
    Re     = rho * w * chord / mu
    rothalpy = h0_rel - 0.5 * blade_speed**2

    # Isentropic references (same relative h0, inlet s)
    relative_stagnation_isentropic_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"])
    relative_static_isentropic_properties     = fluid.get_state(jxp.PSmass_INPUTS, static_properties["p"], inlet_plane["entropy"])

    # Mass flow (with boundary-layer blockage)
    # Safe opening in case user omitted throat; if None, treat as no blockage.
    blockage_factor = compute_blockage_boundary_layer(
        blockage,
        Re,
        chord,
        opening if (opening is not None) else jnp.inf  # => zero blockage if no throat data
    )
    mass_flow = rho * w_m * area * (1 - blockage_factor)

    # Loss model inputs
    min_val = 1e-3
    loss_model_input = {
        "geometry": geometry,
        "loss_model": loss_model,
        "flow": {
            "p0_rel_in": inlet_plane["pressure0_rel"],
            "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
            "p_in": inlet_plane["pressure"],
            "p_out": static_properties["p"],
            "h_out": static_properties["h"],
            "beta_out": beta,
            "w_out" : w,
            "beta_in": inlet_plane["beta"],
            "Ma_rel_in": max(min_val, inlet_plane["Ma_rel"]),
            "Ma_rel_out": max(min_val, Ma_rel),
            "Re_in": max(min_val, inlet_plane["Re"]),
            "Re_out": max(min_val, Re),
            "gamma_out": static_properties["gamma"],
            "p0_rel_is" : relative_stagnation_isentropic_properties["p"],
            "h_is" : relative_static_isentropic_properties["h"],
        },
    }

    # Evaluate loss coefficients
    loss_dict = lm.evaluate_loss_model(loss_model, loss_model_input)

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
        "h_is" : relative_static_isentropic_properties["h"],
    }
    return plane, loss_dict


# @jax.jit
def evaluate_cascade_throat(
    cascade_throat_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    """
    Throat plane evaluation (optional path); kept consistent with exit.
    """
    w     = cascade_throat_input["w"]
    beta  = cascade_throat_input["beta"]
    s     = cascade_throat_input["s"]
    rothalpy = cascade_throat_input["rothalpy"]

    chord   = geometry["chord"]
    opening = geometry.get("opening", None)
    area    = geometry["A_throat"]
    radius  = geometry["radius_mean_throat"]

    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]

    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

    rho = static_properties["d"]
    mu  = static_properties["mu"]
    a   = static_properties["a"]

    h0 = h + 0.5 * v**2
    stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(relative_stagnation_properties, "0_rel")

    Ma     = v / a
    Ma_rel = w / a
    Re     = rho * w * chord / mu
    rothalpy = h0_rel - 0.5 * blade_speed**2

    relative_stagnation_isentropic_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"])
    relative_static_isentropic_properties     = fluid.get_state(jxp.PSmass_INPUTS, static_properties["pressure"], inlet_plane["entropy"])

    blockage_factor = compute_blockage_boundary_layer(
        blockage,
        Re,
        chord,
        opening if (opening is not None) else jnp.inf
    )
    mass_flow = rho * w * area * (1 - blockage_factor)

    min_val = 1e-3
    loss_dict = lm.evaluate_loss_model(
        loss_model,
        {
            "geometry": geometry,
            "loss_model": loss_model,
            "flow": {
                "p0_rel_in": inlet_plane["pressure0_rel"],
                "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
                "p_in": inlet_plane["pressure"],
                "p_out": static_properties["p"],
                "h_out": static_properties["h"],
                "beta_out": beta,
                "w_out" : w,
                "beta_in": inlet_plane["beta"],
                "Ma_rel_in": max(min_val, inlet_plane["Ma_rel"]),
                "Ma_rel_out": max(min_val, Ma_rel),
                "Re_in": max(min_val, inlet_plane["Re"]),
                "Re_out": max(min_val, Re),
                "gamma_out": static_properties["gamma"],
                "p0_rel_is" : relative_stagnation_isentropic_properties["p"],
                "h_is" : relative_static_isentropic_properties["h"],
            },
        }
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
        "h_is" : relative_static_isentropic_properties["h"],
    }
    return plane, loss_dict


# @jax.jit
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
    """
    Propagate exit conditions of cascade i to inlet of i+1 using simple
    invariants (h0, angular momentum) and area scaling with blockage.
    """
    # Assume no heat transfer
    h0_in = h0_exit

    # Angular momentum approx.
    v_t_in = v_t_exit * radius_exit / radius_inlet

    # Density variation negligible
    v_m_in = v_m_exit * area_exit / area_inlet * (1 - blockage_exit)

    v_in = jnp.sqrt(v_t_in**2 + v_m_in**2)
    alpha_in = math.arctand(v_t_in / v_m_in)

    # Thermodynamic state at inlet
    h_in = h0_in - 0.5 * v_in**2
    rho_in = rho_exit
    stagnation_properties = fluid.get_state(jxp.DmassHmass_INPUTS, rho_in, h_in)
    s_in = stagnation_properties["s"]

    return h0_in, s_in, alpha_in, v_in


# @jax.jit
def evaluate_velocity_triangle_in(blade_speed, v, alpha):
    # Absolute velocities
    v_t = v * math.sind(alpha)
    v_m = v * math.cosd(alpha)

    # Relative velocities
    w_t = v_t - blade_speed
    w_m = v_m
    w = jnp.sqrt(w_t**2 + w_m**2)

    # Relative flow angle
    beta = math.arctand(w_t / w_m)

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
    # Relative velocities
    w_t = w * math.sind(beta)
    w_m = w * math.cosd(beta)

    # Absolute velocities
    v_t = w_t + blade_speed
    v_m = w_m
    v   = jnp.sqrt(v_t**2 + v_m**2)

    # Absolute flow angle
    alpha = math.arctand(v_t / v_m)

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
def compute_blockage_boundary_layer(blockage_model, Re, chord, opening):
    r"""
    Blockage due to boundary layer displacement thickness.

    If `opening` is None or infinite, returns 0 for blockage.
    """
    # opening guard
    if opening is None or (jnp.isinf(opening)):
        return 0.0

    # ensure strictly positive to avoid divide-by-zero
    opening_safe = jnp.maximum(jnp.asarray(opening, dtype=jnp.float64), 1e-9)

    if blockage_model == BLOCKAGE_MODELS[0]:
        displacement_thickness = 0.048 / Re ** (1 / 5) * 0.9 * chord
        blockage_factor = 2 * displacement_thickness / opening_safe

    elif isinstance(blockage_model, (float, int)) and 0 <= blockage_model <= 1:
        blockage_factor = float(blockage_model)

    elif blockage_model is None:
        blockage_factor = 0.00

    else:
        raise ValueError(
            f"Invalid throat blockage option: '{blockage_model}'. "
            "Valid options are 'flat_plate_turbulent', a numeric value between 0 and 1, or None."
        )

    return blockage_factor


def compute_efficiency_breakdown(results):
    # (unchanged)
    h_out_s = results["reference_values"]["h_out_s"]
    h0_in = results["planes"]["enthalpy0"][0]
    h_out = results["planes"]["enthalpy"][-1]
    cascade = results["cascades"]
    number_of_cascades = results["geometry"]["number_of_cascades"]

    dhs_total = h_out - h_out_s
    dhs_sum = cascade["dh_s"].sum()
    correction = dhs_total / dhs_sum

    loss_types = ["profile", "incidence", "secondary", "clearance", "trailing"]
    breakdown = pd.DataFrame(columns=[f"efficiency_drop_{t}" for t in loss_types])

    for i in range(number_of_cascades):
        col_names = [f"loss_{t}" for t in loss_types]
        fracs = jnp.array([cascade[c][i] for c in col_names]) / cascade["loss_total"][i]
        dh_s = cascade["dh_s"][i]
        efficiency_drop = correction * dh_s / (h0_in - h_out_s)
        breakdown.loc[len(breakdown)] = (fracs * efficiency_drop).tolist()

    return breakdown


def compute_stage_performance(results):
    r"""
    Stage metrics (reaction per stage).
    """
    number_of_stages = results["geometry"]["number_of_stages"]
    if number_of_stages == 0:
        return {}

    h = results["planes"]["enthalpy"]
    R = jnp.array(
        [
            (h[i * 4 + 1] - h[i * 4 + 3]) / (h[i * 4] - h[i * 4 + 3])
            for i in range(number_of_stages)
        ]
    )
    stages = {"reaction": R}
    return stages


def compute_overall_performance(results, geometry):
    """
    Overall KPIs.
    """
    angular_speed = results["boundary_conditions"]["omega"]
    v0 = results["reference_values"]["v0"]
    h_out_s = results["reference_values"]["h_out_s"]
    d_out_s = results["reference_values"]["d_out_s"]

    p  = results["planes"]["pressure"]
    p0 = results["planes"]["pressure0"]
    h0 = results["planes"]["enthalpy0"]
    v_out = results["planes"]["v"][-1]
    u_out = results["planes"]["blade_speed"][-1]
    mass_flow = results["planes"]["mass_flow"][-1]
    exit_flow_angle = results["planes"]["alpha"][-1]
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
    specific_speed = (
        angular_speed * (mass_flow / d_out_s) ** 0.5 / ((h0_in - h_out_s) ** 0.75)
    )

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
        "specific_speed": specific_speed,
        "blade_jet_ratio_hub": jnp.array(angular_speed * geometry["radius_hub_out"][-1] / v0) if "radius_hub_out" in geometry else jnp.nan,
        "blade_jet_ratio_mean": jnp.array(angular_speed * geometry["radius_mean_out"][-1] / v0),
        "blade_jet_ratio_tip": jnp.array(angular_speed * geometry["radius_tip_out"][-1] / v0) if "radius_tip_out" in geometry else jnp.nan,
    }
    return overall
