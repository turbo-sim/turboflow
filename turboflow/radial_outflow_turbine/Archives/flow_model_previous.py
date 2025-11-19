import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .. import math
from .. import utilities as utils
from . import loss_model as lm
from . import choking_criterion as cm

import turboflow as tf
import jaxprop as jxp

# Valid options for blockage model
BLOCKAGE_MODELS = ["flat_plate_turbulent"]


# ============================================================
# Component-wise axial turbine evaluation (stator → interspace → rotor)
# ============================================================
# Replace with evaluate_turbomachine
def evaluate_axial_turbine_componentwise(
    variables,
    boundary_conditions,
    geom_list,  # list[dict], one per component
    fluid,
    reference_values,
    components=None,  # original YAML components (to read per-component model_options)
    model_options_global=None,  # global fallbacks
    benner_requires_throat=True,
):
    n_comp = len(geom_list)
    names = [c.get("name", f"component_{i+1}") for i, c in enumerate(geom_list)]
    ctypes = [c["cascade_type"] for c in geom_list]

    # Inlet (from BCs)
    h0_in = boundary_conditions["h0_in"]
    s_in = boundary_conditions["s_in"]
    alpha_in = boundary_conditions["alpha_in"]
    omega = boundary_conditions["omega"]

    # Reference scales
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    planes_seq = []
    cascades_seq = []
    residuals = {}

    # inlet absolute speed (scaled)
    v_in = variables["v_in"] * v0

    # Replace geom_list with components
    for i, g in enumerate(geom_list):
        name_i = names[i]
        ctype_i = str(ctypes[i]).lower()
        is_rotor = "rotor" in ctype_i
        omega_i = omega if is_rotor else 0.0

        # ---- normalize per-component model options (merge global + component, fill defaults)
        comp_opts_raw = {}
        if components and i < len(components):
            comp_opts_raw = components[i].get("model_options", {}) or {}

        mo_i = {}
        if model_options_global:
            mo_i.update(model_options_global)
        mo_i.update(comp_opts_raw)

        # required + sensible defaults
        if "loss_model" not in mo_i:
            raise ValueError(f"[{name_i}] model_options must include 'loss_model'.")
        mo_i.setdefault("loss_coefficient", "stagnation_pressure")
        mo_i.setdefault("deviation_model", "aungier")
        mo_i.setdefault("choking_criterion", "critical_mach_number")
        mo_i.setdefault("blockage_model", None)
        mo_i.setdefault("inlet_displacement_thickness_height_ratio", 0.011)

        require_throat = benner_requires_throat and (
            "benner" in str(mo_i["loss_model"]).lower()
        )

        # minimal geometry validation
        _validate_geometry_component(name_i, g, require_throat=require_throat)

        # ---- scaled solver variables for this component
        tag = f"_{i+1}"
        w_out = variables["w_out" + tag] * v0
        s_out = variables["s_out" + tag] * s_range + s_min
        beta_out = variables["beta_out" + tag] * angle_range + angle_min

        choking_input = {
            key.replace(tag, ""): val
            for key, val in variables.items()
            if (("crit" in key) and (tag in key)) or key == "v_crit_in"
        }

        # ---- inlet plane
        inlet_in = {"h0": h0_in, "s": s_in, "alpha": alpha_in, "v": v_in}
        inlet_plane = evaluate_cascade_inlet(inlet_in, fluid, g, omega_i)

        # ---- exit plane (+ losses)
        exit_in = {
            "w": w_out,
            "beta": beta_out,
            "s": s_out,
            "rothalpy": inlet_plane["rothalpy"],
        }
        exit_plane, loss_dict = evaluate_cascade_exit(
            exit_in,
            fluid,
            g,
            inlet_plane,
            omega_i,
            mo_i["blockage_model"],
            mo_i["loss_model"],  # <-- pass string or dict; loss module normalizes
        )

        # isentropic drop to same p & inlet s
        props_is = fluid.get_state(
            jxp.PSmass_INPUTS, exit_plane["pressure"], inlet_plane["entropy"]
        )
        dh_is = exit_plane["enthalpy"] - props_is["enthalpy"]

        # ---- choking / critical
        residuals_critical, critical_state = cm.evaluate_choking(
            choking_input,
            inlet_plane,
            exit_plane,
            fluid,
            g,
            omega_i,
            mo_i,
            reference_values,
        )

        # residuals for this cascade
        mass_error_exit = inlet_plane["mass_flow"] - exit_plane["mass_flow"]
        cascade_residuals = {
            "loss_error_exit": exit_plane["loss_error"],
            "mass_error_exit": mass_error_exit / reference_values["mass_flow_ref"],
            **residuals_critical,
        }
        residuals.update(utils.add_string_to_keys(cascade_residuals, tag))

        # per-component summary
        cascade_data = {
            **loss_dict,
            **critical_state,
            "dh_s": dh_is,
            "incidence": inlet_plane["beta"] - g["leading_edge_angle"],
            "name": name_i,
            "cascade_type": ctype_i,
        }

        planes_seq.append(inlet_plane)
        planes_seq.append(exit_plane)
        cascades_seq.append(cascade_data)

        # ---- interspace: stator -> rotor
        if i < n_comp - 1:
            next_ctype = str(ctypes[i + 1]).lower()
            if ("stator" in ctype_i) and ("rotor" in next_ctype):
                h0_in, s_in, alpha_in, v_in = evaluate_cascade_interspace(
                    exit_plane["enthalpy0"],
                    exit_plane["v_m"],
                    exit_plane["v_t"],
                    exit_plane["density"],
                    g["radius_mean_out"],
                    g["A_out"],
                    exit_plane["blockage"],
                    geom_list[i + 1]["radius_mean_in"],
                    geom_list[i + 1]["A_in"],
                    fluid,
                )
            else:
                h0_in = exit_plane["enthalpy0"]
                s_in = exit_plane["entropy0"]
                alpha_in = exit_plane["alpha"]
                v_in = exit_plane["v"]

    # collect arrays
    planes = tf.combine_to_dict_of_arrays(planes_seq)
    cascades = tf.combine_to_dict_of_arrays(cascades_seq)

    # outlet pressure residual
    p_calc = planes_seq[-1]["pressure"]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # stage & overall
    stage = compute_stage_performance_componentwise(planes, ctypes)
    overall = compute_overall_performance_componentwise(
        planes, boundary_conditions, reference_values, geom_list[-1]
    )

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


# --------------------
# Validation helpers
# --------------------


def _validate_geometry_component(name, g, require_throat: bool):
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


# -------------------------
# Single-cascade primitives
# -------------------------


def evaluate_cascade_inlet(cascade_inlet_input, fluid, geometry, angular_speed):
    """Inlet plane: velocity triangles, thermo state, Re/Ma, mass flow."""
    h0 = cascade_inlet_input["h0"]
    s = cascade_inlet_input["s"]
    v = cascade_inlet_input["v"]
    alpha = cascade_inlet_input["alpha"]

    radius = geometry["radius_mean_in"]
    chord = geometry["chord"]
    area = geometry["A_in"]

    blade_speed = radius * angular_speed
    velocity_triangle = evaluate_velocity_triangle_in(blade_speed, v, alpha)
    w = velocity_triangle["w"]
    w_m = velocity_triangle["w_m"]

    h = h0 - 0.5 * v**2
    static_properties = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)

    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

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
    m = rho * w_m * area
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


def evaluate_cascade_exit(
    cascade_exit_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    """Exit plane: velocity triangles, thermo state, Re/Ma, mass flow, losses."""
    w = cascade_exit_input["w"]
    beta = cascade_exit_input["beta"]
    s = cascade_exit_input["s"]
    rothalpy = cascade_exit_input["rothalpy"]

    chord = geometry["chord"]
    opening = geometry.get("opening", None)
    area = geometry["A_out"]
    radius = geometry["radius_mean_out"]

    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]
    w_m = velocity_triangle["w_m"]

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

    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    rothalpy = h0_rel - 0.5 * blade_speed**2

    relative_stagnation_isentropic_properties = fluid.get_state(
        jxp.HmassSmass_INPUTS, h0_rel, inlet_plane["entropy"]
    )
    relative_static_isentropic_properties = fluid.get_state(
        jxp.PSmass_INPUTS, static_properties["p"], inlet_plane["entropy"]
    )

    blockage_factor = compute_blockage_boundary_layer(
        blockage, Re, chord, opening if (opening is not None) else jnp.inf
    )
    mass_flow = rho * w_m * area * (1 - blockage_factor)

    min_val = 1e-3

    # grab just the loss-model options; fall back to whole per_opts for robustness
    # loss_model_options = model_options.get("loss_model", model_options)

    loss_dict = lm.evaluate_loss_model(
        loss_model,
        {
            "geometry": geometry,
            # "loss_model": loss_model,
            "flow": {
                "p0_rel_in": inlet_plane["pressure0_rel"],
                "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
                "p_in": inlet_plane["pressure"],
                "p_out": static_properties["p"],
                "h_out": static_properties["h"],
                "beta_out": beta,
                "w_out": w,
                "beta_in": inlet_plane["beta"],
                "Ma_rel_in": max(min_val, inlet_plane["Ma_rel"]),
                "Ma_rel_out": max(min_val, Ma_rel),
                "Re_in": max(min_val, inlet_plane["Re"]),
                "Re_out": max(min_val, Re),
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

    # grab just the loss-model options; fall back to whole per_opts for robustness
    # loss_model_options = model_options.get("loss_model", model_options)

    loss_dict = lm.evaluate_loss_model(
        loss_model,
        {
            "geometry": geometry,
            # "loss_model": loss_model,
            "flow": {
                "p0_rel_in": inlet_plane["pressure0_rel"],
                "p0_rel_out": relative_stagnation_properties["pressure0_rel"],
                "p_in": inlet_plane["pressure"],
                "p_out": static_properties["p"],
                "h_out": static_properties["h"],
                "beta_out": beta,
                "w_out": w,
                "beta_in": inlet_plane["beta"],
                "Ma_rel_in": max(min_val, inlet_plane["Ma_rel"]),
                "Ma_rel_out": max(min_val, Ma_rel),
                "Re_in": max(min_val, inlet_plane["Re"]),
                "Re_out": max(min_val, Re),
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


def evaluate_velocity_triangle_in(blade_speed, v, alpha):
    v_t = v * math.sind(alpha)
    v_m = v * math.cosd(alpha)
    w_t = v_t - blade_speed
    w_m = v_m
    w = jnp.sqrt(w_t**2 + w_m**2)
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


def evaluate_velocity_triangle_out(blade_speed, w, beta):
    w_t = w * math.sind(beta)
    w_m = w * math.cosd(beta)
    v_t = w_t + blade_speed
    v_m = w_m
    v = jnp.sqrt(v_t**2 + v_m**2)
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


def compute_blockage_boundary_layer(blockage_model, Re, chord, opening):
    """Boundary-layer blockage. If `opening` is None/inf → 0."""
    if opening is None or (jnp.isinf(opening)):
        return 0.0
    opening_safe = jnp.maximum(jnp.asarray(opening, dtype=jnp.float64), 1e-9)

    if blockage_model == BLOCKAGE_MODELS[0]:
        displacement_thickness = 0.048 / Re ** (1 / 5) * 0.9 * chord
        blockage_factor = 2 * displacement_thickness / opening_safe
    elif isinstance(blockage_model, (float, int)) and 0 <= blockage_model <= 1:
        blockage_factor = float(blockage_model)
    elif blockage_model is None:
        blockage_factor = 0.0
    else:
        raise ValueError(
            f"Invalid throat blockage option: '{blockage_model}'. "
            "Valid: 'flat_plate_turbulent', numeric in [0,1], or None."
        )
    return blockage_factor


# ==========================
# KPIs (stage & overall)
# ==========================


def compute_stage_performance_componentwise(planes, component_types):
    """
    Stage reaction based on plane ordering: for each stage,
      stator_in, stator_out, rotor_in, rotor_out  (4 planes)
    """
    # derive number of stages by counting stator→rotor pairs
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
    # planes list is [stator_in, stator_out, rotor_in, rotor_out, ...]
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
    """Overall KPIs using last component’s geometry for blade-jet ratios."""
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
        "specific_speed": angular_speed
        * (mass_flow / reference_values["d_out_s"]) ** 0.5
        / ((h0_in - h_out_s) ** 0.75),
        # blade-jet ratios using last component’s radii if available
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
    return overall
