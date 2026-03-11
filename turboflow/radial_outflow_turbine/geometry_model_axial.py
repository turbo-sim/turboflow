from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx

from turboflow import math
from turboflow import utilities as utils

# from turboflow.radial_outflow_turbine import blade_parametrization as bp
from turboflow.blade_parametrization import blade_parametrization_update as bp

# ==============================
# Required keys for AXIAL CASCADE geometry
# ==============================
REQUIRED_AXIAL_GEOM_KEYS = {
    "cascade_type",                     # "stator" / "rotor"
    "camberline_type",
    "thickness_model",
    "N_blades",
    "radius_mean_in",
    "radius_mean_out",
    "metal_angle_in",                   # deg
    "metal_angle_out",                  # deg
    "maximum_thickness",                # m
    "blade_height_in",                  # m
    "blade_height_out",                 # m
    "maximum_thickness_location_fraction",
    "leading_edge_wedge_angle",         # deg
    "leading_edge_radius",              # m
    "trailing_edge_radius",             # m
    "trailing_edge_wedge_angle",        # deg
    "throat_location_fraction",
    "chord_axial",                      # m
    "tip_clearance",                    # m
}
OPTIONAL_AXIAL_GEOM_KEYS = {
    # Optional thickness controls used by the Denton route.
    "leading_edge_thickness",
    "denton_thickness_shape_exponent",
    "denton_tk_typ",  # alias for denton_thickness_shape_exponent
}

VALID_CASCADE_TYPES = {"stator", "rotor"}
VALID_COMPONENT_TYPES = {"axial_cascade", "vaneless_channel"}


# ==============================
# Adapters / Input normalization
# ==============================
def _extract_components(obj):
    if isinstance(obj, dict) and "components" in obj:
        return obj["components"]
    if isinstance(obj, list):
        return obj
    raise TypeError(
        "Input must be a YAML dict with 'components' or a list of components."
    )



# ==============
# Validation
# ==============
def _validate_axial_cascade_component(c, index=0):
    """
    Validate one component entry in the *raw geometry* format
    for component_type == 'axial_cascade'.
    """
    if "geometry" not in c or not isinstance(c["geometry"], dict):
        raise ValueError(f"Component #{index+1} missing 'geometry' dict.")

    name = c.get("name", f"component_{index+1}")
    geom = c["geometry"]

    # STRICT required keys + controlled optional keys.
    utils.validate_keys(
        geom,
        REQUIRED_AXIAL_GEOM_KEYS,
        REQUIRED_AXIAL_GEOM_KEYS | OPTIONAL_AXIAL_GEOM_KEYS,
    )

    # cascade_type must be stator or rotor
    ct = geom["cascade_type"]
    if ct not in VALID_CASCADE_TYPES:
        raise ValueError(
            f"Component '{name}' has invalid cascade_type='{ct}'. "
            f"Only {sorted(VALID_CASCADE_TYPES)} allowed."
        )

    # Numeric checks
    for k, v in geom.items():
        if k in ("cascade_type", "camberline_type", "thickness_model"):
            continue

        if not isinstance(v, (int, float)) and not jnp.isscalar(v):
            raise TypeError(
                f"Component '{name}': parameter '{k}' must be a numeric scalar (int/float). "
                f"Got {type(v)}."
            )

        val = float(v)
        # allow angles (keys containing 'angle') to be negative
        if ("angle" not in k) and not (val >= 0.0):
            raise ValueError(
                f"Component '{name}': parameter '{k}' must be non-negative. Got {v}."
            )

    if int(geom["N_blades"]) < 1:
        raise ValueError(
            f"Component '{name}': N_blades must be >= 1. Got {geom['N_blades']}."
        )


def _validate_vaneless_channel_component(c, index=0):
    """
    Minimal structural check for an 'vaneless_channel' (vaneless channel).
    We only require that a geometry dict exists; its contents are not prescribed here.
    """
    if "geometry" not in c or not isinstance(c["geometry"], dict):
        raise ValueError(f"Vaneless_channel component #{index+1} missing 'geometry' dict.")
    # If later you want strict keys for vaneless_channel, add them here.

def _validate_interspace_component(c, index=0):
    """
    Minimal structural check for an 'vaneless_channel' (vaneless channel).
    We only require that a geometry dict exists; its contents are not prescribed here.
    """
    if "geometry" not in c or not isinstance(c["geometry"], dict):
        raise ValueError(f"Vaneless_channel component #{index+1} missing 'geometry' dict.")
    # If later you want strict keys for vaneless_channel, add them here.


def _validate_single_component(c, index=0):
    """
    Dispatch validation based on component_type.
    """
    name = c.get("name", f"component_{index+1}")
    ctype = c.get("component_type", None)
    if ctype is None:
        raise ValueError(f"Component '{name}' missing 'component_type'.")

    if ctype not in VALID_COMPONENT_TYPES:
        raise ValueError(
            f"Component '{name}' has unsupported component_type='{ctype}'. "
            f"Supported: {sorted(VALID_COMPONENT_TYPES)}."
        )

    if ctype == "axial_cascade":
        _validate_axial_cascade_component(c, index)
    elif ctype == "interspace":
        _validate_interspace_component(c, index)
    elif ctype == "vaneless_channel":
        _validate_vaneless_channel_component(c, index)

        

def validate_turbine_geometry(yaml_or_components, display=False):
    components = _extract_components(yaml_or_components)
    for i, comp in enumerate(components):
        _validate_single_component(comp, i)

    msg = "All component geometry entries are valid."
    if display:
        print(msg)
    return msg



# =================
# Core calculations
# =================
def calculate_throat_radius(radius_in, radius_out, throat_location_fraction):
    """JAX-friendly linear interpolation for throat radius."""
    rin = jnp.asarray(radius_in, dtype=jnp.float64)
    rout = jnp.asarray(radius_out, dtype=jnp.float64)
    frac = jnp.asarray(throat_location_fraction, dtype=jnp.float64)
    return (1.0 - frac) * rin + frac * rout



# @partial(
#     jax.jit,
#     static_argnames=(
#         "camberline_type_id",
#         "N_cam_points",
#     ),
# )
@eqx.filter_jit
def _compute_axial_cascade_geometry_jit(
    camberline_type_id: int,
    N_blades,
    radius_mean_in,
    radius_mean_out,
    blade_height_in,
    blade_height_out,
    metal_angle_in_deg,
    metal_angle_out_deg,
    maximum_thickness,
    maximum_thickness_location_fraction,
    leading_edge_wedge_angle,
    leading_edge_radius,
    trailing_edge_radius,
    trailing_edge_wedge_angle,
    throat_location_fraction,
    chord_axial,
    tip_clearance,
    z_in,
    N_cam_points: int = 64,
):
    """JIT-friendly core axial cascade geometry calculator (numbers only)."""
    trailing_edge_thickness = trailing_edge_radius * 2.0

    radius_hub_in = radius_mean_in - 0.5 * blade_height_in
    radius_tip_in = radius_mean_in + 0.5 * blade_height_in

    radius_hub_out = radius_mean_out - 0.5 * blade_height_out
    radius_tip_out = radius_mean_out + 0.5 * blade_height_out

    metal_angle_in_rad = jnp.deg2rad(metal_angle_in_deg)
    metal_angle_out_rad = jnp.deg2rad(metal_angle_out_deg)

    u = jnp.linspace(0.0, 1.0, N_cam_points)
    _, _, _, stagger_rad, chord = bp.compute_camberline_cartesian_by_id(
        camberline_type_id,
        0.0,
        0.0,
        metal_angle_in_rad,
        metal_angle_out_rad,
        chord_axial,
        u,
    )
    stagger_angle = jnp.rad2deg(stagger_rad)

    radius_mean_in = 0.5 * (radius_tip_in + radius_hub_in)
    radius_mean_out = 0.5 * (radius_tip_out + radius_hub_out)

    radius_hub_throat = calculate_throat_radius(
        radius_hub_in, radius_hub_out, throat_location_fraction
    )
    radius_tip_throat = calculate_throat_radius(
        radius_tip_in, radius_tip_out, throat_location_fraction
    )
    radius_mean_throat = 0.5 * (radius_tip_throat + radius_hub_throat)

    radius_shroud_in = radius_tip_in + tip_clearance
    radius_shroud_out = radius_tip_out + tip_clearance
    radius_shroud_throat = calculate_throat_radius(
        radius_shroud_in, radius_shroud_out, throat_location_fraction
    )

    height_in = radius_tip_in - radius_hub_in
    height_out = radius_tip_out - radius_hub_out
    height_throat = radius_tip_throat - radius_hub_throat
    height = 0.5 * (height_in + height_out)

    # N_blades = jnp.maximum(N_blades, 1)
    pitch = 2.0 * jnp.pi * radius_mean_throat / N_blades
    pitch_angle = 2.0 * jnp.pi / N_blades
    pitch_in = 2.0 * jnp.pi * radius_mean_in / N_blades
    pitch_out = 2.0 * jnp.pi * radius_mean_out / N_blades

    gauging_angle = metal_angle_out_deg

    A_in = jnp.pi * (radius_tip_in**2 - radius_hub_in**2)
    A_out = jnp.pi * (radius_tip_out**2 - radius_hub_out**2)
    A_throat = A_out * math.cosd(gauging_angle)

    opening = A_throat * pitch / (
        2.0 * jnp.pi * radius_mean_throat * jnp.maximum(height_throat, 1e-12)
    )

    meridional_chord = chord * math.cosd(stagger_angle)
    flaring_angle = math.arctand(
        (height_out - height_in)
        / jnp.maximum(meridional_chord, 1e-12)
        / 2.0
    )

    aspect_ratio = height / chord
    pitch_chord_ratio = pitch / chord
    solidity = 1.0 / pitch_chord_ratio
    hub_tip_ratio_in = radius_hub_in / radius_tip_in
    hub_tip_ratio_out = radius_hub_out / radius_tip_out
    hub_tip_ratio_throat = radius_hub_throat / jnp.maximum(radius_tip_throat, 1e-12)
    maximum_thickness_chord_ratio = maximum_thickness / chord
    trailing_edge_thickness_opening_ratio = trailing_edge_thickness / jnp.maximum(
        opening, 1e-12
    )
    tip_clearance_height_ratio = tip_clearance / jnp.maximum(height, 1e-12)
    leading_edge_diameter = 2.0 * leading_edge_radius
    leading_edge_diameter_chord_ratio = leading_edge_diameter / chord
    leading_edge_angle = metal_angle_in_deg

    return {
        "N_blades": N_blades,
        "radius_mean_in": radius_mean_in,
        "radius_mean_out": radius_mean_out,
        "metal_angle_in": metal_angle_in_deg,
        "metal_angle_out": metal_angle_out_deg,
        "maximum_thickness": maximum_thickness,
        "trailing_edge_thickness": trailing_edge_thickness,
        "blade_height_in": blade_height_in,
        "blade_height_out": blade_height_out,
        "maximum_thickness_location_fraction": maximum_thickness_location_fraction,
        "leading_edge_wedge_angle": leading_edge_wedge_angle,
        "leading_edge_radius": leading_edge_radius,
        "trailing_edge_radius": trailing_edge_radius,
        "trailing_edge_wedge_angle": trailing_edge_wedge_angle,
        "throat_location_fraction": throat_location_fraction,
        "tip_clearance": tip_clearance,
        "radius_hub_in": radius_hub_in,
        "radius_hub_out": radius_hub_out,
        "radius_tip_in": radius_tip_in,
        "radius_tip_out": radius_tip_out,
        "pitch": pitch,
        "pitch_in": pitch_in,
        "pitch_out": pitch_out,
        "pitch_angle": jnp.rad2deg(pitch_angle),
        "chord": chord,
        "stagger_angle": stagger_angle,
        "wrapping_angle": stagger_angle,
        "opening": opening,
        "throat_opening": opening,
        "leading_edge_diameter": leading_edge_diameter,
        "leading_edge_angle": leading_edge_angle,
        "radius_mean_throat": radius_mean_throat,
        "radius_hub_throat": radius_hub_throat,
        "radius_tip_throat": radius_tip_throat,
        "radius_shroud_in": radius_shroud_in,
        "radius_shroud_out": radius_shroud_out,
        "radius_shroud_throat": radius_shroud_throat,
        "height": height,
        "height_in": height_in,
        "height_out": height_out,
        "height_throat": height_throat,
        "A_in": A_in,
        "area_in": A_in,
        "A_out": A_out,
        "area_out": A_out,
        "A_throat": A_throat,
        "area_throat": A_throat,
        "meridional_chord": meridional_chord,
        "flaring_angle": flaring_angle,
        "aspect_ratio": aspect_ratio,
        "pitch_chord_ratio": pitch_chord_ratio,
        "solidity": solidity,
        "hub_tip_ratio_in": hub_tip_ratio_in,
        "hub_tip_ratio_out": hub_tip_ratio_out,
        "hub_tip_ratio_throat": hub_tip_ratio_throat,
        "maximum_thickness_chord_ratio": maximum_thickness_chord_ratio,
        "trailing_edge_thickness_opening_ratio": trailing_edge_thickness_opening_ratio,
        "tip_clearance_height_ratio": tip_clearance_height_ratio,
        "leading_edge_diameter_chord_ratio": leading_edge_diameter_chord_ratio,
        "gauging_angle": gauging_angle,
        "z_in": z_in,
    }


def _compute_full_geometry_for_axial_cascade(comp):
    """
    Compute 'complete geometry' for a single axial cascade component
    from the *raw* geometry specification.
    """
    name = comp.get("name", None)
    component_type = comp.get("component_type", None)
    g = comp["geometry"]

    cascade_type = g["cascade_type"]
    camberline_type = g["camberline_type"]
    camberline_type_id = bp.camberline_cartesian_type_id(camberline_type)

    base_geom = _compute_axial_cascade_geometry_jit(
        camberline_type_id=camberline_type_id,
        N_blades=g["N_blades"],
        radius_mean_in=g["radius_mean_in"],
        radius_mean_out=g["radius_mean_out"],
        blade_height_in=g["blade_height_in"],
        blade_height_out=g["blade_height_out"],
        metal_angle_in_deg=g["metal_angle_in"],
        metal_angle_out_deg=g["metal_angle_out"],
        maximum_thickness=g["maximum_thickness"],
        maximum_thickness_location_fraction=g["maximum_thickness_location_fraction"],
        leading_edge_wedge_angle=g["leading_edge_wedge_angle"],
        leading_edge_radius=g["leading_edge_radius"],
        trailing_edge_radius=g["trailing_edge_radius"],
        trailing_edge_wedge_angle=g["trailing_edge_wedge_angle"],
        throat_location_fraction=g["throat_location_fraction"],
        chord_axial=g["chord_axial"],
        tip_clearance=g["tip_clearance"],
        z_in=g["z_in"],
        N_cam_points=64,
    )

    out = {
        "name": name,
        "component_type": component_type,
        "cascade_type": cascade_type,
        "camberline_type": camberline_type,
        # Keep raw model selectors so plotting/reconstruction can use YAML intent.
        "thickness_model": g["thickness_model"],
        "chord_axial": g["chord_axial"],
        **base_geom,
    }
    for k in ("leading_edge_thickness", "denton_thickness_shape_exponent", "denton_tk_typ"):
        if k in g:
            out[k] = g[k]
    return out


def calculate_full_geometry(yaml_or_components):
    """
    For each component:
      - if component_type == 'axial_cascade':
            validate raw geometry and compute full geometry.
      - if component_type == 'vaneless_channel':
            validate minimal structure and just return flattened raw geometry.

    Returns a list of per-component geometry dicts.
    """
    components = _extract_components(yaml_or_components)

    full_list = []
    for i, comp in enumerate(components):
        # _validate_single_component(comp, i)
        ctype = comp.get("component_type")

        if ctype == "axial_cascade":
            full = _compute_full_geometry_for_axial_cascade(comp)

        elif ctype == "interspace":
            # Flatten raw geometry: name, component_type, plus raw geometry fields
            geom = comp["geometry"]
            full = {
                "name": comp.get("name", f"component_{i+1}"),
                "component_type": ctype,
                **geom,
            }

        elif ctype == "vaneless_channel":
            # Flatten raw geometry: name, component_type, plus raw geometry fields
            geom = comp["geometry"]
            full = {
                "name": comp.get("name", f"component_{i+1}"),
                "component_type": ctype,
                **geom,
            }

        full_list.append(full)

    return full_list

# ==========================
# NEW: standalone helper for raw cascade geometry
# ==========================

def calculate_full_geometry_for_axial_cascade(raw_geometry: dict, name: str | None = None):
    """
    Standalone helper to compute full geometry for a SINGLE axial cascade,
    given only its raw geometry dict (the 'geometry' block from YAML).

    Parameters
    ----------
    raw_geometry : dict
        A dict containing all REQUIRED_AXIAL_GEOM_KEYS.
    name : str or None
        Optional component name to embed in the result.

    Returns
    -------
    full : dict
        Full geometry dict for this axial cascade (same format as entries
        in calculate_full_geometry(...)).
    """
    comp = {
        "name": name if name is not None else "axial_cascade_1",
        "component_type": "axial_cascade",
        "geometry": raw_geometry,
    }

    # Reuse the existing validator + core computation
    # _validate_single_component(comp, index=0)
    return _compute_full_geometry_for_axial_cascade(comp)


# ==========================
# Optional: report function
# ==========================
def check_turbine_geometry(geom_list, display=True):
    """
    Accepts the **list of per-component geometry dicts** from calculate_full_geometry
    and prints a consolidated report, applying the same recommended ranges.

    Ranges and logic are carried over, but applied component-by-component.
    """
    # Recommended ranges (same as before)
    recommended_ranges = {
        "chord": {"min": 5e-3, "max": jnp.inf},
        "height": {"min": 5e-3, "max": jnp.inf},
        "maximum_thickness": {"min": 1e-3, "max": jnp.inf},
        "trailing_edge_thickness": {"min": 5e-4, "max": jnp.inf},
        "tip_clearance": {"min": 2e-4, "max": jnp.inf},
        "hub_tip_ratio_in": {"min": 0.50, "max": 0.95},
        "aspect_ratio": {"min": 0.8, "max": 5.0},
        "pitch_chord_ratio": {"min": 0.3, "max": 1.1},
        "stagger_angle": {"min": -10, "max": +70},
        "leading_edge_angle": {"min": -60, "max": +25},
        "leading_edge_wedge_angle": {"min": 10, "max": 60},
        "leading_edge_diameter_chord_ratio": {"min": 0.03, "max": 0.30},
        "maximum_thickness_chord_ratio": {"min": 0.05, "max": 0.30},
        "trailing_edge_thickness_opening_ratio": {"min": 0.00, "max": 0.40},
        "tip_clearance_height_ratio": {"min": 0.0, "max": 0.05},
        "throat_location_fraction": {"min": 0.5, "max": 1.0},
    }

    if isinstance(geom_list, dict):
        # allow single-component dict too
        geom_list = [geom_list]

    report_width = 92
    msgs = []
    msgs.append("-" * report_width)
    msgs.append("Axial turbine geometry report (component-wise)".center(report_width))
    msgs.append("-" * report_width)
    table_header = (
        f" {'Component':<16}{'Parameter':<34}{'Value':>8}{'Range':>20}{'In range?':>12}"
    )
    msgs.append(table_header)
    msgs.append("-" * report_width)

    vars_outside = []

    for comp in geom_list:
        name = comp.get("name", "unknown")
        ctype = comp.get("cascade_type", "stator")

        for param, lim in recommended_ranges.items():
            if param not in comp:
                continue
            val = float(comp[param])
            lb, ub = lim["min"], lim["max"]

            # Angle reversals for rotor
            if param in ("leading_edge_angle", "stagger_angle") and ctype == "rotor":
                lb, ub = -ub, -lb

            # Special for tip_clearance lower bound
            if param == "tip_clearance":
                lb = 0.0 if ctype == "stator" else lim["min"]

            in_range = (val >= lb) and (val <= ub)
            bounds = f"({lb:+0.4f}, {ub:+0.4f})"
            msgs.append(
                f" {name:<16}{param:<32}{val:>+8.4f}{bounds:>24}{str(in_range):>12}"
            )
            if not in_range:
                vars_outside.append(f"{name}:{param}")

    msgs.append("-" * report_width)
    if not vars_outside:
        msgs.append(
            " Geometry report summary: All parameters are within recommended ranges."
        )
    else:
        msgs.append(
            " Geometry report summary: Some parameters are outside recommended ranges."
        )
        for w in vars_outside:
            msgs.append(f"     - {w}")
    msgs.append("-" * report_width)

    msg = "\n".join(msgs)
    if display:
        print(msg)
    return msg
