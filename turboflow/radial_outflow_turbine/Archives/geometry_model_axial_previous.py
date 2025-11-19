from turboflow import math
from turboflow import utilities as utils

import jax.numpy as jnp

# ==============================
# Required keys (component-wise)
# ==============================
REQUIRED_GEOM_KEYS = {
    "cascade_type",
    "radius_hub_in",
    "radius_hub_out",
    "radius_tip_in",
    "radius_tip_out",
    "pitch",
    "chord",
    "stagger_angle",
    "opening",
    "leading_edge_diameter",
    "leading_edge_wedge_angle",
    "leading_edge_angle",
    "trailing_edge_thickness",
    "tip_clearance",
    "maximum_thickness",
    "throat_location_fraction",
}

ANGLE_KEYS = {"leading_edge_angle", "stagger_angle"}
VALID_TYPES = {"stator", "rotor"}


# ==============================
# Adapters / Input normalization
# ==============================
def _extract_components(obj):
    """
    Accept either:
      - full YAML dict with 'components' key, or
      - a list of component dicts
    Return a list of component dicts.
    """
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
def _validate_single_component(c, index=0):
    """
    Validate one component entry:
      - has name (optional but recommended), component_type, geometry dict
      - geometry has required keys, valid cascade_type, numeric types, non-negative for non-angles
    """
    # Basic structure
    if "geometry" not in c or not isinstance(c["geometry"], dict):
        raise ValueError(f"Component #{index+1} missing 'geometry' dict.")

    # Optional but helpful
    name = c.get("name", f"component_{index+1}")
    ctype = c.get("component_type", None)
    if ctype is None:
        raise ValueError(f"Component '{name}' missing 'component_type'.")

    geom = c["geometry"]

    # Keys
    utils.validate_keys(geom, REQUIRED_GEOM_KEYS, REQUIRED_GEOM_KEYS)

    # cascade_type
    ct = geom["cascade_type"]
    if ct not in VALID_TYPES:
        raise ValueError(
            f"Component '{name}' has invalid cascade_type='{ct}'. Only {sorted(VALID_TYPES)} allowed."
        )

    # Numeric checks (angles allowed to be negative)
    for k, v in geom.items():
        if k == "cascade_type":
            continue
        # jnp.isscalar handles Python floats/ints; at least ensure numeric
        if not isinstance(v, (int, float)) and not jnp.isscalar(v):
            raise TypeError(
                f"Component '{name}': parameter '{k}' must be a numeric scalar (int/float). Got {type(v)}."
            )
        if k not in ANGLE_KEYS and not (float(v) >= 0.0):
            raise ValueError(
                f"Component '{name}': parameter '{k}' must be non-negative. Got {v}."
            )


def validate_turbine_geometry(yaml_or_components, display=False):
    """
    Component-wise validator. Iterates over components and validates each geometry dict.
    Returns a message string if all pass.
    """
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
    """
    throat = (1 - frac) * r_in + frac * r_out
    Works with scalars or jax scalars.
    """
    return (
        1.0 - throat_location_fraction
    ) * radius_in + throat_location_fraction * radius_out


def _compute_full_geometry_for_component(comp):
    """
    Compute 'complete geometry' for a single component (stator/rotor axial cascade).
    Returns a dict that includes:
      - name, component_type
      - original geometry fields
      - all derived fields (heights, areas, ratios, angles, etc.)
    """
    name = comp.get("name", None)
    component_type = comp.get("component_type", None)
    g = comp["geometry"]

    # Pull scalars
    cascade_type = g["cascade_type"]  # 'stator' or 'rotor'
    radius_hub_in = float(g["radius_hub_in"])
    radius_hub_out = float(g["radius_hub_out"])
    radius_tip_in = float(g["radius_tip_in"])
    radius_tip_out = float(g["radius_tip_out"])
    pitch = float(g["pitch"])
    chord = float(g["chord"])
    stagger_angle = float(g["stagger_angle"])
    opening = float(g["opening"])
    leading_edge_diameter = float(g["leading_edge_diameter"])
    leading_edge_wedge_angle = float(g["leading_edge_wedge_angle"])
    leading_edge_angle = float(g["leading_edge_angle"])
    trailing_edge_thickness = float(g["trailing_edge_thickness"])
    tip_clearance = float(g["tip_clearance"])
    maximum_thickness = float(g["maximum_thickness"])
    throat_location_fraction = float(g["throat_location_fraction"])

    # Mean radii
    radius_mean_in = 0.5 * (radius_tip_in + radius_hub_in)
    radius_mean_out = 0.5 * (radius_tip_out + radius_hub_out)
    radius_hub_throat = calculate_throat_radius(
        radius_hub_in, radius_hub_out, throat_location_fraction
    )
    radius_tip_throat = calculate_throat_radius(
        radius_tip_in, radius_tip_out, throat_location_fraction
    )
    radius_mean_throat = calculate_throat_radius(
        radius_mean_in, radius_mean_out, throat_location_fraction
    )

    # Shroud radii (tip + clearance)
    radius_shroud_in = radius_tip_in + tip_clearance
    radius_shroud_out = radius_tip_out + tip_clearance
    radius_shroud_throat = calculate_throat_radius(
        radius_shroud_in, radius_shroud_out, throat_location_fraction
    )

    # Heights
    height_in = radius_tip_in - radius_hub_in
    height_out = radius_tip_out - radius_hub_out
    height_throat = radius_tip_throat - radius_hub_throat
    height = 0.5 * (height_in + height_out)

    # Areas
    A_in = jnp.pi * (radius_tip_in**2 - radius_hub_in**2)
    A_out = jnp.pi * (radius_tip_out**2 - radius_hub_out**2)
    # Use opening definition: A_throat = (2*pi * r_mean_throat * h_throat) * (opening / pitch)
    A_throat = (2.0 * jnp.pi * radius_mean_throat * height_throat) * (opening / pitch)

    # Gauging angle (sign convention: stator +, rotor -) to mimic prior alternating sign
    base_gauge = math.arccosd(A_throat / A_out)
    gauging_angle = base_gauge if cascade_type == "stator" else -base_gauge

    # Axial chord and flaring angle
    meridional_chord = chord * math.cosd(stagger_angle)
    # Avoid divide-by-zero if meridional_chord==0
    flaring_angle = math.arctand(
        (height_out - height_in) / max(meridional_chord, 1e-12) / 2.0
    )

    # Ratios
    aspect_ratio = height / chord
    pitch_chord_ratio = pitch / chord
    solidity = 1.0 / pitch_chord_ratio
    hub_tip_ratio_in = radius_hub_in / radius_tip_in
    hub_tip_ratio_out = radius_hub_out / radius_tip_out
    hub_tip_ratio_throat = radius_hub_throat / max(radius_tip_throat, 1e-12)
    maximum_thickness_chord_ratio = maximum_thickness / chord
    trailing_edge_thickness_opening_ratio = trailing_edge_thickness / max(
        opening, 1e-12
    )
    tip_clearance_height_ratio = tip_clearance / max(height, 1e-12)
    leading_edge_diameter_chord_ratio = leading_edge_diameter / chord

    # Full dict for this component
    full = {
        # identifiers
        "name": name,
        "component_type": component_type,
        # original geometry (echo back)
        "cascade_type": cascade_type,
        "radius_hub_in": radius_hub_in,
        "radius_hub_out": radius_hub_out,
        "radius_tip_in": radius_tip_in,
        "radius_tip_out": radius_tip_out,
        "pitch": pitch,
        "chord": chord,
        "stagger_angle": stagger_angle,
        "opening": opening,
        "leading_edge_diameter": leading_edge_diameter,
        "leading_edge_wedge_angle": leading_edge_wedge_angle,
        "leading_edge_angle": leading_edge_angle,
        "trailing_edge_thickness": trailing_edge_thickness,
        "tip_clearance": tip_clearance,
        "maximum_thickness": maximum_thickness,
        "throat_location_fraction": throat_location_fraction,
        # derived geometry
        "radius_mean_in": radius_mean_in,
        "radius_mean_out": radius_mean_out,
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
        "A_out": A_out,
        "A_throat": A_throat,
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
    }
    return full


def calculate_full_geometry(yaml_or_components):
    """
    NEW BEHAVIOR:
    Accept the new component-wise YAML (or components list) and return a
    **list of per-component complete geometry dicts**.
    """
    components = _extract_components(yaml_or_components)

    # Validate first (raises on problems)
    for i, comp in enumerate(components):
        _validate_single_component(comp, i)

    # Compute each component independently
    full_list = []
    for comp in components:
        full_list.append(_compute_full_geometry_for_component(comp))
    return full_list


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
