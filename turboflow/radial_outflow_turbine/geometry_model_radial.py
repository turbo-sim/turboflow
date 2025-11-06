# geometry_model_radial.py — component-wise radial-outflow geometry (axial-compatible API)

from typing import Dict, Any, List, Optional
import yaml
import jax
import jax.numpy as jnp

from turboflow import math
from turboflow import utilities as utils
from turboflow.radial_outflow_turbine import blade_parametrization as bp


# ==============================
# Radial helpers (kept from your version)
# ==============================

def _deg2rad(x): return jnp.asarray(x) * jnp.pi / 180.0
def _rad2deg(x): return jnp.asarray(x) * 180.0 / jnp.pi

def _normalize_stagger_rad(st):
    st = jnp.asarray(st)
    st = (st + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    st = jnp.where(st >  jnp.pi/2.0, st - jnp.pi, st)
    st = jnp.where(st <= -jnp.pi/2.0, st + jnp.pi, st)
    return st

def _to_native(obj):
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, (jnp.ndarray, jax.Array)):
        arr = obj
        if arr.ndim == 0:
            return float(arr)
        return _to_native(arr.tolist())
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    return obj

def _safe_div(a, b, eps=1e-12):
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    return a / jnp.maximum(b, eps)

def _opt(row: Dict[str, Any], *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


# ==============================
# Public API parity with axial module
# ==============================

# --- at top of file (near existing REQUIRED/ANGLE sets) ---

# Keys that MUST be present in each component's geometry (radial-outflow)
REQUIRED_GEOM_KEYS = {
    "cascade_type",
    "N_blades",
    "r_in",
    "r_out",
    "metal_angle_in",
    "metal_angle_out",
    "maximum_thickness",
    "trailing_edge_thickness",
    "blade_height_in",
    "blade_height_out",
    "camberline_type",
}

# Keys that are optional but perfectly valid (used by prepare/calc)
OPTIONAL_GEOM_KEYS = {
    # throat / opening
    "throat_location_fraction",
    "opening",                        
    # leading edge
    "leading_edge_radius",
    "leading_edge_radius_fraction",
    "leading_edge_wedge_angle",
    "leading_edge_wedge",             

    # trailing edge 
    "trailing_edge_wedge",

    # clearances / thickness placement
    "tip_clearance",
    "tip_clearance_height",
    "maximum_thickness_location_fraction",
    # anything else you know you pass through:
    "theta0",                         # if you ever set it explicitly
}

# The full set of keys that the validator should accept
ALLOWED_GEOM_KEYS = REQUIRED_GEOM_KEYS | OPTIONAL_GEOM_KEYS


ANGLE_KEYS = {
    "stagger_angle",
    "leading_edge_angle",
    "leading_edge_wedge_angle",
    "trailing_edge_wedge",
    "metal_angle_in",
    "metal_angle_out",
} # deg
VALID_TYPES = {"stator", "rotor"}

# strings which are allowed
ALLOW_STR_KEYS = {
    "cascade_type",      # stator/rotor (we already check separately)
    "camberline_type",   # e.g. circular_arc, polynomial, modified_wiebe, etc
}

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
    raise TypeError("Input must be a YAML dict with 'components' or a list of components.")


def _validate_single_component(c, index=0):
    """
    Validate one component entry (radial-outflow geometry).
    """
    if "geometry" not in c or not isinstance(c["geometry"], dict):
        raise ValueError(f"Component #{index+1} missing 'geometry' dict.")

    name = c.get("name", f"component_{index+1}")
    ctype = c.get("component_type", None)
    if ctype is None:
        raise ValueError(f"Component '{name}' missing 'component_type'.")

    geom = c["geometry"]
    utils.validate_keys(geom, REQUIRED_GEOM_KEYS, ALLOWED_GEOM_KEYS)

    # cascade_type
    ct = str(geom["cascade_type"]).lower()
    if ct not in VALID_TYPES:
        raise ValueError(
            f"Component '{name}' has invalid cascade_type='{ct}'. Only {sorted(VALID_TYPES)} allowed."
        )

    # Numeric checks (angles allowed any sign)
    for k, v in geom.items():
        # some keys we allow string type:
        if k in ALLOW_STR_KEYS:
            if not isinstance(v, str):
                raise TypeError(f"Component '{name}': '{k}' must be a string (got {type(v)}: {v!r})")
            continue

        # cascade_type we already validated earlier via VALID_TYPES check
        if k == "cascade_type":
            continue

        # everything else MUST be numeric scalar
        if not isinstance(v, (int, float)) and not jnp.isscalar(v):
            raise TypeError(
                f"Component '{name}': parameter '{k}' must be a numeric scalar (int/float). Got {type(v)}: {v!r}"
            )

        # angles can be negative (only the ones you defined above in ANGLE_KEYS)
        if (k not in ANGLE_KEYS) and (float(v) < 0.0):
            raise ValueError(
                f"Component '{name}': parameter '{k}' must be non-negative. Got {v}."
            )

def validate_turbine_geometry(yaml_or_components, display=False):
    """
    Component-wise validator (radial-outflow). Same signature as axial.
    """
    components = _extract_components(yaml_or_components)
    for i, comp in enumerate(components):
        _validate_single_component(comp, i)
    msg = "All component geometry entries are valid (radial-outflow)."
    if display:
        print(msg)
    return msg


def calculate_throat_radius(radius_in, radius_out, throat_location_fraction):
    """
    (Radial) throat radius via linear interpolation.
    """
    return (1.0 - throat_location_fraction) * radius_in + throat_location_fraction * radius_out


# =================
# Core calculations
# =================

def prepare_radial_outflow_geometry(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the given radial-outflow *row.geometry* into evaluation parameters.
    (Kept from your version; returns a dict of prepared fields.)
    """
    cascade_type = row["cascade_type"]
    N_blades     = int(row["N_blades"])
    r_in         = jnp.asarray(row["r_in"], dtype=jnp.float64)
    r_out        = jnp.asarray(row["r_out"], dtype=jnp.float64)
    height_in    = jnp.asarray(row["blade_height_in"], dtype=jnp.float64)
    height_out   = jnp.asarray(row["blade_height_out"], dtype=jnp.float64)
    metal_angle_in_deg  = float(row["metal_angle_in"])
    metal_angle_out_deg = float(row["metal_angle_out"])
    camberline_type = row["camberline_type"]

    # Camberline sampling and stagger/chord
    ma1 = _deg2rad(metal_angle_in_deg)
    ma2 = _deg2rad(metal_angle_out_deg)
    theta0 = _deg2rad(0.0)
    u = jnp.linspace(0.0, 1.0, 400)
    _x, _y, _r, theta, _metal_angle, _phi, stagger_rad, chord = bp.compute_camberline_radial(
        camberline_type, r_in, r_out, ma1, ma2, theta0, u
    )
    stagger_rad = _normalize_stagger_rad(stagger_rad)
    stagger_deg = _rad2deg(stagger_rad)
    d_theta = theta[-1] - theta[0]

    # Pitches (from N_blades and radii)
    pitch_in   = 2.0 * jnp.pi * r_in  / float(N_blades)
    pitch_out  = 2.0 * jnp.pi * r_out / float(N_blades)
    r_mean     = 0.5 * (r_in + r_out)
    pitch_mean = 2.0 * jnp.pi * r_mean / float(N_blades)

    # Throat data (optional)
    throat_f = _opt(row, "throat_location_fraction", default=None)
    height_throat = None
    r_throat = None
    throat_opening = None
    throat_area = None
    if throat_f is not None:
        throat_f = float(throat_f)
        r_throat = r_in + throat_f * (r_out - r_in)
        height_throat = (1.0 - throat_f) * height_in + throat_f * height_out
        metal_angle_out_rad = _deg2rad(metal_angle_out_deg)
        throat_opening = pitch_out * jnp.cos(metal_angle_out_rad + 0.5 * d_theta)
        throat_area = throat_opening * height_throat

    # Leading-edge radius
    le_radius = _opt(row, "leading_edge_radius", default=None)
    if le_radius is None:
        le_frac = _opt(row, "leading_edge_radius_fraction", default=None)
        if le_frac is not None:
            le_radius = float(le_frac) * float(chord)

    # Wedge angle (deg)
    le_wedge_angle = _opt(row, "leading_edge_wedge_angle", "leading_edge_wedge", default=None)

    prepared = dict(row)
    prepared.update({
        "cascade_type": cascade_type,
        "N_blades": N_blades,
        "r_in": r_in,
        "r_out": r_out,
        "blade_height_in": height_in,
        "blade_height_out": height_out,
        "metal_angle_in": metal_angle_in_deg,
        "metal_angle_out": metal_angle_out_deg,
        "camberline_type": camberline_type,

        # Derived
        "theta0": 0.0,
        "theta": theta,
        "d_theta": d_theta,
        "chord": chord,
        "stagger_angle": stagger_deg,
        "pitch_in": pitch_in,
        "pitch_out": pitch_out,
        "pitch_mean": pitch_mean,

        # Throat (optional)
        "throat_location_fraction": throat_f,
        "r_throat": r_throat,
        "height_throat": height_throat,
        "throat_opening": throat_opening,
        "opening": throat_opening,
        "throat_area": throat_area,

        # Leading edge
        "leading_edge_radius_abs": le_radius,
        "leading_edge_wedge_angle": le_wedge_angle,
        "leading_edge_angle": metal_angle_in_deg,  # incidence ref
    })
    return _to_native(prepared)


def calculate_full_radial_outflow_geometry(prepared: Dict[str, Any],
                                           meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Complete geometry using prepared values; returns a full per-component dict.
    """
    r_in   = jnp.asarray(prepared["r_in"], dtype=jnp.float64)
    r_out  = jnp.asarray(prepared["r_out"], dtype=jnp.float64)
    h_in   = jnp.asarray(prepared["blade_height_in"], dtype=jnp.float64)
    h_out  = jnp.asarray(prepared["blade_height_out"], dtype=jnp.float64)
    chord  = jnp.asarray(prepared["chord"], dtype=jnp.float64)
    pitch_in   = jnp.asarray(prepared["pitch_in"], dtype=jnp.float64)
    pitch_out  = jnp.asarray(prepared["pitch_out"], dtype=jnp.float64)
    pitch_mean = jnp.asarray(prepared["pitch_mean"], dtype=jnp.float64)
    stagger_deg = jnp.asarray(prepared["stagger_angle"], dtype=jnp.float64)

    r_throat       = prepared.get("r_throat", None)
    height_throat  = prepared.get("height_throat", None)
    throat_opening = prepared.get("throat_opening", None)

    t_max = jnp.asarray(prepared.get("maximum_thickness", 0.0), dtype=jnp.float64)
    t_te  = jnp.asarray(prepared.get("trailing_edge_thickness", 0.0), dtype=jnp.float64)
    le_radius_abs = prepared.get("leading_edge_radius_abs", None)
    tip_clearance_height = prepared.get("tip_clearance_height", None)

    # We model hub/tip radii equal (annular line at r_in/out) for this cross-section
    radius_hub_in   = r_in
    radius_hub_out  = r_out
    radius_hub_mean = 0.5 * (r_in + r_out)
    radius_hub_throat = jnp.asarray(r_throat, dtype=jnp.float64) if (r_throat is not None) else None

    radius_tip_in   = radius_hub_in
    radius_tip_out  = radius_hub_out
    radius_tip_mean = radius_hub_mean
    radius_tip_throat = radius_hub_throat if (radius_hub_throat is not None) else None

    height_in   = h_in
    height_out  = h_out
    height_mean = 0.5 * (h_in + h_out)
    height_th   = jnp.asarray(height_throat, dtype=jnp.float64) if (height_throat is not None) else None

    # Areas
    A_in  = 2.0 * jnp.pi * r_in  * h_in
    A_out = 2.0 * jnp.pi * r_out * h_out
    A_throat = None
    if (r_throat is not None) and (height_throat is not None):
        A_throat = 2.0 * jnp.pi * radius_hub_throat * height_th

    # Meridional chord & flaring
    stagger_rad = jnp.deg2rad(stagger_deg)
    meridional_chord = chord * jnp.cos(stagger_rad)
    flaring_angle = jnp.rad2deg(jnp.arctan(_safe_div((h_out - h_in), meridional_chord)))

    # Ratios
    pitch_chord_ratio = _safe_div(pitch_mean, chord)
    solidity = _safe_div(chord, pitch_mean)
    aspect_ratio = _safe_div(height_mean, chord)
    maximum_thickness_chord_ratio = _safe_div(t_max, chord)

    leading_edge_diameter = None
    leading_edge_diameter_chord_ratio = None
    if le_radius_abs is not None:
        le_radius_abs = jnp.asarray(le_radius_abs, dtype=jnp.float64)
        leading_edge_diameter = 2.0 * le_radius_abs
        leading_edge_diameter_chord_ratio = _safe_div(leading_edge_diameter, chord)

    trailing_edge_thickness_opening_ratio = None
    if throat_opening is not None:
        trailing_edge_thickness_opening_ratio = _safe_div(t_te, throat_opening)

    tip_clearance_height_ratio = None
    if tip_clearance_height is not None:
        tip_clearance_height_ratio = _safe_div(jnp.asarray(tip_clearance_height, dtype=jnp.float64), height_mean)

    # Hub–tip ratios
    hub_tip_ratio_in   = _safe_div(radius_hub_in,  radius_tip_in)
    hub_tip_ratio_out  = _safe_div(radius_hub_out, radius_tip_out)
    hub_tip_ratio_mean = _safe_div(radius_hub_mean, radius_tip_mean)
    hub_tip_ratio_throat = None
    if (radius_hub_throat is not None) and (radius_tip_throat is not None):
        hub_tip_ratio_throat = _safe_div(radius_hub_throat, radius_tip_throat)

    # Counts (if provided via meta)
    number_of_cascades = meta.get("number_of_cascades") if meta else None
    number_of_stages   = meta.get("number_of_stages")   if meta else None

    # Gauging angle (deg): same definition as axial (arccos(A_throat/A_out)); sign by cascade_type later
    base_gauge = None
    if (A_throat is not None) and (A_out is not None):
        ratio = jnp.clip(_safe_div(A_throat, A_out), 0.0, 1.0)
        base_gauge = jnp.degrees(jnp.arccos(ratio))

    # Aliases to axial keys (mean radii)
    radius_mean_in     = radius_hub_in
    radius_mean_out    = radius_hub_out
    radius_mean_throat = radius_hub_throat

    out = dict(prepared)
    out.update({
        # radii
        "radius_hub_in": radius_hub_in,
        "radius_hub_out": radius_hub_out,
        "radius_tip_in": radius_tip_in,
        "radius_tip_out": radius_tip_out,
        "radius_mean_in": radius_mean_in,
        "radius_mean_out": radius_mean_out,
        "radius_mean_throat": radius_mean_throat,

        # heights
        "height_in": height_in,
        "height_out": height_out,
        "height_throat": height_th,
        "height": height_mean,  # axial alias

        # areas
        "A_in": A_in,
        "A_out": A_out,
        "A_throat": A_throat,

        # projections / angles
        "meridional_chord": meridional_chord,
        "flaring_angle": flaring_angle,  # deg

        # pitches
        "pitch_in": pitch_in,
        "pitch_out": pitch_out,
        "pitch_mean": pitch_mean,
        "pitch": pitch_mean,  # axial alias

        # ratios
        "aspect_ratio": aspect_ratio,
        "pitch_chord_ratio": pitch_chord_ratio,
        "solidity": solidity,
        "maximum_thickness_chord_ratio": maximum_thickness_chord_ratio,
        "trailing_edge_thickness_opening_ratio": trailing_edge_thickness_opening_ratio,
        "leading_edge_diameter_chord_ratio": leading_edge_diameter_chord_ratio,
        "tip_clearance_height_ratio": tip_clearance_height_ratio,
        "hub_tip_ratio_in": hub_tip_ratio_in,
        "hub_tip_ratio_out": hub_tip_ratio_out,
        "hub_tip_ratio_mean": hub_tip_ratio_mean,
        "hub_tip_ratio_throat": hub_tip_ratio_throat,
        

        # edge geometry
        "leading_edge_diameter": leading_edge_diameter,   # m if known
        "leading_edge_angle": prepared.get("leading_edge_angle"),
        "leading_edge_wedge_angle": prepared.get("leading_edge_wedge_angle"),

        # machine-level counts (duplicated per row)
        "number_of_cascades": number_of_cascades,
        "number_of_stages": number_of_stages,

        # base gauging (sign applied later)
        "gauging_angle_base": base_gauge,
    })

    return _to_native(out)


# Adapter to axial-style “compute one component”
def _compute_full_geometry_for_component(comp: Dict[str, Any], index: int, ncomp: int) -> Dict[str, Any]:
    """
    Axial-compatible per-component builder using the radial pipeline.
    """
    name = comp.get("name", f"component_{index+1}")
    component_type = comp.get("component_type", None)
    geom = comp["geometry"]

    prepared = prepare_radial_outflow_geometry(geom)
    meta = {"number_of_cascades": ncomp, "number_of_stages": max(0, ncomp // 2)}
    full = calculate_full_radial_outflow_geometry(prepared, meta=meta)

    # Ensure required axial aliases exist
    full["name"] = name
    full["component_type"] = component_type
    full["cascade_type"] = str(geom["cascade_type"]).lower()

    # tip_clearance alias (radial spec may have tip_clearance_height only)
    if "tip_clearance" not in full:
        full["tip_clearance"] = 0.0

    # Ensure max thickness / TE thickness present (copied from YAML)
    full.setdefault("maximum_thickness", float(geom["maximum_thickness"]))
    full.setdefault("trailing_edge_thickness", float(geom["trailing_edge_thickness"]))

    # Gauging sign convention: +stator, −rotor (like axial)
    base = full.get("gauging_angle_base", None)
    if base is not None:
        gsign = +1.0 if full["cascade_type"] == "stator" else -1.0
        full["gauging_angle"] = float(base) * gsign
    else:
        full["gauging_angle"] = None
    if "gauging_angle_base" in full:
        del full["gauging_angle_base"]

    return full


def calculate_full_geometry(yaml_or_components):
    """
    NEW BEHAVIOR (axial-compatible):
    Accepts the component-wise YAML (or list of components) and returns
    a **list of per-component complete geometry dicts** for radial outflow.
    """
    components = _extract_components(yaml_or_components)

    # validate inputs first
    for i, comp in enumerate(components):
        _validate_single_component(comp, i)

    ncomp = len(components)
    out = []
    for i, comp in enumerate(components):
        out.append(_compute_full_geometry_for_component(comp, i, ncomp))
    return out


# ==========================
# Optional: report function
# ==========================

def check_turbine_geometry(geom_list, display=True):
    """
    Component-wise report (radial-outflow) with ranges similar to axial for compatibility.
    """
    if isinstance(geom_list, dict):
        geom_list = [geom_list]

    recommended_ranges = {
        "chord": {"min": 5e-3, "max": jnp.inf},
        "height": {"min": 5e-3, "max": jnp.inf},
        "maximum_thickness": {"min": 1e-3, "max": jnp.inf},
        "trailing_edge_thickness": {"min": 5e-4, "max": jnp.inf},
        "tip_clearance": {"min": 0.0, "max": jnp.inf},
        "aspect_ratio": {"min": 0.5, "max": 6.0},
        "pitch_chord_ratio": {"min": 0.3, "max": 1.5},
        "leading_edge_wedge_angle": {"min": 5, "max": 80},
        "leading_edge_diameter_chord_ratio": {"min": 0.02, "max": 0.40},
        "maximum_thickness_chord_ratio": {"min": 0.05, "max": 0.35},
        "trailing_edge_thickness_opening_ratio": {"min": 0.00, "max": 0.50},
        "throat_location_fraction": {"min": 0.4, "max": 1.0},
    }

    width = 92
    msgs = []
    msgs.append("-" * width)
    msgs.append("Radial-outflow turbine geometry report (component-wise)".center(width))
    msgs.append("-" * width)
    header = f" {'Component':<16}{'Parameter':<34}{'Value':>8}{'Range':>20}{'In range?':>12}"
    msgs.append(header)
    msgs.append("-" * width)

    warnings = []

    for comp in geom_list:
        name = comp.get("name", "unknown")
        ctype = comp.get("cascade_type", "stator")

        for param, lim in recommended_ranges.items():
            if param not in comp or comp[param] is None:
                continue
            val = float(comp[param])
            lb, ub = float(lim["min"]), float(lim["max"])
            in_range = (val >= lb) and (val <= ub)
            rng = f"({lb:+0.4f}, {ub:+0.4f})"
            msgs.append(f" {name:<16}{param:<32}{val:>+8.4f}{rng:>24}{str(in_range):>12}")
            if not in_range:
                warnings.append(f"{name}:{param}")

    msgs.append("-" * width)
    if not warnings:
        msgs.append(" Geometry report summary: All parameters are within recommended ranges.")
    else:
        msgs.append(" Geometry report summary: Some parameters are outside recommended ranges.")
        for w in warnings:
            msgs.append(f"     - {w}")
    msgs.append("-" * width)

    msg = "\n".join(msgs)
    if display:
        print(msg)
    return msg


# -------------------------
# (Optional) YAML helpers
# -------------------------

def load_config_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
