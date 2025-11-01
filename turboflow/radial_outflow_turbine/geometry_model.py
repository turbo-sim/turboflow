# radial_outflow_geometry.py  (JAX-only, enriched keys)
# ------------------------------------------------------------
# Functions:
#   - prepare_radial_outflow_geometry(row)
#   - calculate_full_radial_outflow_geometry(prepared_row, meta=None)
#   - load_config_yaml(path)
#   - prepare_all_rows(cfg)
#   - calculate_full_geometries(prepared_by_name, meta=None)
#   - run_radial_outflow_pipeline_from_yaml(path)
#   - sanity_check_row(...)
#   - sanity_check_all_rows(...)
#
# Notes:
# - Radial outflow turbine conventions:
#     * height_in/out come directly from YAML (blade_height_in/out).
#     * A_in = 2π r_in * height_in; A_out = 2π r_out * height_out.
# - Throat opening formula:
#     throat_opening = pitch_out * cos(metal_angle_out_rad + 0.5*(theta[-1]-theta[0]))
#   computed if throat_location_fraction is provided (for r_throat, height_throat).
# - We avoid the word "shroud" entirely; use only "hub" naming.
# ------------------------------------------------------------

from typing import Dict, Any, List, Tuple, Optional
import yaml
import jax
import jax.numpy as jnp

# Your JAX-only camberline module
from turboflow.radial_outflow_turbine import blade_parametrization as bp


# ---------------------------
# Small helpers (JAX-only)
# ---------------------------

def _deg2rad(x):
    return jnp.asarray(x) * jnp.pi / 180.0

def _rad2deg(x):
    return jnp.asarray(x) * 180.0 / jnp.pi

def _normalize_stagger_rad(st):
    """Map stagger to (-pi/2, +pi/2] without changing the geometry. (JAX-only)"""
    st = jnp.asarray(st)
    st = (st + jnp.pi) % (2.0 * jnp.pi) - jnp.pi  # (-pi, pi]
    st = jnp.where(st >  jnp.pi / 2.0, st - jnp.pi, st)
    st = jnp.where(st <= -jnp.pi / 2.0, st + jnp.pi, st)
    return st

def _to_native(obj):
    """Recursively convert JAX scalars/arrays (and Python containers) to native Python types."""
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
    """Return first present key among names, else default."""
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


# ------------------------------------------------------------
# Geometry preparation and full-geometry calculations
# ------------------------------------------------------------

def prepare_radial_outflow_geometry(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the given radial-outflow row (from YAML) into evaluation parameters.

    Inputs (from YAML):
      - cascade_type, N_blades
      - r_in, r_out
      - metal_angle_in (deg), metal_angle_out (deg)   # wrt tangential
      - blade_height_in, blade_height_out             # span at inlet/outlet
      - maximum_thickness, trailing_edge_thickness
      - camberline_type
      - Optional:
          * maximum_thickness_location_fraction
          * leading_edge_radius or leading_edge_radius_fraction
          * leading_edge_wedge_angle (or leading_edge_wedge)
          * throat_location_fraction

    Derived here:
      - chord, stagger_angle, theta (array), d_theta
      - pitch_in/out/mean
      - throat_opening, throat_area (if throat info available)
    """

    # Required
    cascade_type = row["cascade_type"]
    N_blades     = int(row["N_blades"])
    r_in         = jnp.asarray(row["r_in"], dtype=jnp.float64)
    r_out        = jnp.asarray(row["r_out"], dtype=jnp.float64)
    height_in    = jnp.asarray(row["blade_height_in"], dtype=jnp.float64)
    height_out   = jnp.asarray(row["blade_height_out"], dtype=jnp.float64)
    metal_angle_in_deg  = float(row["metal_angle_in"])
    metal_angle_out_deg = float(row["metal_angle_out"])
    camberline_type = row["camberline_type"]

    # Angles (for camberline)
    ma1 = _deg2rad(metal_angle_in_deg)
    ma2 = _deg2rad(metal_angle_out_deg)
    theta0 = _deg2rad(0.0)

    # Camberline sampling
    u = jnp.linspace(0.0, 1.0, 400)
    # bp.compute_camberline_radial -> x, y, r, theta, metal_angle, phi, stagger, chord
    _x, _y, _r, theta, _metal_angle, _phi, stagger_rad, chord = bp.compute_camberline_radial(
        camberline_type, r_in, r_out, ma1, ma2, theta0, u
    )
    stagger_rad = _normalize_stagger_rad(stagger_rad)
    stagger_deg = _rad2deg(stagger_rad)
    d_theta = theta[-1] - theta[0]

    # Pitches
    pitch_in   = 2.0 * jnp.pi * r_in  / float(N_blades)
    pitch_out  = 2.0 * jnp.pi * r_out / float(N_blades)
    r_mean     = 0.5 * (r_in + r_out)
    pitch_mean = 2.0 * jnp.pi * r_mean / float(N_blades)

    # Optional parameters
    throat_f = _opt(row, "throat_location_fraction", default=None)

    # Throat height (if fraction provided) – linear interpolation of blade height
    height_throat = None
    r_throat = None
    throat_opening = None
    throat_area = None

    if throat_f is not None:
        throat_f = float(throat_f)
        r_throat = r_in + throat_f * (r_out - r_in)
        height_throat = (1.0 - throat_f) * height_in + throat_f * height_out

        # throat_opening per spec: pitch_out * cos(metal_angle_out_rad + 0.5 * d_theta)
        metal_angle_out_rad = _deg2rad(metal_angle_out_deg)
        throat_opening = pitch_out * jnp.cos(metal_angle_out_rad + 0.5 * d_theta)
        throat_area = throat_opening * height_throat

    # Leading-edge radius (absolute); prefer absolute, fallback to fraction * chord
    le_radius = _opt(row, "leading_edge_radius", default=None)
    if le_radius is None:
        le_frac = _opt(row, "leading_edge_radius_fraction", default=None)
        if le_frac is not None:
            le_radius = float(le_frac) * float(chord)

    # Leading-edge wedge angle (pass-through from YAML; accept two common key names)
    le_wedge_angle = _opt(row, "leading_edge_wedge_angle", "leading_edge_wedge", default=None)

    # Pack prepared dictionary
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

        # Derived here
        "theta0": 0.0,                  # deg
        "theta": theta,                 # rad array (keep for throat & checks)
        "d_theta": d_theta,             # rad
        "chord": chord,                 # m
        "stagger_angle": stagger_deg,   # deg
        "pitch_in": pitch_in,           # m
        "pitch_out": pitch_out,         # m
        "pitch_mean": pitch_mean,       # m

        # Throat (optional)
        "throat_location_fraction": throat_f,
        "r_throat": r_throat,
        "height_throat": height_throat,
        "throat_opening": throat_opening,
        "opening": throat_opening,
        "throat_area": throat_area,

        # Leading edge
        "leading_edge_radius_abs": le_radius,          # absolute radius
        "leading_edge_wedge_angle": le_wedge_angle,    # deg, pass-through from YAML
        "leading_edge_angle": metal_angle_in_deg,      # deg, alias for incidence model
    })

    return _to_native(prepared)


def calculate_full_radial_outflow_geometry(prepared: Dict[str, Any],
                                           meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Complete the geometry using prepared values + pure geometric relations.

    Adds (all as JAX then cast to native):
      Areas: A_in, A_out, A_throat
      Radii (hub & tip): radius_hub_in/out/mean(/throat), radius_tip_in/out/mean(/throat)
      Heights: height_in/out/mean(/throat)
      Pitches: pitch_in/out/mean
      Chords/angles: chord, meridional_chord, flaring_angle, stagger_angle (deg)
      Ratios: aspect_ratio, pitch_chord_ratio, solidity,
              maximum_thickness_chord_ratio,
              trailing_edge_thickness_opening_ratio,
              leading_edge_diameter_chord_ratio,
              tip_clearance_height_ratio,
              hub_tip_ratio_in/out/mean(/throat)
      Counts: number_of_cascades, number_of_stages (if meta provided)
      Gauging: gauging_angle (computed if A_throat & A_out available)
      Aliases: radius_mean_in/out/throat (equal to hub radii here)
               pitch (= pitch_mean), height (= height_mean),
               leading_edge_diameter, leading_edge_angle
    """

    # Pull essentials as JAX scalars
    r_in   = jnp.asarray(prepared["r_in"], dtype=jnp.float64)
    r_out  = jnp.asarray(prepared["r_out"], dtype=jnp.float64)
    h_in   = jnp.asarray(prepared["blade_height_in"], dtype=jnp.float64)
    h_out  = jnp.asarray(prepared["blade_height_out"], dtype=jnp.float64)
    chord  = jnp.asarray(prepared["chord"], dtype=jnp.float64)
    pitch_in   = jnp.asarray(prepared["pitch_in"], dtype=jnp.float64)
    pitch_out  = jnp.asarray(prepared["pitch_out"], dtype=jnp.float64)
    pitch_mean = jnp.asarray(prepared["pitch_mean"], dtype=jnp.float64)
    stagger_deg = jnp.asarray(prepared["stagger_angle"], dtype=jnp.float64)

    # Optional throat items
    r_throat        = prepared.get("r_throat", None)
    height_throat   = prepared.get("height_throat", None)
    throat_opening  = prepared.get("throat_opening", None)
    throat_area     = prepared.get("throat_area", None)

    # Optional inputs for ratios
    t_max = jnp.asarray(prepared.get("maximum_thickness", 0.0), dtype=jnp.float64)
    t_te  = jnp.asarray(prepared.get("trailing_edge_thickness", 0.0), dtype=jnp.float64)
    le_radius_abs = prepared.get("leading_edge_radius_abs", None)
    tip_clearance_height = prepared.get("tip_clearance_height", None)

    # Radii (hub-only vocabulary retained)
    radius_hub_in   = r_in
    radius_hub_out  = r_out
    radius_hub_mean = 0.5 * (r_in + r_out)
    radius_hub_throat = jnp.asarray(r_throat, dtype=jnp.float64) if (r_throat is not None) else None

    # Heights
    height_in   = h_in
    height_out  = h_out
    height_mean = 0.5 * (h_in + h_out)
    height_th   = jnp.asarray(height_throat, dtype=jnp.float64) if (height_throat is not None) else None

    # Tip radii (same as hub for radial outflow cross-section here)
    radius_tip_in   = radius_hub_in
    radius_tip_out  = radius_hub_out
    radius_tip_mean = radius_hub_mean
    radius_tip_throat = radius_hub_throat if (radius_hub_throat is not None) else None

    # Areas
    A_in  = 2.0 * jnp.pi * r_in  * h_in
    A_out = 2.0 * jnp.pi * r_out * h_out
    A_throat = None
    if (r_throat is not None) and (height_throat is not None):
        A_throat = 2.0 * jnp.pi * radius_hub_throat * height_th

    # Camber projections and flaring
    stagger_rad = jnp.deg2rad(stagger_deg)
    meridional_chord = chord * jnp.cos(stagger_rad)
    flaring_angle = jnp.rad2deg(jnp.arctan(_safe_div((h_out - h_in), meridional_chord)))

    # Ratios
    pitch_chord_ratio = _safe_div(pitch_mean, chord)
    solidity = _safe_div(chord, pitch_mean)
    aspect_ratio = _safe_div(height_mean, chord)
    maximum_thickness_chord_ratio = _safe_div(t_max, chord)

    leading_edge_diameter_chord_ratio = None
    leading_edge_diameter = None
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

    # Optional machine-level counts
    number_of_cascades = meta.get("number_of_cascades") if meta else None
    number_of_stages   = meta.get("number_of_stages")   if meta else None

    # Gauging angle (degrees) — computed if A_throat & A_out exist
    gauging_angle = None
    if (A_throat is not None) and (A_out is not None):
        ratio = _safe_div(A_throat, A_out)
        ratio = jnp.clip(ratio, 0.0, 1.0)                     # safe domain for arccos
        base_deg = jnp.degrees(jnp.arccos(ratio))             # arccosd(A_throat / A_out)
        if number_of_cascades is not None:
            n_casc = int(number_of_cascades)
            signs = jnp.array([(-1.0) ** i for i in range(n_casc)], dtype=jnp.float64)
            gauging_angle = base_deg * signs                  # alternate +/- across cascades
        else:
            gauging_angle = base_deg

    # Aliases to match your earlier list (equals hub radii here)
    radius_mean_in     = radius_hub_in
    radius_mean_out    = radius_hub_out
    radius_mean_throat = radius_hub_throat

    out = dict(prepared)
    out.update({
        # Radii (hub)
        "radius_hub_in": radius_hub_in,
        "radius_hub_out": radius_hub_out,
        "radius_hub_mean": radius_hub_mean,
        "radius_hub_throat": radius_hub_throat,

        # Radii (tip)
        "radius_tip_in": radius_tip_in,
        "radius_tip_out": radius_tip_out,
        "radius_tip_mean": radius_tip_mean,
        "radius_tip_throat": radius_tip_throat,

        # Aliases (per your key list)
        "radius_mean_in": radius_mean_in,
        "radius_mean_out": radius_mean_out,
        "radius_mean_throat": radius_mean_throat,

        # Heights
        "height_in": height_in,
        "height_out": height_out,
        "height_mean": height_mean,
        "height_throat": height_th,

        # NEW alias for loss model compatibility
        "height": height_mean,              # == height_mean

        # Areas
        "A_in": A_in,
        "A_out": A_out,
        "A_throat": A_throat,

        # Geometry projections / angles
        "meridional_chord": meridional_chord,
        "flaring_angle": flaring_angle,  # deg

        # Pitches
        "pitch_in": pitch_in,
        "pitch_out": pitch_out,
        "pitch_mean": pitch_mean,

        # NEW alias for loss model compatibility
        "pitch": pitch_mean,              # == pitch_mean

        # Ratios
        "aspect_ratio": aspect_ratio,
        "pitch_chord_ratio": pitch_chord_ratio,
        "solidity": solidity,
        "maximum_thickness_chord_ratio": maximum_thickness_chord_ratio,
        "trailing_edge_thickness_opening_ratio": trailing_edge_thickness_opening_ratio,
        "leading_edge_diameter_chord_ratio": leading_edge_diameter_chord_ratio,
        "tip_clearance_height_ratio": tip_clearance_height_ratio,

        # Leading edge (explicit fields for loss model)
        "leading_edge_angle": prepared.get("leading_edge_angle"),         # deg (metal_angle_in)
        "leading_edge_wedge_angle": prepared.get("leading_edge_wedge_angle"),  # deg (from YAML)
        "leading_edge_diameter": leading_edge_diameter,                    # m (if radius available)

        # Hub–tip ratios
        "hub_tip_ratio_in": hub_tip_ratio_in,
        "hub_tip_ratio_out": hub_tip_ratio_out,
        "hub_tip_ratio_mean": hub_tip_ratio_mean,
        "hub_tip_ratio_throat": hub_tip_ratio_throat,

        # Machine-level (duplicated per row for convenience)
        "number_of_cascades": number_of_cascades,
        "number_of_stages": number_of_stages,

        # Gauging (computed here if possible)
        "gauging_angle": gauging_angle,
    })

    return _to_native(out)


# ------------------------------------------------------------------
# Organized YAML → dict pipeline
# ------------------------------------------------------------------

_REQUIRED_KEYS = [
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
]

def _validate_row_inputs(row_name: str, row: Dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in row]
    if missing:
        raise KeyError(f"[{row_name}] Missing required keys: {missing}")

def load_config_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def prepare_all_rows(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    geom_section = cfg.get("geometry", [])
    if not isinstance(geom_section, list):
        raise TypeError("`geometry` must be a list of row dictionaries (row-wise structure).")

    prepared_by_name: Dict[str, Dict[str, Any]] = {}
    for item in geom_section:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Each `geometry` list item must be a single-key dict, e.g. {'stator_1': {...}}.")
        row_name, row_data = next(iter(item.items()))
        _validate_row_inputs(row_name, row_data)

        prepared_row = prepare_radial_outflow_geometry(row_data)
        prepared_by_name[row_name] = prepared_row

    return prepared_by_name

def calculate_full_geometries(prepared_by_name: Dict[str, Dict[str, Any]],
                              meta: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    full_by_name: Dict[str, Dict[str, Any]] = {}
    for row_name, prepared_row in prepared_by_name.items():
        full_row = calculate_full_radial_outflow_geometry(prepared_row, meta=meta)
        full_by_name[row_name] = full_row
    return full_by_name

def _infer_stages(geom_items: List[Dict[str, Any]]) -> int:
    """Heuristic: count 'rotor' rows as stages (adjust if your definition differs)."""
    n_rotors = 0
    for item in geom_items:
        _, data = next(iter(item.items()))
        if str(data.get("cascade_type", "")).lower() == "rotor":
            n_rotors += 1
    return n_rotors

def run_radial_outflow_pipeline_from_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns: {row_name: full_geometry_row}
    Additionally injects machine-level counts into each row:
      - number_of_cascades
      - number_of_stages
    """
    cfg = load_config_yaml(path)
    geom_items = cfg.get("geometry", [])
    num_cascades = len(geom_items)
    num_stages   = _infer_stages(geom_items)

    prepared = prepare_all_rows(cfg)
    meta = {"number_of_cascades": num_cascades, "number_of_stages": num_stages}
    radial_outflow_geometry = calculate_full_geometries(prepared, meta=meta)
    return radial_outflow_geometry


# ------------------------------------------------------------
# Sanity checks (user-supplied formulae) — JAX only
# ------------------------------------------------------------

def sanity_check_row(
    row_full: Dict[str, Any],
    u_points: int = 200,
    formulae: List = (),
    display: bool = True,
) -> Dict[str, Any]:
    """
    Run user-supplied sanity checks for a single row using stagger, chord, and theta.
    Each formula in `formulae` must be a callable: ok, name, info = fn(ctx)
    """

    camberline_type = row_full["camberline_type"]
    r_in  = float(row_full["r_in"])
    r_out = float(row_full["r_out"])
    metal_angle1_deg = float(row_full["metal_angle_in"])
    metal_angle2_deg = float(row_full["metal_angle_out"])
    theta0_deg = float(row_full.get("theta0", 0.0))
    chord = float(row_full["chord"])
    stagger_deg = float(row_full["stagger_angle"])

    metal_angle1 = jnp.deg2rad(metal_angle1_deg)
    metal_angle2 = jnp.deg2rad(metal_angle2_deg)
    theta0 = jnp.deg2rad(theta0_deg)

    u = jnp.linspace(0.0, 1.0, int(u_points))
    _x, _y, _r, theta, *_rest = bp.compute_camberline_radial(
        camberline_type, r_in, r_out, metal_angle1, metal_angle2, theta0, u
    )
    d_theta = float(theta[-1] - theta[0])

    ctx = {
        "r_in": r_in,
        "r_out": r_out,
        "metal_angle1_rad": float(metal_angle1),
        "metal_angle2_rad": float(metal_angle2),
        "theta0_rad": float(theta0),
        "theta": theta,
        "d_theta": d_theta,
        "chord": chord,
        "stagger_rad": float(jnp.deg2rad(stagger_deg)),
        "stagger_deg": stagger_deg,
        "row": row_full,
    }

    checks = []
    for i, fn in enumerate(formulae):
        try:
            ok, name, info = fn(ctx)
            checks.append({"name": str(name), "ok": bool(ok), "info": str(info)})
        except Exception as ex:
            checks.append({"name": f"check_{i+1}", "ok": False, "info": f"Exception: {ex}"})

    all_passed = all(item["ok"] for item in checks) if checks else True

    result = {
        "stagger_deg": stagger_deg,
        "chord": chord,
        "d_theta": d_theta,
        "checks": checks,
        "all_passed": all_passed,
    }

    if display:
        line = "-" * 72
        print(line)
        print("Sanity check".center(72))
        print(line)
        print(f"stagger [deg] : {stagger_deg: .6f}")
        print(f"chord   [m]   : {chord: .6f}")
        print(f"d_theta [rad] : {d_theta: .6f}")
        print(line)
        for item in checks:
            mark = "PASS" if item["ok"] else "FAIL"
            print(f"[{mark}] {item['name']}: {item['info']}")
        print(line)
        print(f"ALL PASSED: {all_passed}")
        print(line)

    return _to_native(result)


def sanity_check_all_rows(
    full_by_name: Dict[str, Dict[str, Any]],
    formulae: List = (),
    u_points: int = 200,
    display: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row_name, row_full in full_by_name.items():
        res = sanity_check_row(
            row_full=row_full,
            u_points=u_points,
            formulae=formulae,
            display=display,
        )
        out[row_name] = res
    return out
