# radial_outflow_prepare_and_full.py  (JAX-only)
# ------------------------------------------------------------
# Functions:
#   - prepare_radial_outflow_geometry(row)
#   - calculate_full_radial_outflow_geometry(prepared_row)
#   - load_config_yaml(path)
#   - prepare_all_rows(cfg)
#   - calculate_full_geometries(prepared_by_name)
#   - run_radial_outflow_pipeline_from_yaml(path)
#   - sanity_check_row(...)
#   - sanity_check_all_rows(...)
#
# Uses ONLY the info you provided + two explicit assumptions:
#   (A1) theta0 = 0.0 deg (inlet circumferential reference for camberline)
#   (A2) pitch = 2π * ((r_in + r_out)/2) / N_blades  (pitch at mean radius)
# ------------------------------------------------------------

from typing import Dict, Any, List, Tuple
import yaml
import jax
import jax.numpy as jnp

# Use your camberline implementation module name here:
# import .blade_parametrization as bp
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
    # first to (-pi, pi]
    st = (st + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    # then to (-pi/2, +pi/2]
    st = jnp.where(st >  jnp.pi / 2.0, st - jnp.pi, st)
    st = jnp.where(st <= -jnp.pi / 2.0, st + jnp.pi, st)
    return st

def _to_native(obj):
    """Recursively convert JAX scalars/arrays (and Python containers) to native Python types."""
    # Primitive python types stay as-is
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    # JAX arrays (DeviceArray / jax.Array)
    if isinstance(obj, (jnp.ndarray, jax.Array)):
        arr = obj
        if arr.ndim == 0:
            # Convert 0-d array to float
            return float(arr)
        # For higher dims, go through .tolist() then recurse (to be safe)
        return _to_native(arr.tolist())
    # Containers
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    # Fallback
    return obj


# ------------------------------------------------------------
# Geometry preparation and full-geometry calculations
# ------------------------------------------------------------

def prepare_radial_outflow_geometry(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the given radial-outflow row (from YAML) into evaluation parameters.

    Inputs used (as provided):
      - cascade_type, N_blades
      - r_in, r_out
      - metal_angle_in (deg), metal_angle_out (deg)   # metal angles wrt tangential
      - maximum_thickness, trailing_edge_thickness
      - blade_height_in (b_in), blade_height_out (b_out)
      - camberline_type

    Assumptions (explicit):
      - (A1) theta0 = 0.0 deg
      - (A2) pitch computed at r_ref=(r_in+r_out)/2: pitch = 2π*r_ref/N_blades

    Returns a new dict with derived:
      - chord [m]
      - stagger_angle [deg]
      - pitch [m]
      - aspect_ratio [-] = b_mean / chord
      - plus all original inputs passed through unchanged
    """

    # Required inputs (as provided)
    cascade_type = row["cascade_type"]
    N_blades     = int(row["N_blades"])
    r_in         = jnp.asarray(row["r_in"])
    r_out        = jnp.asarray(row["r_out"])
    metal_angle_in_deg  = float(row["metal_angle_in"])
    metal_angle_out_deg = float(row["metal_angle_out"])
    b_in         = jnp.asarray(row["blade_height_in"])
    b_out        = jnp.asarray(row["blade_height_out"])
    camberline_type = row["camberline_type"]

    # Degrees → radians for camberline
    metal_angle1 = _deg2rad(metal_angle_in_deg)
    metal_angle2 = _deg2rad(metal_angle_out_deg)

    # (A1) inlet circumferential reference
    theta0 = _deg2rad(0.0)

    # Parameter along camberline curve
    u = jnp.linspace(0.0, 1.0, 200)

    # Derive chord & stagger (radians) using your camberline code
    # compute_camberline_radial returns: x, y, r, theta, metal_angle, phi, stagger, chord
    _x, _y, *_rest, theta, stagger_rad, chord = bp.compute_camberline_radial(
        camberline_type, r_in, r_out, metal_angle1, metal_angle2, theta0, u
    )
    stagger_rad = _normalize_stagger_rad(stagger_rad)
    stagger_deg = _rad2deg(stagger_rad)

    # (A2) pitch at mean radius using blade count
    r_ref = 0.5 * (r_in + r_out)
    pitch = 2.0 * jnp.pi * r_ref / float(N_blades)

    # Aspect ratio with mean span
    b_mean = 0.5 * (b_in + b_out)
    aspect_ratio = b_mean / chord

    # Pitch at exit (explicit for throat formula)
    pitch_exit = float(2.0 * jnp.pi * r_out / N_blades)

    # Throat opening (JAX helper in parametrization module)
    throat_opening = float(bp.compute_throat_opening(theta, metal_angle2, pitch_exit))
    throat_area = throat_opening * float(b_out)

    # Build output dict (no mutation of input)
    prepared = dict(row)
    prepared.update({
        "cascade_type": cascade_type,
        "N_blades": N_blades,
        "r_in": r_in,
        "r_out": r_out,
        "metal_angle_in": metal_angle_in_deg,
        "metal_angle_out": metal_angle_out_deg,
        "blade_height_in": b_in,
        "blade_height_out": b_out,
        "camberline_type": camberline_type,

        # Derived here:
        "theta0": 0.0,                        # deg, from (A1)
        "chord": chord,                       # from camberline
        "stagger_angle": stagger_deg,         # deg, from camberline
        "pitch_mean": pitch,                       # from (A2)
        "aspect_ratio": aspect_ratio,         # b_mean / chord
        "throat_opening": throat_opening,     # m
        "throat_area": throat_area,           # m²

        # Pass-through thickness info (provided)
        "maximum_thickness": float(row["maximum_thickness"]),
        "trailing_edge_thickness": float(row["trailing_edge_thickness"]),
    })
    return _to_native(prepared)


def calculate_full_radial_outflow_geometry(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete the geometry using only prepared values and pure geometry relations.

    Adds:
      - A_in, A_out = 2π r b
      - meridional_chord = chord * cos(stagger)
      - flaring_angle = atan( (b_out - b_in) / meridional_chord )  [deg]
      - pitch_chord_ratio = pitch / chord
      - solidity = chord / pitch

    Returns a dict named 'radial_outflow_geometry' with all parameters.
    """
    # Read prepared values (convert to JAX scalars)
    r_in  = jnp.asarray(prepared["r_in"], dtype=jnp.float64)
    r_out = jnp.asarray(prepared["r_out"], dtype=jnp.float64)
    b_in  = jnp.asarray(prepared["blade_height_in"], dtype=jnp.float64)
    b_out = jnp.asarray(prepared["blade_height_out"], dtype=jnp.float64)
    chord = jnp.asarray(prepared["chord"], dtype=jnp.float64)
    pitch = jnp.asarray(prepared["pitch_mean"], dtype=jnp.float64)
    stagger_deg = jnp.asarray(prepared["stagger_angle"], dtype=jnp.float64)

    # Annulus areas for radial outflow
    A_in  = 2.0 * jnp.pi * r_in  * b_in
    A_out = 2.0 * jnp.pi * r_out * b_out

    # Meridional chord and simple flaring analogue
    stagger_rad = jnp.deg2rad(stagger_deg)
    meridional_chord = chord * jnp.cos(stagger_rad)
    eps = jnp.asarray(1e-12)
    denom = jnp.maximum(meridional_chord, eps)
    flaring_angle_deg = jnp.rad2deg(jnp.arctan((b_out - b_in) / denom))

    # Ratios
    pitch_chord_ratio = pitch / chord
    solidity = chord / pitch

    # Package everything
    radial_outflow_geometry = dict(prepared)
    radial_outflow_geometry.update({
        "A_in": A_in,
        "A_out": A_out,
        "meridional_chord": meridional_chord,
        "flaring_angle": flaring_angle_deg,
        "pitch_chord_ratio": pitch_chord_ratio,
        "solidity": solidity,
    })

    return _to_native(radial_outflow_geometry)


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
    """Read the YAML config (no mutation)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def prepare_all_rows(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    From config dict, run `prepare_radial_outflow_geometry` for each geometry row.
    Returns: {row_name: prepared_row_dict}
    """
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

def calculate_full_geometries(prepared_by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    From prepared rows, run `calculate_full_radial_outflow_geometry` per row.
    Returns the final dict: {row_name: radial_outflow_geometry_row}
    """
    full_by_name: Dict[str, Dict[str, Any]] = {}
    for row_name, prepared_row in prepared_by_name.items():
        full_row = calculate_full_radial_outflow_geometry(prepared_row)
        full_by_name[row_name] = full_row
    return full_by_name

def run_radial_outflow_pipeline_from_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper:
      1) load YAML
      2) prepare rows
      3) compute full geometry rows

    Returns: {row_name: radial_outflow_geometry_row}
    """
    cfg = load_config_yaml(path)
    prepared = prepare_all_rows(cfg)
    radial_outflow_geometry = calculate_full_geometries(prepared)
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

    Each formula in `formulae` must be a callable:
        ok, name, info = fn(ctx)

    where ctx contains:
        - r_in, r_out
        - metal_angle1_rad, metal_angle2_rad (radians)
        - theta0_rad
        - theta (jnp.ndarray along camberline)
        - d_theta (float)
        - chord (float)
        - stagger_rad, stagger_deg
        - row (the full row dict)
    """

    # Pull required items
    camberline_type = row_full["camberline_type"]
    r_in  = float(row_full["r_in"])
    r_out = float(row_full["r_out"])
    metal_angle1_deg = float(row_full["metal_angle_in"])
    metal_angle2_deg = float(row_full["metal_angle_out"])
    theta0_deg = float(row_full.get("theta0", 0.0))
    chord = float(row_full["chord"])
    stagger_deg = float(row_full["stagger_angle"])

    # Angles to radians
    metal_angle1 = jnp.deg2rad(metal_angle1_deg)
    metal_angle2 = jnp.deg2rad(metal_angle2_deg)
    theta0 = jnp.deg2rad(theta0_deg)

    # Compute theta along camberline (for checks)
    u = jnp.linspace(0.0, 1.0, int(u_points))
    _x, _y, _r, theta, *_rest = bp.compute_camberline_radial(
        camberline_type, r_in, r_out, metal_angle1, metal_angle2, theta0, u
    )
    d_theta = float(theta[-1] - theta[0])

    # Build context for user formulae (JAX types + Python floats where helpful)
    ctx = {
        "r_in": r_in,
        "r_out": r_out,
        "metal_angle1_rad": float(metal_angle1),
        "metal_angle2_rad": float(metal_angle2),
        "theta0_rad": float(theta0),
        "theta": theta,                              # jnp array
        "d_theta": d_theta,                          # float
        "chord": chord,                              # float
        "stagger_rad": float(jnp.deg2rad(stagger_deg)),
        "stagger_deg": stagger_deg,
        "row": row_full,
    }

    # Evaluate user-supplied checks
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
    """
    Run sanity checks for every row in the full-geometry dict.
    Returns a dict {row_name: sanity_check_row_result}.
    """
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
