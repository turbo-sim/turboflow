"""Chord-resolution helpers for axial and radial geometry adapters."""

from __future__ import annotations

from math import cos, radians, sqrt
from typing import Optional, Tuple


def chord_from_theta(
    radius_in: float,
    radius_out: float,
    theta_in_rad: float,
    theta_out_rad: float,
) -> float:
    """Chord from two polar end points using the law of cosines."""
    r1 = float(radius_in)
    r2 = float(radius_out)
    t1 = float(theta_in_rad)
    t2 = float(theta_out_rad)
    value = (r1 * r1) + (r2 * r2) - (2.0 * r1 * r2 * cos(t2 - t1))
    return float(sqrt(max(value, 0.0)))


def _try_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _chord_from_projection(projection: Optional[float], stagger_angle_rad: Optional[float]) -> Optional[float]:
    if projection is None or stagger_angle_rad is None:
        return None
    proj = float(projection)
    if proj <= 0.0:
        return None
    cstag = abs(cos(float(stagger_angle_rad)))
    if cstag < 1e-12:
        return None
    return float(proj / cstag)


def resolve_axial_chord(
    geometry: dict,
    *,
    stagger_angle_rad: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Resolve axial chord with explicit chord taking precedence."""
    chord = _try_float(geometry.get("chord", None))
    if chord is not None and chord > 0.0:
        return chord, "input_chord"

    projection = _try_float(geometry.get("chord_axial", geometry.get("meridional_chord", None)))
    chord_from_proj = _chord_from_projection(projection, stagger_angle_rad)
    if chord_from_proj is not None:
        return chord_from_proj, "projection_plus_stagger"

    return None, "unresolved"


def _extract_theta_pair_rad(geometry: dict) -> Tuple[Optional[float], Optional[float]]:
    theta_in_rad = _try_float(geometry.get("theta_in_rad", geometry.get("theta1_rad", None)))
    theta_out_rad = _try_float(geometry.get("theta_out_rad", geometry.get("thetaN_rad", None)))
    if theta_in_rad is not None and theta_out_rad is not None:
        return theta_in_rad, theta_out_rad

    theta_in_deg = _try_float(geometry.get("theta_in", geometry.get("theta1", None)))
    theta_out_deg = _try_float(geometry.get("theta_out", geometry.get("thetaN", None)))
    if theta_in_deg is not None and theta_out_deg is not None:
        return radians(theta_in_deg), radians(theta_out_deg)

    return None, None


def resolve_radial_chord(
    geometry: dict,
    *,
    stagger_angle_rad: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Resolve radial chord from available geometric information.

    Resolution priority:
    1) `chord` if explicitly provided
    2) law-of-cosines from radii + theta endpoints
    3) meridional projection + stagger
    """
    chord = _try_float(geometry.get("chord", None))
    if chord is not None and chord > 0.0:
        return chord, "input_chord"

    radius_in = _try_float(geometry.get("radius_mean_in", geometry.get("radius_in", None)))
    radius_out = _try_float(geometry.get("radius_mean_out", geometry.get("radius_out", None)))
    theta_in_rad, theta_out_rad = _extract_theta_pair_rad(geometry)
    if (
        radius_in is not None
        and radius_out is not None
        and radius_in > 0.0
        and radius_out > 0.0
        and theta_in_rad is not None
        and theta_out_rad is not None
    ):
        return chord_from_theta(radius_in, radius_out, theta_in_rad, theta_out_rad), "radius_theta"

    projection = _try_float(geometry.get("meridional_chord", None))
    chord_from_proj = _chord_from_projection(projection, stagger_angle_rad)
    if chord_from_proj is not None:
        return chord_from_proj, "projection_plus_stagger"

    return None, "unresolved"
