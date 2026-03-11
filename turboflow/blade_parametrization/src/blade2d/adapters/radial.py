"""Radial-outflow geometry adapter for curvature-based camberline generation."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from ..core.camber_curvature import (
    compute_curvature_camberline,
    default_curvature_control_points,
)
from ..core.chord import resolve_radial_chord
from ..core.section_builder import build_section_from_results
from ..core.thickness_quartic import thickness_from_geometry


def curvature_camberline_from_radial_geometry(
    geometry: dict,
    *,
    n_points: int = 161,
    n_control: int = 8,
    curvature_control_points: Optional[jnp.ndarray] = None,
    curvature_control_u: Optional[jnp.ndarray] = None,
    degree: int = 3,
) -> dict:
    """Build curvature-based camberline from a radial geometry dictionary.

    Expected keys (minimum):
    - metal_angle_in (deg)
    - metal_angle_out (deg)
    Optional:
    - chord (m), if already known
    - radius_mean_in, radius_mean_out and theta endpoints for chord-from-theta
    - meridional_chord (m), used as projection with computed stagger
    """
    if "metal_angle_in" not in geometry or "metal_angle_out" not in geometry:
        raise KeyError("geometry must include 'metal_angle_in' and 'metal_angle_out'.")

    metal_in = float(geometry["metal_angle_in"])
    metal_out = float(geometry["metal_angle_out"])
    total_camber = jnp.deg2rad(metal_out - metal_in)

    if curvature_control_points is None:
        curvature_control_points = default_curvature_control_points(
            n_control=n_control, total_camber_rad=float(total_camber)
        )
    else:
        curvature_control_points = jnp.asarray(curvature_control_points, dtype=jnp.float64)

    if curvature_control_u is not None:
        curvature_control_u = jnp.asarray(curvature_control_u, dtype=jnp.float64)

    u = jnp.linspace(0.0, 1.0, n_points, dtype=jnp.float64)
    chord_projection = geometry.get("meridional_chord", None)

    out = compute_curvature_camberline(
        u=u,
        metal_angle_in_rad=float(jnp.deg2rad(metal_in)),
        metal_angle_out_rad=float(jnp.deg2rad(metal_out)),
        curvature_cp=curvature_control_points,
        degree=degree,
        u_cp=curvature_control_u,
        chord_projection=None if chord_projection is None else float(chord_projection),
        wing_flag=0,
    )
    chord_value, chord_source = resolve_radial_chord(
        geometry,
        stagger_angle_rad=out["stagger_angle_rad"],
    )
    out["chord"] = chord_value
    out["chord_source"] = chord_source
    out["cascade_type"] = geometry.get("cascade_type", None)
    out["curvature_control_points"] = curvature_control_points
    out["curvature_control_u"] = curvature_control_u
    return out


def thickness_profile_from_radial_geometry(
    geometry: dict,
    *,
    n_points: int = 161,
    n_control: int = 11,
    u: Optional[jnp.ndarray] = None,
    chord_override: Optional[float] = None,
    enable_le_blend: bool = False,
    lambda_smooth_d2: float = 5e-4,
    lambda_smooth_d1: float = 1e-5,
) -> dict:
    """Build quartic B-spline thickness profile from radial geometry dictionary."""
    if u is None:
        u = jnp.linspace(0.0, 1.0, n_points, dtype=jnp.float64)
    out = thickness_from_geometry(
        geometry,
        u=u,
        n_control=n_control,
        chord_override=chord_override,
        enable_le_blend=enable_le_blend,
        lambda_smooth_d2=lambda_smooth_d2,
        lambda_smooth_d1=lambda_smooth_d1,
    )
    out["cascade_type"] = geometry.get("cascade_type", None)
    return out


def section_from_radial_geometry(
    geometry: dict,
    *,
    n_points: int = 161,
    camber_n_control: int = 8,
    thickness_n_control: int = 11,
    curvature_control_points: Optional[jnp.ndarray] = None,
    curvature_control_u: Optional[jnp.ndarray] = None,
    degree: int = 3,
    enable_le_blend: bool = False,
    thickness_lambda_smooth_d2: float = 5e-4,
    thickness_lambda_smooth_d1: float = 1e-5,
) -> dict:
    """Generate a complete 2D section (top/bottom/camber) from radial geometry dict."""
    camber = curvature_camberline_from_radial_geometry(
        geometry,
        n_points=n_points,
        n_control=camber_n_control,
        curvature_control_points=curvature_control_points,
        curvature_control_u=curvature_control_u,
        degree=degree,
    )
    if camber.get("chord", None) is None:
        raise KeyError(
            "Unable to resolve radial chord for section construction. "
            "Provide one of: 'chord', ('radius_mean_in'/'radius_mean_out' with "
            "'theta_in'/'theta_out' or *_rad variants), or 'meridional_chord'."
        )
    thickness = thickness_profile_from_radial_geometry(
        geometry,
        n_points=n_points,
        n_control=thickness_n_control,
        u=camber["u"],
        chord_override=camber.get("chord", None),
        enable_le_blend=enable_le_blend,
        lambda_smooth_d2=thickness_lambda_smooth_d2,
        lambda_smooth_d1=thickness_lambda_smooth_d1,
    )
    section = build_section_from_results(
        camber,
        thickness,
    )
    section["cascade_type"] = geometry.get("cascade_type", None)
    return section
