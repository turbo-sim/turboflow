"""Curvature-based camberline construction (paper + T-Blade3-inspired).

This module provides a practical Python/JAX implementation of the core idea:
1) define a smooth camber second-derivative profile using a cubic B-spline
2) integrate to obtain slope and camber
3) enforce inlet/outlet angle consistency via a scalar scaling factor
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp

from .bspline import curve_points, open_uniform_knot_vector, basis_matrix


def cumulative_trapezoid(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Cumulative trapezoidal integration from x[0] to each x[i]."""
    y = jnp.asarray(y)
    x = jnp.asarray(x)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same length.")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples for integration.")

    dx = x[1:] - x[:-1]
    trap = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.array([0.0], dtype=y.dtype), jnp.cumsum(trap)])


def _angle_residual(k: float, a1: float, b1: float, total_camber: float) -> float:
    """Residual for angle-difference constraint at section ends."""
    s_in = k * (0.0 - b1)
    s_out = k * (a1 - b1)
    return float(np.arctan(s_out) - np.arctan(s_in) - total_camber)


def _secant_refine(a1: float, b1: float, total_camber: float, k0: float, k1: float) -> float:
    """Secant refinement on the angle residual."""
    f0 = _angle_residual(k0, a1, b1, total_camber)
    f1 = _angle_residual(k1, a1, b1, total_camber)
    if not np.isfinite(f0) or not np.isfinite(f1):
        return k1

    for _ in range(50):
        den = f1 - f0
        if abs(den) < 1e-14:
            break
        k2 = k1 - f1 * (k1 - k0) / den
        f2 = _angle_residual(k2, a1, b1, total_camber)
        if not np.isfinite(f2):
            break
        if abs(f2) < 1e-12:
            return float(k2)
        k0, f0 = k1, f1
        k1, f1 = float(k2), float(f2)
    return float(k1)


def choose_scaling_factor(a1: float, b1: float, total_camber: float) -> float:
    """Choose scaling factor k using T-Blade3-style quadratic + robust fallback."""
    if abs(total_camber) < 1e-14:
        return 0.0

    p = (a1 * b1) - (b1 * b1)
    tan_tc = float(np.tan(total_camber))
    candidates = []

    if abs(p) > 1e-14 and abs(tan_tc) > 1e-14:
        det = (a1 * a1) + (4.0 * p * (tan_tc * tan_tc))
        if det >= 0.0:
            sq = np.sqrt(det)
            k1 = (-a1 + sq) / (2.0 * p * tan_tc)
            k2 = (-a1 - sq) / (2.0 * p * tan_tc)
            for k in (k1, k2):
                if np.isfinite(k):
                    candidates.append(float(k))

    # Evaluate candidates and refine the best pair.
    if candidates:
        scored = [(abs(_angle_residual(k, a1, b1, total_camber)), k) for k in candidates]
        scored.sort(key=lambda t: t[0])
        k_best = scored[0][1]
        if len(scored) > 1:
            k_other = scored[1][1]
            return _secant_refine(a1, b1, total_camber, k_best, k_other)
        # Single candidate: use coarse companion and refine.
        k_probe = k_best * 0.5 if abs(k_best) > 1e-8 else 1.0
        return _secant_refine(a1, b1, total_camber, k_probe, k_best)

    # Fallback: coarse scan + secant refine around best two points.
    scan = np.concatenate(
        [
            np.linspace(-200.0, -0.05, 1200),
            np.linspace(0.05, 200.0, 1200),
        ]
    )
    vals = np.array([abs(_angle_residual(k, a1, b1, total_camber)) for k in scan])
    i0 = int(np.argmin(vals))
    k0 = float(scan[i0])
    if i0 == 0:
        k1 = float(scan[1])
    elif i0 == len(scan) - 1:
        k1 = float(scan[-2])
    else:
        left = i0 - 1
        right = i0 + 1
        k1 = float(scan[left] if vals[left] < vals[right] else scan[right])
    return _secant_refine(a1, b1, total_camber, k0, k1)


def _evaluate_curvature_profile(
    u: jnp.ndarray,
    curvature_cp: jnp.ndarray,
    degree: int = 3,
    u_cp: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Evaluate unscaled curvature profile over u from control points.

    If `u_cp` is None, control points are treated as direct function controls in [0, 1]
    using a standard open-uniform basis.
    If `u_cp` is provided, a parametric B-spline (u(t), y(t)) is sampled and projected
    to y(u) by interpolation.
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    cp = jnp.asarray(curvature_cp, dtype=jnp.float64)

    if cp.ndim != 1:
        raise ValueError("curvature_cp must be a 1D array.")
    if cp.shape[0] < degree + 1:
        raise ValueError("Need at least degree+1 control points.")

    if u_cp is None:
        knots = open_uniform_knot_vector(cp.shape[0], degree, dtype=jnp.float64)
        B = basis_matrix(u, degree=degree, knots=knots, n_control=cp.shape[0])
        return B @ cp

    u_cp = jnp.asarray(u_cp, dtype=jnp.float64)
    if u_cp.ndim != 1 or u_cp.shape != cp.shape:
        raise ValueError("u_cp must be 1D and have same length as curvature_cp.")
    if not bool(jnp.all(u_cp[1:] >= u_cp[:-1])):
        raise ValueError("u_cp must be non-decreasing.")

    ctrl = jnp.stack([u_cp, cp], axis=1)
    t_dense = jnp.linspace(0.0, 1.0, 2000, dtype=jnp.float64)
    curve = curve_points(ctrl, u=t_dense, degree=degree)
    x_dense = np.asarray(curve[:, 0])
    y_dense = np.asarray(curve[:, 1])
    order = np.argsort(x_dense)
    x_sorted = x_dense[order]
    y_sorted = y_dense[order]
    x_unique, idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[idx]
    y_u = np.interp(np.asarray(u), x_unique, y_unique)
    return jnp.asarray(y_u, dtype=jnp.float64)


def default_curvature_control_points(
    n_control: int,
    total_camber_rad: float,
    peak_location: float = 0.35,
) -> jnp.ndarray:
    """Generate a smooth default curvature-control vector for auto mode."""
    if n_control < 4:
        raise ValueError("n_control must be >= 4.")
    x = jnp.linspace(0.0, 1.0, n_control, dtype=jnp.float64)

    w1 = 0.17
    w2 = 0.13
    g1 = jnp.exp(-((x - peak_location) ** 2) / (2.0 * (w1 * w1)))
    g2 = 0.45 * jnp.exp(-((x - 0.85) ** 2) / (2.0 * (w2 * w2)))
    shape = g1 + g2
    shape = shape / jnp.maximum(jnp.max(jnp.abs(shape)), 1e-12)

    sign = jnp.where(total_camber_rad >= 0.0, 1.0, -1.0)
    cp = sign * shape
    # Slight endpoint flattening for cleaner LE/TE transitions.
    cp = cp.at[0].set(cp[1])
    cp = cp.at[-1].set(cp[-2])
    return cp


def compute_curvature_camberline(
    u: jnp.ndarray,
    metal_angle_in_rad: float,
    metal_angle_out_rad: float,
    curvature_cp: jnp.ndarray,
    *,
    degree: int = 3,
    u_cp: Optional[jnp.ndarray] = None,
    chord_projection: Optional[float] = None,
    wing_flag: int = 0,
) -> dict:
    """Compute a curvature-based 2D camberline in (u, v) coordinates.

    Parameters
    ----------
    u
        Chordwise coordinate vector in [0, 1].
    metal_angle_in_rad, metal_angle_out_rad
        Inlet/outlet metal angles in radians.
    curvature_cp
        Control points for unscaled second-derivative shape.
    degree
        B-spline degree used for curvature shape (default: 3).
    u_cp
        Optional u-locations for curvature control points.
    chord_projection
        Axial/meridional projection used to infer actual chord from stagger.
    wing_flag
        Same convention as T-Blade3:
        - 0: total camber = a_out - a_in
        - 1: total camber = a_out and stagger = a_in
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    if u.ndim != 1:
        raise ValueError("u must be a 1D array.")
    if not bool(jnp.all(u[1:] >= u[:-1])):
        raise ValueError("u must be sorted ascending.")

    base_curv = _evaluate_curvature_profile(u, curvature_cp, degree=degree, u_cp=u_cp)
    int_curv = cumulative_trapezoid(base_curv, u)   # A(u)
    int_slope = cumulative_trapezoid(int_curv, u)   # B(u)

    a1 = float(int_curv[-1])
    b1 = float(int_slope[-1])
    total_camber = (
        float(metal_angle_out_rad - metal_angle_in_rad)
        if wing_flag == 0
        else float(metal_angle_out_rad)
    )

    k = choose_scaling_factor(a1, b1, total_camber)
    k_arr = jnp.asarray(k, dtype=jnp.float64)

    curvature = k_arr * base_curv
    slope = k_arr * (int_curv - b1)
    camber = k_arr * (int_slope - (u * b1))

    if wing_flag == 0:
        stagger = float(metal_angle_in_rad - np.arctan(float(slope[0])))
    else:
        stagger = float(metal_angle_in_rad)

    angle_residual = _angle_residual(k, a1, b1, total_camber)

    chord = None
    if chord_projection is not None:
        chord = float(chord_projection / max(abs(np.cos(stagger)), 1e-12))

    return {
        "u": u,
        "camber": camber,
        "slope": slope,
        "curvature": curvature,
        "scaling_factor": float(k),
        "stagger_angle_rad": float(stagger),
        "stagger_angle_deg": float(np.rad2deg(stagger)),
        "chord_projection": None if chord_projection is None else float(chord_projection),
        "chord": chord,
        "total_camber_target_rad": float(total_camber),
        "angle_residual_rad": float(angle_residual),
        "u_end": float(u[-1]),
        "A1_int_curvature": float(a1),
        "B1_int_slope": float(b1),
    }

