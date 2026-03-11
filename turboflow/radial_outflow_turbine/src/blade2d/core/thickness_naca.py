"""JAX-friendly modified NACA thickness distribution.

This module implements the same 5-coefficient system commonly used in
legacy blade parameterizations:
- LE radius constraint
- max-thickness value + zero slope at max-thickness location
- finite TE thickness + TE wedge slope
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


@jax.jit
def _solve_naca5_coefficients(
    max_thickness_ratio: float,
    max_thickness_location: float,
    trailing_edge_thickness_ratio: float,
    trailing_edge_wedge_angle_rad: float,
    leading_edge_radius_ratio: float,
) -> jax.Array:
    """Solve modified NACA 5-term coefficient vector [A, B, C, D, E]."""
    xm = jnp.asarray(max_thickness_location, dtype=jnp.float64)
    tmax = jnp.asarray(max_thickness_ratio, dtype=jnp.float64)
    tte = jnp.asarray(trailing_edge_thickness_ratio, dtype=jnp.float64)
    wedge = jnp.asarray(trailing_edge_wedge_angle_rad, dtype=jnp.float64)
    rle = jnp.asarray(leading_edge_radius_ratio, dtype=jnp.float64)

    lhs = jnp.zeros((5, 5), dtype=jnp.float64)
    rhs = jnp.zeros((5,), dtype=jnp.float64)

    lhs = lhs.at[0, :].set(jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64))
    rhs = rhs.at[0].set(jnp.sqrt(2.0 * rle))

    lhs = lhs.at[1, :].set(jnp.asarray([jnp.sqrt(xm), xm, xm**2, xm**3, xm**4], dtype=jnp.float64))
    rhs = rhs.at[1].set(0.5 * tmax)

    lhs = lhs.at[2, :].set(
        jnp.asarray(
            [0.5 / jnp.sqrt(jnp.maximum(xm, 1e-12)), 1.0, 2.0 * xm, 3.0 * xm**2, 4.0 * xm**3],
            dtype=jnp.float64,
        )
    )
    rhs = rhs.at[2].set(0.0)

    lhs = lhs.at[3, :].set(jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))
    rhs = rhs.at[3].set(0.5 * tte)

    lhs = lhs.at[4, :].set(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
    rhs = rhs.at[4].set(-jnp.tan(0.5 * wedge))

    return jnp.linalg.solve(lhs, rhs)


@jax.jit
def _evaluate_half_thickness_ratio(x: jax.Array, coeff: jax.Array) -> jax.Array:
    x = jnp.clip(x, 0.0, 1.0)
    a, b, c, d, e = coeff
    return (
        (a * jnp.sqrt(jnp.maximum(x, 0.0)))
        + (b * x)
        + (c * x**2)
        + (d * x**3)
        + (e * x**4)
    )


def naca_modified_thickness(
    u: jax.Array,
    *,
    max_thickness_ratio: float,
    max_thickness_location: float,
    trailing_edge_thickness_ratio: float = 0.0,
    leading_edge_radius_ratio: Optional[float] = None,
    trailing_edge_wedge_angle_deg: Optional[float] = None,
) -> dict:
    """Compute modified-NACA half/full thickness profile as nondimensional ratios."""
    u = jnp.asarray(u, dtype=jnp.float64)
    if u.ndim != 1:
        raise ValueError("u must be a 1D array.")
    if u.shape[0] < 8:
        raise ValueError("u must have at least 8 samples.")
    if not bool(jnp.all(u[1:] >= u[:-1])):
        raise ValueError("u must be sorted ascending.")

    xm = float(max_thickness_location)
    tmax = float(max_thickness_ratio)
    tte = float(trailing_edge_thickness_ratio)

    if not (0.02 <= xm <= 0.98):
        raise ValueError("max_thickness_location must be in [0.02, 0.98].")
    if tmax <= 0.0:
        raise ValueError("max_thickness_ratio must be > 0.")
    if tte < 0.0:
        raise ValueError("trailing_edge_thickness_ratio must be >= 0.")
    if tte >= tmax:
        raise ValueError("trailing_edge_thickness_ratio must be smaller than max_thickness_ratio.")

    # If LE radius is not given, pick a conservative value.
    if leading_edge_radius_ratio is None:
        rle = max(1e-8, 0.0125 * tmax * tmax)
    else:
        rle = float(leading_edge_radius_ratio)
    if rle <= 0.0:
        raise ValueError("leading_edge_radius_ratio must be > 0.")

    wedge_deg = 0.0 if trailing_edge_wedge_angle_deg is None else float(trailing_edge_wedge_angle_deg)
    wedge_rad = float(jnp.deg2rad(wedge_deg))

    coeff = _solve_naca5_coefficients(
        max_thickness_ratio=tmax,
        max_thickness_location=xm,
        trailing_edge_thickness_ratio=tte,
        trailing_edge_wedge_angle_rad=wedge_rad,
        leading_edge_radius_ratio=rle,
    )

    thk_half = _evaluate_half_thickness_ratio(u, coeff)
    thk_half = jnp.maximum(thk_half, 0.0)
    thk_half = thk_half.at[0].set(0.0)
    thk_half = thk_half.at[-1].set(0.5 * tte)

    thk_full = 2.0 * thk_half
    dthk_du = jnp.gradient(thk_half, u)

    i_max = int(jnp.argmax(thk_half))
    u_max_actual = float(u[i_max])
    max_thickness_ratio_actual = float(thk_full[i_max])

    return {
        "u": u,
        "thickness_half_ratio": thk_half,
        "thickness_full_ratio": thk_full,
        "dthickness_half_du": dthk_du,
        "degree": 4,
        "n_control": 5,
        "control_points": coeff,
        "target_max_thickness_ratio": tmax,
        "target_max_thickness_location": xm,
        "target_trailing_edge_thickness_ratio": tte,
        "target_leading_edge_radius_ratio": rle,
        "target_trailing_edge_wedge_angle_deg": wedge_deg,
        "max_thickness_ratio_actual": max_thickness_ratio_actual,
        "max_thickness_location_actual": u_max_actual,
    }
