"""JAX-friendly B-spline kernels used by Blade2D geometry modules.

This module intentionally keeps the API small and explicit:
- open uniform knot vector generation
- basis evaluation for a scalar parameter
- basis matrix evaluation for vectorized parameters
- curve and first-derivative evaluation
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


Array = jax.Array


def open_uniform_knot_vector(
    n_control: int, degree: int, dtype=jnp.float64
) -> Array:
    """Return an open-uniform knot vector in [0, 1].

    Parameters
    ----------
    n_control
        Number of control points.
    degree
        B-spline degree (3 for cubic, 4 for quartic, ...).
    dtype
        JAX dtype for returned array.
    """
    if n_control < degree + 1:
        raise ValueError(
            f"n_control={n_control} is invalid for degree={degree}; need at least degree+1."
        )

    n_knots = n_control + degree + 1
    n_interior = n_knots - 2 * (degree + 1)

    if n_interior > 0:
        interior = jnp.linspace(0.0, 1.0, n_interior + 2, dtype=dtype)[1:-1]
        return jnp.concatenate(
            [
                jnp.zeros(degree + 1, dtype=dtype),
                interior,
                jnp.ones(degree + 1, dtype=dtype),
            ]
        )

    return jnp.concatenate(
        [jnp.zeros(degree + 1, dtype=dtype), jnp.ones(degree + 1, dtype=dtype)]
    )


def _n_control_from_knots(knots: Array, degree: int) -> int:
    return int(knots.shape[0] - degree - 1)


def basis_functions_at_u(
    u: Array | float, degree: int, knots: Array, n_control: Optional[int] = None
) -> Array:
    """Evaluate all B-spline basis functions N_i,degree(u) for one u."""
    if n_control is None:
        n_control = _n_control_from_knots(knots, degree)
    if n_control <= 0:
        raise ValueError("n_control must be positive.")

    u = jnp.asarray(u, dtype=knots.dtype)
    one = jnp.asarray(1.0, dtype=knots.dtype)
    zero = jnp.asarray(0.0, dtype=knots.dtype)

    left = knots[:n_control]
    right = knots[1 : n_control + 1]

    N = jnp.where((u >= left) & (u < right), one, zero)

    # Right-endpoint convention for open knot vectors: N_{n-1,p}(u=1)=1.
    is_u_max = jnp.isclose(u, knots[-1])
    N = N.at[n_control - 1].set(jnp.where(is_u_max, one, N[n_control - 1]))

    for p in range(1, degree + 1):
        left_den = knots[p : p + n_control] - knots[:n_control]
        right_den = knots[p + 1 : p + n_control + 1] - knots[1 : n_control + 1]

        N_next = jnp.concatenate([N[1:], jnp.array([zero], dtype=N.dtype)])

        left_term = jnp.where(
            left_den > 0.0, ((u - knots[:n_control]) / left_den) * N, zero
        )
        right_term = jnp.where(
            right_den > 0.0,
            ((knots[p + 1 : p + n_control + 1] - u) / right_den) * N_next,
            zero,
        )

        N = left_term + right_term

    return N


def basis_matrix(
    u: Array, degree: int, knots: Array, n_control: Optional[int] = None
) -> Array:
    """Evaluate basis matrix B where B[j, i] = N_i,degree(u_j)."""
    u = jnp.asarray(u, dtype=knots.dtype)
    if u.ndim != 1:
        raise ValueError("u must be a 1D array.")
    if n_control is None:
        n_control = _n_control_from_knots(knots, degree)

    return jax.vmap(
        lambda ui: basis_functions_at_u(ui, degree=degree, knots=knots, n_control=n_control)
    )(u)


def basis_derivative_at_u(
    u: Array | float, degree: int, knots: Array, n_control: Optional[int] = None
) -> Array:
    """Evaluate dN_i,degree/du at one u for all i."""
    if degree < 1:
        raise ValueError("degree must be >= 1 for derivative evaluation.")
    if n_control is None:
        n_control = _n_control_from_knots(knots, degree)

    n_low = n_control + 1
    N_low = basis_functions_at_u(u, degree - 1, knots, n_control=n_low)

    zero = jnp.asarray(0.0, dtype=knots.dtype)
    out = []
    for i in range(n_control):
        den_l = knots[i + degree] - knots[i]
        den_r = knots[i + degree + 1] - knots[i + 1]
        a = jnp.where(den_l > 0.0, degree * N_low[i] / den_l, zero)
        b = jnp.where(den_r > 0.0, degree * N_low[i + 1] / den_r, zero)
        out.append(a - b)
    return jnp.stack(out)


def basis_derivative_matrix(
    u: Array, degree: int, knots: Array, n_control: Optional[int] = None
) -> Array:
    """Evaluate derivative basis matrix dB where dB[j, i] = dN_i,degree/du at u_j."""
    u = jnp.asarray(u, dtype=knots.dtype)
    if u.ndim != 1:
        raise ValueError("u must be a 1D array.")
    if n_control is None:
        n_control = _n_control_from_knots(knots, degree)

    return jax.vmap(
        lambda ui: basis_derivative_at_u(
            ui, degree=degree, knots=knots, n_control=n_control
        )
    )(u)


def curve_points(
    control_points: Array, u: Array, degree: int = 3, knots: Optional[Array] = None
) -> Array:
    """Evaluate B-spline curve points for control points at parameter vector u."""
    control_points = jnp.asarray(control_points)
    if control_points.ndim != 2:
        raise ValueError("control_points must have shape (n_control, dim).")

    n_control = int(control_points.shape[0])
    if knots is None:
        knots = open_uniform_knot_vector(n_control, degree, dtype=control_points.dtype)

    B = basis_matrix(u, degree=degree, knots=knots, n_control=n_control)
    return B @ control_points


def curve_derivative_points(
    control_points: Array, u: Array, degree: int = 3, knots: Optional[Array] = None
) -> Array:
    """Evaluate first derivative dC/du for B-spline curve at parameter vector u."""
    control_points = jnp.asarray(control_points)
    if control_points.ndim != 2:
        raise ValueError("control_points must have shape (n_control, dim).")

    n_control = int(control_points.shape[0])
    if knots is None:
        knots = open_uniform_knot_vector(n_control, degree, dtype=control_points.dtype)

    dB = basis_derivative_matrix(u, degree=degree, knots=knots, n_control=n_control)
    return dB @ control_points

