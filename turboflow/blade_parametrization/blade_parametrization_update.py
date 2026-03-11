"""JAX-only blade parametrization kernels for radial/axial cascades.

This module intentionally contains only numerical functions (no plotting,
no YAML/pipeline code).

Added capabilities over legacy structure:
- `camberline_type="curvature_based"` for radial and cartesian camberlines
- `thickness_model="B_spline"` (quartic-style constrained B-spline thickness)
- `thickness_model="Denton"` (parametric Denton-style thickness law)

The original camberline/thickness paths are preserved.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax import lax

# src = Path(__file__).resolve().parent / "src"
# if str(src) not in sys.path:
#     sys.path.insert(0, str(src))
# from blade2d.adapters.radial import section_from_radial_geometry
# from blade2d.adapters.axial import section_from_axial_geometry

from .src.blade2d.adapters.radial import section_from_radial_geometry
from .src.blade2d.adapters.axial import section_from_axial_geometry


# --- geometry helpers -------------------------------------------------------


@jax.jit
def rotate_counterclockwise_2D(x, y, theta):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    X = ct * x - st * y
    Y = st * x + ct * y
    return X, Y


@jax.jit
def _chord_from_theta(r1, r2, theta1, thetaN):
    return jnp.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * jnp.cos(thetaN - theta1))


@jax.jit
def _cumtrapz(y, x):
    dx = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.array([0.0], dtype=y.dtype), jnp.cumsum(area)])


# --- B-spline basis helpers -------------------------------------------------


def open_uniform_knot_vector(n_control: int, degree: int, dtype=jnp.float64):
    if n_control < degree + 1:
        raise ValueError("n_control must be >= degree+1.")
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
    return jnp.concatenate([jnp.zeros(degree + 1, dtype=dtype), jnp.ones(degree + 1, dtype=dtype)])


def basis_functions_at_u(u, degree: int, knots, n_control: Optional[int] = None):
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)
    one = jnp.asarray(1.0, dtype=knots.dtype)
    zero = jnp.asarray(0.0, dtype=knots.dtype)
    u = jnp.asarray(u, dtype=knots.dtype)

    left = knots[:n_control]
    right = knots[1 : n_control + 1]
    N = jnp.where((u >= left) & (u < right), one, zero)
    N = N.at[n_control - 1].set(jnp.where(jnp.isclose(u, knots[-1]), one, N[n_control - 1]))

    for p in range(1, degree + 1):
        left_den = knots[p : p + n_control] - knots[:n_control]
        right_den = knots[p + 1 : p + n_control + 1] - knots[1 : n_control + 1]
        N_next = jnp.concatenate([N[1:], jnp.array([zero], dtype=N.dtype)])
        left_term = jnp.where(left_den > 0.0, ((u - knots[:n_control]) / left_den) * N, zero)
        right_term = jnp.where(
            right_den > 0.0,
            ((knots[p + 1 : p + n_control + 1] - u) / right_den) * N_next,
            zero,
        )
        N = left_term + right_term
    return N


def basis_matrix(u, degree: int, knots, n_control: Optional[int] = None):
    u = jnp.asarray(u, dtype=knots.dtype)
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)
    return jax.vmap(lambda ui: basis_functions_at_u(ui, degree, knots, n_control))(u)


def basis_derivative_at_u(u, degree: int, knots, n_control: Optional[int] = None):
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)

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


# --- thickness models -------------------------------------------------------


@jax.jit
def compute_thickness_distribution_NACA_modified(
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
):
    LHS = jnp.zeros((5, 5))
    RHS = jnp.zeros((5, 1))
    i = 0
    LHS = LHS.at[i, :].set(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    RHS = RHS.at[i, 0].set(jnp.sqrt(2.0 * (radius_leading / chord)))

    i += 1
    row = jnp.array([jnp.sqrt(loc_max), loc_max, loc_max**2, loc_max**3, loc_max**4])
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(0.5 * (thickness_max / chord))

    i += 1
    row = jnp.array(
        [
            0.5 / jnp.sqrt(jnp.maximum(loc_max, 1e-12)),
            1.0,
            2.0 * loc_max,
            3.0 * loc_max**2,
            4.0 * loc_max**3,
        ]
    )
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(0.0)

    i += 1
    row = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(0.5 * (thickness_trailing / chord))

    i += 1
    slope_trailing = -jnp.tan(wedge_trailing / 2.0)
    row = jnp.array([0.5, 1.0, 2.0, 3.0, 4.0])
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(slope_trailing)

    coeff = jnp.linalg.solve(LHS, RHS).reshape((-1,))
    A, B, C, D, E = coeff

    x_norm = jnp.clip(x_norm, 0.0, 1.0)
    return chord * (
        A * jnp.sqrt(jnp.maximum(x_norm, 0.0))
        + B * x_norm
        + C * x_norm**2
        + D * x_norm**3
        + E * x_norm**4
    )


@jax.jit
def compute_thickness_distribution_Denton(
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    thickness_leading,
    thickness_shape_exponent=2.0,
    radius_leading=0.0,
):
    """Denton full-thickness law with STAGEN-style LE thinning.

    Body law mirrors STAGEN INTYPE=1:
      x_trans = x ** POWER, POWER = log(0.5)/log(x_tmax)
      t_lin(x) = t_le + x*(t_te - t_le)
      t_body(x) = t_lin + t_add * (1 - |x_trans-0.5|^p / 0.5^p)

    LE closure mirrors STAGEN's `FACLETE` idea:
    - form the body thickness first,
    - apply an elliptic-like thinning factor near LE over x <= xmod_le,
    - enforce t(0)=0.
    """
    eps = 1e-12
    x = jnp.clip(x_norm, 0.0, 1.0)
    x_tmax = jnp.clip(loc_max, 0.02, 0.98)
    power = jnp.log(0.5) / jnp.log(jnp.maximum(x_tmax, eps))
    x_trans = x**power

    t_lin = thickness_leading + x * (thickness_trailing - thickness_leading)
    t_add = thickness_max - (
        thickness_leading + x_tmax * (thickness_trailing - thickness_leading)
    )

    p = jnp.maximum(thickness_shape_exponent, 1e-6)
    bell = 1.0 - (jnp.abs(x_trans - 0.5) ** p) / jnp.maximum((0.5**p), eps)
    t_body = t_lin + t_add * bell

    # STAGEN-like LE thinning factor:
    # FAC_LE = sqrt(1 - |x/xmod_le - 1|^LE_EXP) for x <= xmod_le, else 1.
    # Use LE radius (or LE thickness fallback) to infer xmod_le.
    r_eff = jnp.maximum(radius_leading, 0.5 * jnp.maximum(thickness_leading, 0.0))
    xmod_upper = jnp.minimum(0.30, 0.8 * x_tmax)
    xmod_le = jnp.clip(2.0 * r_eff / jnp.maximum(chord, eps), 0.01, xmod_upper)
    le_exp = 3.0
    x_mle = x / jnp.maximum(xmod_le, eps)
    fac_le_inner = jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.abs(x_mle - 1.0) ** le_exp))
    fac_le = jnp.where(x <= xmod_le, fac_le_inner, 1.0)
    t_full = t_body * fac_le

    # Keep positive and enforce exact endpoint values.
    t_full = jnp.maximum(t_full, 0.0)
    t_full = t_full.at[0].set(0.0)
    t_full = t_full.at[-1].set(jnp.maximum(thickness_trailing, eps))

    _ = chord
    return 0.5 * t_full


_THICKNESS_TYPES = ("NACA", "Denton")


def thickness_type_id(thickness_model: str) -> int:
    tm = str(thickness_model).strip().lower()
    if tm == "naca":
        return 0
    if tm == "denton":
        return 1
    if tm in ("b_spline", "bspline", "quartic_bspline"):
        raise ValueError(
            "B_spline thickness model is routed through the T-Blade3 path only "
            "and is not available in the legacy thickness dispatcher."
        )
    try:
        return _THICKNESS_TYPES.index(thickness_model)
    except ValueError as e:
        raise ValueError(f"Unsupported thickness_model: {thickness_model}") from e


@jax.jit
def _compute_thickness_distribution_by_id(
    thickness_model_id: jnp.ndarray,
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    thickness_leading=0.0,
    thickness_shape_exponent=2.0,
):
    te = jnp.maximum(thickness_trailing, 1e-12)
    # Denton body uses an LE thickness anchor (not applied at x=0 directly).
    le_raw = jnp.where(thickness_leading > 0.0, thickness_leading, 2.0 * radius_leading)
    le_max = jnp.maximum(0.95 * thickness_max, te + 1e-12)
    le = jnp.minimum(jnp.maximum(le_raw, te), le_max)

    def _naca(_):
        return compute_thickness_distribution_NACA_modified(
            x_norm,
            chord,
            loc_max,
            thickness_max,
            thickness_trailing,
            wedge_trailing,
            radius_leading,
        )

    def _denton(_):
        return compute_thickness_distribution_Denton(
            x_norm,
            chord,
            loc_max,
            thickness_max,
            thickness_trailing,
            le,
            thickness_shape_exponent,
            radius_leading,
        )

    idx = jnp.clip(thickness_model_id, 0, len(_THICKNESS_TYPES) - 1)
    return lax.switch(idx, (_naca, _denton), operand=None)


# --- Throat opening ---------------------------------------------------------

@jax.jit
def compute_throat_opening(pitch_angle_rad, metal_angle_out_rad, pitch_out):
# def compute_throat_opening(theta, metal_angle_out_rad, pitch_out):
    # d_theta = theta[-1] - theta[0]
    return pitch_out * jnp.cos(metal_angle_out_rad - 0.5 * pitch_angle_rad)


# --- camberline primitives --------------------------------------------------


@jax.jit
def compute_camberline_straight_polar(r1, r2, phi, theta0, u):
    L = jnp.sqrt((r2 / r1) ** 2 - jnp.sin(phi) ** 2) - jnp.cos(phi)
    x = r1 * jnp.cos(theta0) + u * L * jnp.cos(phi + theta0)
    y = r1 * jnp.sin(theta0) + u * L * jnp.sin(phi + theta0)
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    metal_angle = jnp.arctan(jnp.sin(phi) / jnp.sqrt((r / r1) ** 2 - jnp.sin(phi) ** 2))
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    phi_out = phi + theta0
    return x, y, r, theta, metal_angle, phi_out, stagger


def _bisect_jax(fun, a, b, iters=64):
    def body(_, state):
        a, b = state
        m = 0.5 * (a + b)
        fa = fun(a)
        fm = fun(m)
        left = (fa * fm) <= 0.0
        a = jnp.where(left, a, m)
        b = jnp.where(left, m, b)
        return (a, b)

    a, b = lax.fori_loop(0, iters, body, (a, b))
    return 0.5 * (a + b)


@jax.jit
def compute_camberline_circular_arc_polar(
    r1, r2, metal_angle1, metal_angle2, theta1, u
):
    smax = jnp.arcsin(jnp.minimum(1.0, r2 / r1)) - 1e-6
    stag0 = metal_angle1 + theta1
    a = stag0 - smax
    b = stag0 + smax

    def exit_metal_angle_error(stagger):
        rad = (r2 / r1) ** 2 - jnp.sin(stagger) ** 2
        rad = jnp.maximum(rad, 0.0)
        c = r1 * (jnp.sqrt(rad) - jnp.cos(stagger))

        x1 = r1 * jnp.cos(theta1)
        y1 = r1 * jnp.sin(theta1)
        x2 = x1 + c * jnp.cos(stagger + theta1)
        y2 = y1 + c * jnp.sin(stagger + theta1)
        _ = x2
        _ = y2

        angle_1 = jnp.pi / 2.0 - metal_angle1 - theta1
        angle_2 = 2.0 * (jnp.pi / 2.0 - stagger) - 2.0 * theta1 - angle_1

        cosarg = (r1**2 + r2**2 - c**2) / (2.0 * r1 * r2)
        cosarg = jnp.clip(cosarg, -1.0, 1.0)
        theta_2 = theta1 + jnp.arccos(cosarg)

        trial_metal_angle_2 = jnp.pi / 2.0 - angle_2 - theta_2
        return metal_angle2 - trial_metal_angle_2

    stagger = _bisect_jax(exit_metal_angle_error, a, b, iters=64)

    rad = (r2 / r1) ** 2 - jnp.sin(stagger) ** 2
    rad = jnp.maximum(rad, 0.0)
    c = r1 * (jnp.sqrt(rad) - jnp.cos(stagger))

    x1 = r1 * jnp.cos(theta1)
    y1 = r1 * jnp.sin(theta1)
    x2 = x1 + c * jnp.cos(stagger + theta1)
    y2 = y1 + c * jnp.sin(stagger + theta1)

    angle_1 = jnp.pi / 2.0 - metal_angle1 - theta1
    angle_2 = 2.0 * (jnp.pi / 2.0 - stagger) - 2.0 * theta1 - angle_1
    angle = angle_1 + u * (angle_2 - angle_1)

    x = x1 + (x2 - x1) * (jnp.cos(angle) - jnp.cos(angle_1)) / (
        jnp.cos(angle_2) - jnp.cos(angle_1)
    )
    y = y1 - (x2 - x1) * (jnp.sin(angle) - jnp.sin(angle_1)) / (
        jnp.cos(angle_2) - jnp.cos(angle_1)
    )

    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    metal_angle = jnp.pi / 2.0 - theta - angle
    phi = jnp.pi / 2.0 - angle

    stagger = jnp.where(r1 > r2, stagger + jnp.pi, stagger)
    return x, y, r, theta, metal_angle, phi, stagger


@jax.jit
def compute_camberline_linear_angle_change_polar(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    r = r1 + u * (r2 - r1)
    n = max(2, int(r.shape[0]))
    rs = jnp.linspace(r1, r2, n)
    metal_angle_rs = ((r2 - rs) / (r2 - r1)) * metal_angle1 + (
        (rs - r1) / (r2 - r1)
    ) * metal_angle2
    dtheta_dr = jnp.tan(metal_angle_rs) / rs
    dr = rs[1:] - rs[:-1]
    avg = 0.5 * (dtheta_dr[1:] + dtheta_dr[:-1])
    integ = jnp.cumsum(avg * dr)
    theta_rs = jnp.concatenate([jnp.array([theta0]), theta0 + integ])
    theta = jnp.interp(r, rs, theta_rs)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    metal_angle = ((r2 - r) / (r2 - r1)) * metal_angle1 + (
        (r - r1) / (r2 - r1)
    ) * metal_angle2
    phi = metal_angle + theta
    return x, y, r, theta, metal_angle, phi, stagger


@jax.jit
def compute_camberline_linear_slope_change_polar(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    r = r1 + u * (r2 - r1)
    theta = (
        theta0
        + (r2 * jnp.tan(metal_angle1) - r1 * jnp.tan(metal_angle2))
        * jnp.log(r / r1)
        / (r2 - r1)
        - (jnp.tan(metal_angle1) - jnp.tan(metal_angle2)) * (r - r1) / (r2 - r1)
    )
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    tan_metal_angle = ((r2 - r) / (r2 - r1)) * jnp.tan(metal_angle1) + (
        (r - r1) / (r2 - r1)
    ) * jnp.tan(metal_angle2)
    metal_angle = jnp.arctan(tan_metal_angle)
    phi = metal_angle + theta
    return x, y, r, theta, metal_angle, phi, stagger


# --- curvature-based camberline helpers -------------------------------------


def default_curvature_control_points(total_camber_rad, n_control: int = 8):
    x = jnp.linspace(0.0, 1.0, n_control, dtype=jnp.float64)
    peak_location = 0.35
    w1 = 0.17
    w2 = 0.13
    g1 = jnp.exp(-((x - peak_location) ** 2) / (2.0 * w1 * w1))
    g2 = 0.45 * jnp.exp(-((x - 0.85) ** 2) / (2.0 * w2 * w2))
    shape = g1 + g2
    shape = shape / jnp.maximum(jnp.max(jnp.abs(shape)), 1e-12)
    sign = jnp.where(total_camber_rad >= 0.0, 1.0, -1.0)
    cp = sign * shape
    cp = cp.at[0].set(cp[1])
    cp = cp.at[-1].set(cp[-2])
    return cp


@jax.jit
def _default_curvature_control_points_fixed(total_camber_rad):
    return default_curvature_control_points(total_camber_rad, n_control=8)


@jax.jit
def _curvature_angle_residual(k, a1, b1, total_camber):
    s_in = k * (0.0 - b1)
    s_out = k * (a1 - b1)
    return jnp.arctan(s_out) - jnp.arctan(s_in) - total_camber


@jax.jit
def _curvature_angle_residual_derivative(k, a1, b1):
    s_in = k * (0.0 - b1)
    s_out = k * (a1 - b1)
    return ((a1 - b1) / (1.0 + s_out * s_out)) - ((-b1) / (1.0 + s_in * s_in))


@jax.jit
def _solve_curvature_scaling_factor(a1, b1, total_camber):
    eps = 1.0e-12
    tan_tc = jnp.tan(total_camber)
    p = (a1 * b1) - (b1 * b1)
    det = (a1 * a1) + (4.0 * p * (tan_tc * tan_tc))
    sq = jnp.sqrt(jnp.maximum(det, 0.0))
    den = 2.0 * p * tan_tc

    k1 = (-a1 + sq) / (den + eps)
    k2 = (-a1 - sq) / (den + eps)

    r1 = jnp.abs(_curvature_angle_residual(k1, a1, b1, total_camber))
    r2 = jnp.abs(_curvature_angle_residual(k2, a1, b1, total_camber))
    k_quad = jnp.where(r1 <= r2, k1, k2)

    k_lin = total_camber / (a1 + eps)
    bad_quad = (jnp.abs(den) < 1.0e-10) | (det < 0.0) | (~jnp.isfinite(k_quad))
    k0 = jnp.where(jnp.abs(total_camber) < 1.0e-14, 0.0, jnp.where(bad_quad, k_lin, k_quad))

    def newton_body(_, k):
        f = _curvature_angle_residual(k, a1, b1, total_camber)
        df = _curvature_angle_residual_derivative(k, a1, b1)
        step = f / (df + eps)
        k_new = k - step
        return jnp.where(jnp.isfinite(k_new), k_new, k)

    return lax.fori_loop(0, 10, newton_body, k0)


def _evaluate_curvature_profile(u, curvature_cp):
    n_control = int(curvature_cp.shape[0])
    knots = open_uniform_knot_vector(n_control, degree=3, dtype=jnp.float64)
    B = basis_matrix(u, degree=3, knots=knots, n_control=n_control)
    return B @ curvature_cp


def _curvature_core(u, metal_angle_in, metal_angle_out, curvature_cp):
    base_curv = _evaluate_curvature_profile(u, curvature_cp)
    int_curv = _cumtrapz(base_curv, u)
    int_slope = _cumtrapz(int_curv, u)

    a1 = int_curv[-1]
    b1 = int_slope[-1]
    total_camber = metal_angle_out - metal_angle_in
    k = _solve_curvature_scaling_factor(a1, b1, total_camber)

    slope_local = k * (int_curv - b1)
    camber_local = k * (int_slope - (u * b1))
    stagger = metal_angle_in - jnp.arctan(slope_local[0])
    metal_angle = jnp.arctan(slope_local) + stagger
    return camber_local, slope_local, metal_angle, stagger


def compute_camberline_curvature_based_cart(
    x1, y1, metal_angle1, metal_angle2, c_ax, u, curvature_cp=None
):
    if curvature_cp is None:
        curvature_cp = default_curvature_control_points(metal_angle2 - metal_angle1, n_control=8)
    curvature_cp = jnp.asarray(curvature_cp, dtype=jnp.float64)

    camber_uv, slope_uv, metal_angle, stagger = _curvature_core(
        u, metal_angle1, metal_angle2, curvature_cp
    )
    chord = c_ax / jnp.maximum(jnp.cos(stagger), 1e-12)

    cts = jnp.cos(stagger)
    sts = jnp.sin(stagger)
    x = x1 + chord * (u * cts - camber_uv * sts)
    y = y1 + chord * (u * sts + camber_uv * cts)
    dydx = jnp.tan(metal_angle)
    _ = slope_uv
    return x, y, dydx, stagger, chord


def compute_camberline_curvature_based_polar(
    r1, r2, metal_angle1, metal_angle2, theta0, u, curvature_cp=None
):
    if curvature_cp is None:
        curvature_cp = default_curvature_control_points(metal_angle2 - metal_angle1, n_control=8)
    curvature_cp = jnp.asarray(curvature_cp, dtype=jnp.float64)

    camber_uv, slope_uv, metal_angle, stagger_uv = _curvature_core(
        u, metal_angle1, metal_angle2, curvature_cp
    )
    _ = camber_uv
    _ = slope_uv
    _ = stagger_uv

    r = r1 + u * (r2 - r1)
    drdu = r2 - r1
    dtheta_du = jnp.tan(metal_angle) * drdu / jnp.maximum(r, 1e-12)
    theta = theta0 + _cumtrapz(dtheta_du, u)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    phi = metal_angle + theta
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    chord = _chord_from_theta(r1, r2, theta[0], theta[-1])
    return x, y, r, theta, metal_angle, phi, stagger, chord


# ----------------------------------------------------------------------------
# Camberline dispatch (avoid string arguments in jitted code)
# ----------------------------------------------------------------------------

_CAMBERLINE_RADIAL_TYPES = (
    "straight",
    "circular_arc",
    "linear_angle_change",
    "linear_slope_change",
    "circular_arc_conformal",
    "linear_angle_change_conformal",
    "linear_slope_change_conformal",
    "curvature_based",
)

_CAMBERLINE_CARTESIAN_TYPES = (
    "NACA",
    "circular_arc",
    "linear_angle_change",
    "linear_slope_change",
    "curvature_based",
)


def camberline_radial_type_id(camberline_type: str) -> int:
    try:
        return _CAMBERLINE_RADIAL_TYPES.index(camberline_type)
    except ValueError as e:
        raise ValueError(f"Unsupported camberline_type: {camberline_type}") from e


def camberline_cartesian_type_id(camberline_type: str) -> int:
    try:
        return _CAMBERLINE_CARTESIAN_TYPES.index(camberline_type)
    except ValueError as e:
        raise ValueError(
            f"Unsupported camberline_type for cartesian camberline: {camberline_type}"
        ) from e


@jax.jit
def _compute_camberline_cartesian_by_id(
    camberline_type_id: jnp.ndarray, x1, y1, metal_angle1, metal_angle2, c_ax, u
):
    def naca(args):
        x1, y1, a1, a2, c_ax, u = args
        x, y, stagger, dydx = _compute_camberline_NACA(x1, y1, a1, a2, c_ax, u)
        chord = c_ax / jnp.maximum(jnp.cos(stagger), 1e-12)
        return x, y, dydx, stagger, chord

    def circ(args):
        x1, y1, a1, a2, c_ax, u = args
        x, y, stagger, dydx = _compute_camberline_circular_arc_cart(x1, y1, a1, a2, c_ax, u)
        chord = c_ax / jnp.maximum(jnp.cos(stagger), 1e-12)
        return x, y, dydx, stagger, chord

    def lin_ang(args):
        x1, y1, a1, a2, c_ax, u = args
        x, y, stagger, dydx = _compute_camberline_linear_angle_change_cart(x1, y1, a1, a2, c_ax, u)
        chord = c_ax / jnp.maximum(jnp.cos(stagger), 1e-12)
        return x, y, dydx, stagger, chord

    def lin_slp(args):
        x1, y1, a1, a2, c_ax, u = args
        x, y, stagger, dydx = _compute_camberline_linear_slope_change_cart(x1, y1, a1, a2, c_ax, u)
        chord = c_ax / jnp.maximum(jnp.cos(stagger), 1e-12)
        return x, y, dydx, stagger, chord

    def curvature(args):
        x1, y1, a1, a2, c_ax, u = args
        cp = _default_curvature_control_points_fixed(a2 - a1)
        return compute_camberline_curvature_based_cart(x1, y1, a1, a2, c_ax, u, curvature_cp=cp)

    branches = (naca, circ, lin_ang, lin_slp, curvature)
    return lax.switch(camberline_type_id, branches, (x1, y1, metal_angle1, metal_angle2, c_ax, u))


def compute_camberline_cartesian_by_id(
    camberline_type_id: int | jnp.ndarray, x1, y1, metal_angle1, metal_angle2, c_ax, u
):
    camberline_type_id = jnp.asarray(camberline_type_id, dtype=jnp.int32)
    return _compute_camberline_cartesian_by_id(
        camberline_type_id, x1, y1, metal_angle1, metal_angle2, c_ax, u
    )


@jax.jit
def _create_camberline_conformal_by_id(
    camberline_type_id: jnp.ndarray, r1, r2, metal_angle1, metal_angle2, theta0, u
):
    x1 = 0.0
    y1 = 0.0
    c_ax = 1.0
    x_lin, y_lin, dydx_lin, stagger, _ = _compute_camberline_cartesian_by_id(
        camberline_type_id, x1, y1, metal_angle1, metal_angle2, c_ax, u
    )
    x_rad, y_rad = apply_conformal_mapping(x_lin, y_lin, x1, y1, r1, r2, c_ax, theta0)
    r = jnp.sqrt(x_rad**2 + y_rad**2)
    theta = jnp.arctan2(y_rad, x_rad)
    metal_angle = jnp.arctan(dydx_lin)
    phi = metal_angle + theta
    d_theta = theta[-1] - theta[0]
    stagger_pol = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    _ = stagger
    return x_rad, y_rad, r, theta, metal_angle, phi, stagger_pol


@jax.jit
def compute_camberline_radial_by_id(
    camberline_type_id: jnp.ndarray, r1, r2, metal_angle1, metal_angle2, theta0, u
):
    def straight(args):
        r1, r2, a1, a2, theta0, u = args
        x, y, r, theta, metal_angle, phi_out, stagger = compute_camberline_straight_polar(
            r1, r2, a1, theta0, u
        )
        _ = a2
        phi = jnp.full_like(u, phi_out)
        return x, y, r, theta, metal_angle, phi, stagger

    def circ_arc(args):
        r1, r2, a1, a2, theta0, u = args
        return compute_camberline_circular_arc_polar(r1, r2, a1, a2, theta0, u)

    def lin_ang(args):
        r1, r2, a1, a2, theta0, u = args
        return compute_camberline_linear_angle_change_polar(r1, r2, a1, a2, theta0, u)

    def lin_slp(args):
        r1, r2, a1, a2, theta0, u = args
        return compute_camberline_linear_slope_change_polar(r1, r2, a1, a2, theta0, u)

    def circ_conf(args):
        r1, r2, a1, a2, theta0, u = args
        return _create_camberline_conformal_by_id(
            jnp.array(1, dtype=jnp.int32), r1, r2, a1, a2, theta0, u
        )

    def lin_ang_conf(args):
        r1, r2, a1, a2, theta0, u = args
        return _create_camberline_conformal_by_id(
            jnp.array(2, dtype=jnp.int32), r1, r2, a1, a2, theta0, u
        )

    def lin_slp_conf(args):
        r1, r2, a1, a2, theta0, u = args
        return _create_camberline_conformal_by_id(
            jnp.array(3, dtype=jnp.int32), r1, r2, a1, a2, theta0, u
        )

    def curv(args):
        r1, r2, a1, a2, theta0, u = args
        cp = _default_curvature_control_points_fixed(a2 - a1)
        x, y, r, theta, metal_angle, phi, stagger, _ = compute_camberline_curvature_based_polar(
            r1, r2, a1, a2, theta0, u, curvature_cp=cp
        )
        return x, y, r, theta, metal_angle, phi, stagger

    branches = (
        straight,
        circ_arc,
        lin_ang,
        lin_slp,
        circ_conf,
        lin_ang_conf,
        lin_slp_conf,
        curv,
    )
    x, y, r, theta, metal_angle, phi, stagger = lax.switch(
        camberline_type_id,
        branches,
        (r1, r2, metal_angle1, metal_angle2, theta0, u),
    )

    chord = _chord_from_theta(r1, r2, theta[0], theta[-1])
    return x, y, r, theta, metal_angle, phi, stagger, chord


def compute_camberline_radial(
    camberline_type, r1, r2, metal_angle1, metal_angle2, theta0, u, curvature_cp=None
):
    if camberline_type == "straight":
        x, y, r, theta, metal_angle, phi_out, stagger = compute_camberline_straight_polar(
            r1, r2, metal_angle1, theta0, u
        )
        chord = _chord_from_theta(r1, r2, theta[0], theta[-1])
        return x, y, r, theta, metal_angle, phi_out, stagger, chord

    if camberline_type == "curvature_based":
        return compute_camberline_curvature_based_polar(
            r1, r2, metal_angle1, metal_angle2, theta0, u, curvature_cp=curvature_cp
        )

    camberline_type_id = jnp.array(
        camberline_radial_type_id(camberline_type), dtype=jnp.int32
    )
    return compute_camberline_radial_by_id(
        camberline_type_id, r1, r2, metal_angle1, metal_angle2, theta0, u
    )


# --- Blade coordinates (camber + thickness + TE arc) -----------------------


def _is_bspline_thickness_model(thickness_model: str) -> bool:
    tm = str(thickness_model).strip().lower()
    return tm in (
        "b_spline",
        "bspline",
        "quartic_bspline",
    )


def _use_tblade3_strategy(camberline_type: str, thickness_model: str) -> bool:
    _ = camberline_type
    return _is_bspline_thickness_model(thickness_model)


def _resolve_leading_edge_thickness_for_tblade3(
    *,
    thickness_leading,
    radius_leading,
    thickness_max,
    thickness_trailing,
):
    """Resolve LE thickness like T-Blade-style explicit lethk, with safe fallback."""
    if thickness_leading is None or float(thickness_leading) <= 0.0:
        le = 2.0 * float(radius_leading)
    else:
        le = float(thickness_leading)

    le_min = max(1e-12, 1.5 * float(thickness_trailing))
    le_max = 0.95 * float(thickness_max)
    if le_max <= le_min:
        return float(le_min)
    return float(jnp.clip(le, le_min, le_max))


def compute_blade_coordinates_radial_segments_tblade3(
    camberline_type,
    r1,
    r2,
    metal_angle1,
    metal_angle2,
    theta0,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points: int,
    thickness_model: str = "B_spline",
    thickness_leading=None,
    thickness_shape_exponent: float = 2.0,
):
    """Radial segments using the same thickness + section assembly strategy as Blade2D core.

    This path mirrors:
      curvature camber -> quartic B-spline thickness -> section_builder closure.
    """
    _ = (theta0, thickness_shape_exponent)  # Local section coordinates; Denton exponent unused here.
    le_thickness = _resolve_leading_edge_thickness_for_tblade3(
        thickness_leading=thickness_leading,
        radius_leading=radius_leading,
        thickness_max=thickness_max,
        thickness_trailing=thickness_trailing,
    )

    n_surface = max(81, int((N_points + 2) // 3))
    geometry = {
        "cascade_type": "rotor" if float(metal_angle2) < 0.0 else "stator",
        "camberline_type": str(camberline_type),
        "thickness_model": str(thickness_model),
        "radius_mean_in": float(r1),
        "radius_mean_out": float(r2),
        # Keep chord resolvable for radial case, consistent with your YAML practice.
        "meridional_chord": float(abs(r2 - r1)),
        "metal_angle_in": float(jnp.rad2deg(metal_angle1)),
        "metal_angle_out": float(jnp.rad2deg(metal_angle2)),
        "maximum_thickness": float(thickness_max),
        "maximum_thickness_location_fraction": float(loc_max),
        "leading_edge_radius": float(radius_leading),
        "leading_edge_thickness": float(le_thickness),
        "trailing_edge_radius": float(0.5 * thickness_trailing),
        "trailing_edge_wedge_angle": float(jnp.rad2deg(wedge_trailing)),
    }

    sec = section_from_radial_geometry(
        geometry,
        n_points=n_surface,
        thickness_n_control=11,
        enable_le_blend=False,
        thickness_lambda_smooth_d2=2.0e-3,
        thickness_lambda_smooth_d1=5.0e-5,
    )

    x_lower = jnp.asarray(sec["x_bot"], dtype=jnp.float64)
    y_lower = jnp.asarray(sec["y_bot"], dtype=jnp.float64)
    x_upper = jnp.asarray(sec["x_top"], dtype=jnp.float64)
    y_upper = jnp.asarray(sec["y_top"], dtype=jnp.float64)
    if "x_te_arc" in sec and "y_te_arc" in sec:
        x_te = jnp.asarray(sec["x_te_arc"], dtype=jnp.float64)
        y_te = jnp.asarray(sec["y_te_arc"], dtype=jnp.float64)
    else:
        x_te = jnp.asarray([], dtype=jnp.float64)
        y_te = jnp.asarray([], dtype=jnp.float64)

    cam = sec.get("camber_result", {})
    stagger = cam.get("stagger_angle_rad", None)
    if stagger is None:
        stagger = jnp.deg2rad(float(cam.get("stagger_angle_deg", 0.0)))
    chord = float(cam.get("chord", 1.0))

    # section_from_radial_geometry returns local nondimensional section coordinates.
    # Map them to radial-plane coordinates so they are compatible with annulus plotting.
    le_x = 0.5 * (x_lower[0] + x_upper[0])
    le_y = 0.5 * (y_lower[0] + y_upper[0])

    x0 = float(r1) * float(jnp.cos(theta0))
    y0 = float(r1) * float(jnp.sin(theta0))
    ang = float(theta0) + float(stagger)
    c = float(jnp.cos(ang))
    s = float(jnp.sin(ang))

    def _map_local(x_loc, y_loc):
        if x_loc.size == 0:
            return x_loc, y_loc
        xs = (x_loc - le_x) * chord
        ys = (y_loc - le_y) * chord
        xg = x0 + (c * xs) - (s * ys)
        yg = y0 + (s * xs) + (c * ys)
        return xg, yg

    x_lower, y_lower = _map_local(x_lower, y_lower)
    x_upper, y_upper = _map_local(x_upper, y_upper)
    x_te, y_te = _map_local(x_te, y_te)

    return x_lower, y_lower, x_te, y_te, x_upper, y_upper, float(stagger), chord


def compute_blade_coordinates_cartesian_segments_tblade3(
    camberline_type,
    x1,
    y1,
    beta1,
    beta2,
    chord_ax,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points: int,
    thickness_model: str = "B_spline",
    thickness_leading=None,
    thickness_shape_exponent: float = 2.0,
):
    """Cartesian segments using the same Blade2D core strategy as radial T-Blade3 path.

    This path mirrors:
      curvature camber -> quartic B-spline thickness -> section_builder closure.
    """
    _ = thickness_shape_exponent  # Denton exponent unused in T-Blade3 B-spline route.
    le_thickness = _resolve_leading_edge_thickness_for_tblade3(
        thickness_leading=thickness_leading,
        radius_leading=radius_leading,
        thickness_max=thickness_max,
        thickness_trailing=thickness_trailing,
    )

    n_surface = max(81, int((N_points + 2) // 3))
    geometry = {
        "cascade_type": "rotor" if float(beta2) < 0.0 else "stator",
        "camberline_type": str(camberline_type),
        "thickness_model": str(thickness_model),
        "chord_axial": float(chord_ax),
        "metal_angle_in": float(jnp.rad2deg(beta1)),
        "metal_angle_out": float(jnp.rad2deg(beta2)),
        "maximum_thickness": float(thickness_max),
        "maximum_thickness_location_fraction": float(loc_max),
        "leading_edge_radius": float(radius_leading),
        "leading_edge_thickness": float(le_thickness),
        "trailing_edge_radius": float(0.5 * thickness_trailing),
        "trailing_edge_wedge_angle": float(jnp.rad2deg(wedge_trailing)),
    }

    sec = section_from_axial_geometry(
        geometry,
        n_points=n_surface,
        thickness_n_control=11,
        enable_le_blend=False,
        thickness_lambda_smooth_d2=2.0e-3,
        thickness_lambda_smooth_d1=5.0e-5,
    )

    x_lower = jnp.asarray(sec["x_bot"], dtype=jnp.float64)
    y_lower = jnp.asarray(sec["y_bot"], dtype=jnp.float64)
    x_upper = jnp.asarray(sec["x_top"], dtype=jnp.float64)
    y_upper = jnp.asarray(sec["y_top"], dtype=jnp.float64)
    if "x_te_arc" in sec and "y_te_arc" in sec:
        x_te = jnp.asarray(sec["x_te_arc"], dtype=jnp.float64)
        y_te = jnp.asarray(sec["y_te_arc"], dtype=jnp.float64)
    else:
        x_te = jnp.asarray([], dtype=jnp.float64)
        y_te = jnp.asarray([], dtype=jnp.float64)

    cam = sec.get("camber_result", {})
    stagger = cam.get("stagger_angle_rad", None)
    if stagger is None:
        stagger = jnp.deg2rad(float(cam.get("stagger_angle_deg", 0.0)))
    chord = cam.get("chord", None)
    if chord is None:
        chord = float(chord_ax) / max(float(jnp.abs(jnp.cos(stagger))), 1e-12)
    chord = float(chord)

    # section_from_axial_geometry returns local nondimensional section coordinates.
    # Map them to Cartesian coordinates anchored at (x1, y1) and local stagger.
    le_x = 0.5 * (x_lower[0] + x_upper[0])
    le_y = 0.5 * (y_lower[0] + y_upper[0])
    c = float(jnp.cos(stagger))
    s = float(jnp.sin(stagger))

    def _map_local(x_loc, y_loc):
        if x_loc.size == 0:
            return x_loc, y_loc
        xs = (x_loc - le_x) * chord
        ys = (y_loc - le_y) * chord
        xg = float(x1) + (c * xs) - (s * ys)
        yg = float(y1) + (s * xs) + (c * ys)
        return xg, yg

    x_lower, y_lower = _map_local(x_lower, y_lower)
    x_upper, y_upper = _map_local(x_upper, y_upper)
    x_te, y_te = _map_local(x_te, y_te)
    return x_lower, y_lower, x_te, y_te, x_upper, y_upper, float(stagger), chord


def compute_blade_coordinates_radial_segments(
    camberline_type,
    r1,
    r2,
    metal_angle1,
    metal_angle2,
    theta0,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points: int,
    thickness_model: str,
    curvature_cp=None,
    thickness_leading: float = 0.0,
    thickness_shape_exponent: float = 2.0,
):
    """Return radial blade segments explicitly: lower, TE arc, upper.

    Segment orientation:
    - lower: LE -> TE
    - te arc: lower-TE -> upper-TE
    - upper: LE -> TE
    """
    if _use_tblade3_strategy(camberline_type, thickness_model):
        return compute_blade_coordinates_radial_segments_tblade3(
            camberline_type=camberline_type,
            r1=r1,
            r2=r2,
            metal_angle1=metal_angle1,
            metal_angle2=metal_angle2,
            theta0=theta0,
            loc_max=loc_max,
            thickness_max=thickness_max,
            thickness_trailing=thickness_trailing,
            wedge_trailing=wedge_trailing,
            radius_leading=radius_leading,
            N_points=N_points,
            thickness_model=thickness_model,
            thickness_leading=thickness_leading,
            thickness_shape_exponent=thickness_shape_exponent,
        )

    seg = (N_points + 2) // 3
    u = jnp.linspace(0.0, 1.0, seg)
    x_c, y_c, _, theta, _, phi, stagger, chord = compute_camberline_radial(
        camberline_type,
        r1,
        r2,
        metal_angle1,
        metal_angle2,
        theta0,
        u,
        curvature_cp=curvature_cp,
    )
    x_norm = (x_c - r1 * jnp.cos(theta0)) / chord
    y_norm = (y_c - r1 * jnp.sin(theta0)) / chord
    x_norm_rot, _ = rotate_counterclockwise_2D(x_norm, y_norm, -(stagger + theta0))
    x_norm_rot = jnp.abs(x_norm_rot)

    t_id = jnp.asarray(thickness_type_id(thickness_model), dtype=jnp.int32)
    half_t = _compute_thickness_distribution_by_id(
        t_id,
        x_norm_rot,
        chord,
        loc_max,
        thickness_max,
        thickness_trailing,
        wedge_trailing,
        radius_leading,
        thickness_leading,
        thickness_shape_exponent,
    )

    x_lower = x_c + half_t * jnp.sin(phi)
    y_lower = y_c - half_t * jnp.cos(phi)
    x_upper = x_c - half_t * jnp.sin(phi)
    y_upper = y_c + half_t * jnp.cos(phi)

    x2 = r1 * jnp.cos(theta0) + chord * jnp.cos(stagger + theta0)
    y2 = r1 * jnp.sin(theta0) + chord * jnp.sin(stagger + theta0)
    radius_trailing = 0.5 * thickness_trailing / jnp.cos(wedge_trailing / 2.0)
    phi2 = metal_angle2 + theta[-1]
    sin_half = jnp.sin(wedge_trailing / 2.0)
    xc = x2 - jnp.sign(r2 - r1) * radius_trailing * sin_half * jnp.cos(phi2)
    yc = y2 - jnp.sign(r2 - r1) * radius_trailing * sin_half * jnp.sin(phi2)
    angle1 = +(jnp.pi / 2.0 - wedge_trailing / 2.0) + phi2
    angle2 = -(jnp.pi / 2.0 - wedge_trailing / 2.0) + phi2
    seg_tr = N_points // 3
    angle = jnp.linspace(angle1, angle2, seg_tr)
    x_tr = xc + jnp.sign(r2 - r1) * radius_trailing * jnp.cos(angle)
    y_tr = yc + jnp.sign(r2 - r1) * radius_trailing * jnp.sin(angle)

    x_tr = jnp.where(r1 > r2, x_tr[::-1], x_tr)
    y_tr = jnp.where(r1 > r2, y_tr[::-1], y_tr)

    # Match contour orientation used in compute_blade_coordinates_radial.
    x_te = x_tr[::-1]
    y_te = y_tr[::-1]
    return x_lower, y_lower, x_te, y_te, x_upper, y_upper, stagger, chord


def compute_blade_coordinates_radial(
    camberline_type,
    r1,
    r2,
    metal_angle1,
    metal_angle2,
    theta0,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points: int,
    thickness_model: str = "NACA",
    curvature_cp=None,
    thickness_leading: float = 0.0,
    thickness_shape_exponent: float = 2.0,
):
    x_lower, y_lower, x_te, y_te, x_upper, y_upper, stagger, chord = (
        compute_blade_coordinates_radial_segments(
            camberline_type=camberline_type,
            r1=r1,
            r2=r2,
            metal_angle1=metal_angle1,
            metal_angle2=metal_angle2,
            theta0=theta0,
            loc_max=loc_max,
            thickness_max=thickness_max,
            thickness_trailing=thickness_trailing,
            wedge_trailing=wedge_trailing,
            radius_leading=radius_leading,
            N_points=N_points,
            thickness_model=thickness_model,
            curvature_cp=curvature_cp,
            thickness_leading=thickness_leading,
            thickness_shape_exponent=thickness_shape_exponent,
        )
    )

    x = jnp.concatenate([x_lower, x_te, x_upper[::-1]])
    y = jnp.concatenate([y_lower, y_te, y_upper[::-1]])
    return x, y, stagger, chord


def compute_blade_coordinates_cartesian(
    camberline_type,
    x1,
    y1,
    beta1,
    beta2,
    chord_ax,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points: int,
    thickness_model: str = "NACA",
    curvature_cp=None,
    thickness_leading: float = 0.0,
    thickness_shape_exponent: float = 2.0,
):
    if _use_tblade3_strategy(camberline_type, thickness_model):
        x_lower, y_lower, x_te, y_te, x_upper, y_upper, stagger, chord = (
            compute_blade_coordinates_cartesian_segments_tblade3(
                camberline_type=camberline_type,
                x1=x1,
                y1=y1,
                beta1=beta1,
                beta2=beta2,
                chord_ax=chord_ax,
                loc_max=loc_max,
                thickness_max=thickness_max,
                thickness_trailing=thickness_trailing,
                wedge_trailing=wedge_trailing,
                radius_leading=radius_leading,
                N_points=N_points,
                thickness_model=thickness_model,
                thickness_leading=thickness_leading,
                thickness_shape_exponent=thickness_shape_exponent,
            )
        )
        x = jnp.concatenate([x_lower, x_te[::-1], x_upper[::-1]])
        y = jnp.concatenate([y_lower, y_te[::-1], y_upper[::-1]])
        return x, y, stagger, chord

    u = jnp.linspace(0.0, 1.0, N_points)
    x_c, y_c, dydx, stagger, chord = compute_camberline_cartesian(
        camberline_type,
        x1,
        y1,
        beta1,
        beta2,
        chord_ax,
        u,
        curvature_cp=curvature_cp,
    )

    x_norm = (x_c - x1) / chord
    y_norm = (y_c - y1) / chord
    x_rot, _ = rotate_counterclockwise_2D(x_norm, y_norm, -stagger)
    x_norm_rot = jnp.abs(x_rot)

    t_id = jnp.asarray(thickness_type_id(thickness_model), dtype=jnp.int32)
    half_t = _compute_thickness_distribution_by_id(
        t_id,
        x_norm_rot,
        chord,
        loc_max,
        thickness_max,
        thickness_trailing,
        wedge_trailing,
        radius_leading,
        thickness_leading,
        thickness_shape_exponent,
    )

    theta = jnp.arctan(dydx)
    x_lower = x_c + half_t * jnp.sin(theta)
    y_lower = y_c - half_t * jnp.cos(theta)
    x_upper = x_c - half_t * jnp.sin(theta)
    y_upper = y_c + half_t * jnp.cos(theta)

    x2 = x1 + chord_ax
    y2 = y1 + chord_ax * jnp.tan(stagger)

    radius_trailing = 0.5 * thickness_trailing / jnp.cos(wedge_trailing / 2.0)
    x_c_te = x2 - radius_trailing * jnp.sin(wedge_trailing / 2.0) * jnp.cos(beta2)
    y_c_te = y2 - radius_trailing * jnp.sin(wedge_trailing / 2.0) * jnp.sin(beta2)

    phi1 = (jnp.pi / 2.0 - wedge_trailing / 2.0) + beta2
    phi2 = -(jnp.pi / 2.0 - wedge_trailing / 2.0) + beta2
    seg_tr = N_points // 2
    angle = jnp.linspace(phi1, phi2, seg_tr)

    x_tr = x_c_te + radius_trailing * jnp.cos(angle)
    y_tr = y_c_te + radius_trailing * jnp.sin(angle)

    x = jnp.concatenate([x_lower, x_tr[::-1], x_upper[::-1]])
    y = jnp.concatenate([y_lower, y_tr[::-1], y_upper[::-1]])

    return x, y, stagger, chord


# =============================
# Linear (Cartesian) camberlines + conformal map
# =============================


def compute_camberline_cartesian(
    camberline_type: str, x1, y1, metal_angle1, metal_angle2, c_ax, u, curvature_cp=None
):
    if camberline_type == "curvature_based":
        return compute_camberline_curvature_based_cart(
            x1,
            y1,
            metal_angle1,
            metal_angle2,
            c_ax,
            u,
            curvature_cp=curvature_cp,
        )

    types = ("NACA", "circular_arc", "linear_angle_change", "linear_slope_change", "curvature_based")
    try:
        camberline_type_id = jnp.array(types.index(camberline_type), dtype=jnp.int32)
    except ValueError as e:
        raise ValueError("Unsupported camberline_type for cartesian camberline") from e
    return _compute_camberline_cartesian_by_id(
        camberline_type_id, x1, y1, metal_angle1, metal_angle2, c_ax, u
    )


def _compute_camberline_circular_arc_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    x2 = x1 + c_ax
    stagger = (metal_angle1 + metal_angle2) / 2.0
    metal_angle = metal_angle1 + u * (metal_angle2 - metal_angle1)

    def curved(_):
        denom = jnp.sin(metal_angle2) - jnp.sin(metal_angle1)
        x = x1 + (x2 - x1) * (jnp.sin(metal_angle) - jnp.sin(metal_angle1)) / denom
        y = y1 - (x2 - x1) * (jnp.cos(metal_angle) - jnp.cos(metal_angle1)) / denom
        return x, y

    def straight(_):
        x = x1 + u * (x2 - x1)
        y = y1 + (x - x1) * jnp.tan(stagger)
        return x, y

    use_curved = jnp.abs(metal_angle1 - metal_angle2) > 1e-6
    x, y = lax.cond(use_curved, curved, straight, operand=None)
    dydx = jnp.tan(metal_angle)
    return x, y, stagger, dydx


def _compute_camberline_NACA(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    stagger = (metal_angle1 + metal_angle2) / 2.0
    denom = jnp.tan(metal_angle2 - stagger) - jnp.tan(metal_angle1 - stagger)
    p = jnp.tan(metal_angle2 - stagger) / (denom + 1e-12)
    m = p / 2.0 * jnp.tan(metal_angle1 - stagger)
    x_c = u

    left = x_c <= p
    y_left = m / (p**2 + 1e-12) * (2.0 * p * x_c - x_c**2)
    y_right = m / ((1 - p) ** 2 + 1e-12) * (1.0 - 2.0 * p + 2.0 * p * x_c - x_c**2)
    y_c = jnp.where(left, y_left, y_right)

    dy_left = 2.0 * m / (p**2 + 1e-12) * (p - x_c)
    dy_right = 2.0 * m / ((1 - p) ** 2 + 1e-12) * (p - x_c)
    dy_c = jnp.where(left, dy_left, dy_right)

    chord = c_ax / jnp.maximum(jnp.cos(stagger), 1e-12)
    R = jnp.array(
        [[jnp.cos(stagger), -jnp.sin(stagger)], [jnp.sin(stagger), jnp.cos(stagger)]]
    )
    coords = jnp.array([[x1], [y1]]) + chord * R @ jnp.vstack((x_c, y_c))
    x = coords[0, :]
    y = coords[1, :]
    metal_angle = jnp.arctan(dy_c) + stagger
    dydx = jnp.tan(metal_angle)
    return x, y, stagger, dydx


def _compute_camberline_linear_angle_change_cart(
    x1, y1, metal_angle1, metal_angle2, c_ax, u
):
    x2 = x1 + c_ax

    def varying(_):
        stagger = jnp.arctan(
            -jnp.log(jnp.cos(metal_angle2) / jnp.cos(metal_angle1))
            / (metal_angle2 - metal_angle1 + 1e-6)
        )
        metal_angle = metal_angle1 + u * (metal_angle2 - metal_angle1)
        x = x1 + (metal_angle - metal_angle1) / (metal_angle2 - metal_angle1) * (x2 - x1)
        y = y1 - (x2 - x1) / (metal_angle2 - metal_angle1) * jnp.log(
            jnp.cos(metal_angle) / jnp.cos(metal_angle1)
        )
        return x, y, stagger

    def constant(_):
        stagger = metal_angle1
        x = x1 + u * (x2 - x1)
        y = y1 + (x - x1) * jnp.tan(stagger)
        return x, y, stagger

    use_varying = jnp.abs(metal_angle1 - metal_angle2) > 1e-6
    x, y, stagger = lax.cond(use_varying, varying, constant, operand=None)
    metal_angle_x = metal_angle1 + (metal_angle2 - metal_angle1) * (x - x1) / (x2 - x1)
    dydx = jnp.tan(metal_angle_x)
    return x, y, stagger, dydx


def _compute_camberline_linear_slope_change_cart(
    x1, y1, metal_angle1, metal_angle2, c_ax, u
):
    x2 = x1 + c_ax
    x = x1 + u * (x2 - x1)
    temp = (
        0.5 * jnp.tan(metal_angle1) * (1.0 - ((x2 - x) / (x2 - x1)) ** 2)
        + 0.5 * jnp.tan(metal_angle2) * ((x - x1) / (x2 - x1)) ** 2
    )
    y = y1 + temp * (x2 - x1)
    stagger = jnp.arctan(0.5 * (jnp.tan(metal_angle1) + jnp.tan(metal_angle2)))
    dydx = jnp.tan(metal_angle1) * (x2 - x) / (x2 - x1) + jnp.tan(metal_angle2) * (
        x - x1
    ) / (x2 - x1)
    return x, y, stagger, dydx


# ---- Conformal mapping (linear -> radial) ----------------------------------


def apply_conformal_mapping(x, y, x1, y1, r1, r2, c_ax, theta0):
    r = r1 * jnp.exp(jnp.log(r2 / r1) * (x - x1) / c_ax)
    theta = theta0 + jnp.log(r2 / r1) / c_ax * (y - y1)
    X = r * jnp.cos(theta)
    Y = r * jnp.sin(theta)
    return X, Y


def create_camberline_conformal(
    camberline_type: str, r1, r2, metal_angle1, metal_angle2, theta0, u
):
    x1 = 0.0
    y1 = 0.0
    c_ax = 1.0
    x_lin, y_lin, dydx_lin, stagger, _ = compute_camberline_cartesian(
        camberline_type, x1, y1, metal_angle1, metal_angle2, c_ax, u
    )
    x_rad, y_rad = apply_conformal_mapping(x_lin, y_lin, x1, y1, r1, r2, c_ax, theta0)
    r = jnp.sqrt(x_rad**2 + y_rad**2)
    theta = jnp.arctan2(y_rad, x_rad)
    metal_angle = jnp.arctan(dydx_lin)
    phi = metal_angle + theta
    d_theta = theta[-1] - theta[0]
    stagger_pol = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    _ = stagger
    return x_rad, y_rad, r, theta, metal_angle, phi, stagger_pol


def create_camberline_circular_arc_conformal(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    return create_camberline_conformal(
        "circular_arc", r1, r2, metal_angle1, metal_angle2, theta0, u
    )


def compute_camberline_linear_angle_change_conformal(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    return create_camberline_conformal(
        "linear_angle_change", r1, r2, metal_angle1, metal_angle2, theta0, u
    )


def compute_camberline_linear_slope_change_conformal(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    return create_camberline_conformal(
        "linear_slope_change", r1, r2, metal_angle1, metal_angle2, theta0, u
    )
