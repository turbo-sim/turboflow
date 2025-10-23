# blade_parametrization_polar_jax.py
# ------------------------------------------------------------
# JAX-only blade parametrization for polar/radial cascades
# (no plotting, no YAML/pipeline code)
# ------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jax import lax


# --- geometry helpers -------------------------------------------------------

def rotate_counterclockwise_2D(x, y, theta):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    X = ct * x - st * y
    Y = st * x + ct * y
    return X, Y


# --- thickness (NACA 4-series modified) ------------------------------------

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
    row = jnp.array([0.5 / jnp.sqrt(loc_max), 1.0, 2.0 * loc_max, 3.0 * loc_max**2, 4.0 * loc_max**3])
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

    return chord * (
        A * jnp.sqrt(x_norm)
        + B * x_norm
        + C * x_norm**2
        + D * x_norm**3
        + E * x_norm**4
    )


# --- camberline primitives --------------------------------------------------

def _chord_from_theta(r1, r2, theta1, thetaN):
    return jnp.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * jnp.cos(thetaN - theta1))

# --- Throat opening ---------------------------------------------------------

def compute_throat_opening(theta, metal_angle_out, pitch_at_exit):
    """
    throat_opening = pitch_at_exit * cos( metal_angle_out + 0.5*(theta[-1] - theta[0]) )
    """
    d_theta = theta[-1] - theta[0]
    return pitch_at_exit * jnp.cos(metal_angle_out + 0.5 * d_theta)


def compute_camberline_straight_polar(r1, r2, phi, theta0, u):
    L = (jnp.sqrt((r2 / r1) ** 2 - jnp.sin(phi) ** 2) - jnp.cos(phi))
    x = r1 * jnp.cos(theta0) + u * L * jnp.cos(phi + theta0)
    y = r1 * jnp.sin(theta0) + u * L * jnp.sin(phi + theta0)
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    metal_angle = jnp.arctan(jnp.sin(phi) / jnp.sqrt((r / r1) ** 2 - jnp.sin(phi) ** 2))
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    phi_out = phi + theta0
    return x, y, r, theta, metal_angle, phi_out, stagger


# --- JAX-native bisection (fixed-iteration) for circular-arc ----------------

def _bisect_jax(fun, a, b, iters=64):
    def body(i, state):
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


def compute_camberline_circular_arc_polar(r1, r2, metal_angle1, metal_angle2, theta1, u):
    smax = float(jnp.arcsin(jnp.minimum(1.0, float(r2 / r1)))) - 1e-6
    stag0 = metal_angle1 + theta1
    a = stag0 - smax
    b = stag0 + smax

    def exit_metal_angle_error(stagger):
        rad = (r2 / r1) ** 2 - jnp.sin(stagger) ** 2
        rad = jnp.maximum(rad, 0.0)
        c = r1 * (jnp.sqrt(rad) - jnp.cos(stagger))

        x1 = r1 * jnp.cos(theta1); y1 = r1 * jnp.sin(theta1)
        x2 = x1 + c * jnp.cos(stagger + theta1)
        y2 = y1 + c * jnp.sin(stagger + theta1)

        angle_1 = jnp.pi/2.0 - metal_angle1 - theta1
        angle_2 = 2.0*(jnp.pi/2.0 - stagger) - 2.0*theta1 - angle_1

        cosarg = (r1**2 + r2**2 - c**2) / (2.0 * r1 * r2)
        cosarg = jnp.clip(cosarg, -1.0, 1.0)
        theta_2 = theta1 + jnp.arccos(cosarg)

        trial_metal_angle_2 = jnp.pi/2.0 - angle_2 - theta_2
        return metal_angle2 - trial_metal_angle_2

    stagger = _bisect_jax(exit_metal_angle_error, a, b, iters=64)

    rad = (r2 / r1) ** 2 - jnp.sin(stagger) ** 2
    rad = jnp.maximum(rad, 0.0)
    c = r1 * (jnp.sqrt(rad) - jnp.cos(stagger))

    x1 = r1 * jnp.cos(theta1); y1 = r1 * jnp.sin(theta1)
    x2 = x1 + c * jnp.cos(stagger + theta1)
    y2 = y1 + c * jnp.sin(stagger + theta1)

    angle_1 = jnp.pi/2.0 - metal_angle1 - theta1
    angle_2 = 2.0*(jnp.pi/2.0 - stagger) - 2.0*theta1 - angle_1
    angle = angle_1 + u * (angle_2 - angle_1)

    x = x1 + (x2 - x1) * (jnp.cos(angle) - jnp.cos(angle_1)) / (jnp.cos(angle_2) - jnp.cos(angle_1))
    y = y1 - (x2 - x1) * (jnp.sin(angle) - jnp.sin(angle_1)) / (jnp.cos(angle_2) - jnp.cos(angle_1))

    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    metal_angle = jnp.pi/2.0 - theta - angle
    phi  = jnp.pi/2.0 - angle

    stagger = jnp.where(r1 > r2, stagger + jnp.pi, stagger)
    return x, y, r, theta, metal_angle, phi, stagger


def compute_camberline_linear_angle_change_polar(r1, r2, metal_angle1, metal_angle2, theta0, u):
    r = r1 + u * (r2 - r1)
    n = int(max(2, r.shape[0]))
    rs = jnp.linspace(r1, r2, n)
    metal_angle_rs = ((r2 - rs) / (r2 - r1)) * metal_angle1 + ((rs - r1) / (r2 - r1)) * metal_angle2
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
    metal_angle = ((r2 - r) / (r2 - r1)) * metal_angle1 + ((r - r1) / (r2 - r1)) * metal_angle2
    phi = metal_angle + theta
    return x, y, r, theta, metal_angle, phi, stagger


def compute_camberline_linear_slope_change_polar(r1, r2, metal_angle1, metal_angle2, theta0, u):
    r = r1 + u * (r2 - r1)
    theta = (
        theta0
        + (r2 * jnp.tan(metal_angle1) - r1 * jnp.tan(metal_angle2)) * jnp.log(r / r1) / (r2 - r1)
        - (jnp.tan(metal_angle1) - jnp.tan(metal_angle2)) * (r - r1) / (r2 - r1)
    )
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    tan_metal_angle = ((r2 - r) / (r2 - r1)) * jnp.tan(metal_angle1) + ((r - r1) / (r2 - r1)) * jnp.tan(metal_angle2)
    metal_angle = jnp.arctan(tan_metal_angle)
    phi = metal_angle + theta
    return x, y, r, theta, metal_angle, phi, stagger


def compute_camberline_radial(camberline_type, r1, r2, metal_angle1, metal_angle2, theta0, u):
    if camberline_type == "straight":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_straight_polar(
            r1, r2, metal_angle1, theta0, u
        )
    elif camberline_type == "circular_arc":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_circular_arc_polar(
            r1, r2, metal_angle1, metal_angle2, theta0, u
        )
    elif camberline_type == "linear_angle_change":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_linear_angle_change_polar(
            r1, r2, metal_angle1, metal_angle2, theta0, u
        )
    elif camberline_type == "linear_slope_change":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_linear_slope_change_polar(
            r1, r2, metal_angle1, metal_angle2, theta0, u
        )
    elif camberline_type == "circular_arc_conformal":
        x, y, r, theta, metal_angle, phi, stagger = create_camberline_circular_arc_conformal(
            r1, r2, metal_angle1, metal_angle2, theta0, u
        )
    elif camberline_type == "linear_angle_change_conformal":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_linear_angle_change_conformal(
            r1, r2, metal_angle1, metal_angle2, theta0, u
        )
    elif camberline_type == "linear_slope_change_conformal":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_linear_slope_change_conformal(
            r1, r2, metal_angle1, metal_angle2, theta0, u
        )
    else:
        raise ValueError(f"Unsupported camberline_type: {camberline_type}")

    chord = _chord_from_theta(r1, r2, theta[0], theta[-1])
    return x, y, r, theta, metal_angle, phi, stagger, chord


# --- Blade coordinates (camber + thickness + TE arc) -----------------------

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
    N_points,
):
    seg = int(jnp.ceil(N_points / 3.0))
    u = jnp.linspace(0.0, 1.0, seg)
    x_c, y_c, _, theta, _, phi, stagger, chord = compute_camberline_radial(
        camberline_type, r1, r2, metal_angle1, metal_angle2, theta0, u
    )
    x_norm = (x_c - r1 * jnp.cos(theta0)) / chord
    y_norm = (y_c - r1 * jnp.sin(theta0)) / chord
    x_norm_rot, _ = rotate_counterclockwise_2D(x_norm, y_norm, -(stagger + theta0))
    x_norm_rot = jnp.abs(x_norm_rot)

    half_t = compute_thickness_distribution_NACA_modified(
        x_norm_rot, chord, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading
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
    seg_tr = int(jnp.floor(N_points / 3.0))
    angle = jnp.linspace(angle1, angle2, seg_tr)
    x_tr = xc + jnp.sign(r2 - r1) * radius_trailing * jnp.cos(angle)
    y_tr = yc + jnp.sign(r2 - r1) * radius_trailing * jnp.sin(angle)

    x_tr = jnp.where(r1 > r2, x_tr[::-1], x_tr)
    y_tr = jnp.where(r1 > r2, y_tr[::-1], y_tr)

    x = jnp.concatenate([x_lower, x_tr[::-1], x_upper[::-1]])
    y = jnp.concatenate([y_lower, y_tr[::-1], y_upper[::-1]])
    return x, y, stagger, chord


# =============================
# Linear (Cartesian) camberlines + conformal map
# =============================

def compute_camberline_cartesian(camberline_type: str, x1, y1, metal_angle1, metal_angle2, c_ax, u):
    if camberline_type == "NACA":
        x, y, stagger, dydx = _compute_camberline_NACA(x1, y1, metal_angle1, metal_angle2, c_ax, u)
    elif camberline_type == "circular_arc":
        x, y, stagger, dydx = _compute_camberline_circular_arc_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u)
    elif camberline_type == "linear_angle_change":
        x, y, stagger, dydx = _compute_camberline_linear_angle_change_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u)
    elif camberline_type == "linear_slope_change":
        x, y, stagger, dydx = _compute_camberline_linear_slope_change_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u)
    else:
        raise ValueError("Unsupported camberline_type for cartesian camberline")
    chord = c_ax / jnp.cos(stagger)
    return x, y, dydx, stagger, chord


def _compute_camberline_circular_arc_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    x2 = x1 + c_ax
    stagger = (metal_angle1 + metal_angle2) / 2.0
    metal_angle = metal_angle1 + u * (metal_angle2 - metal_angle1)
    if float(jnp.abs(metal_angle1 - metal_angle2)) > 1e-6:
        x = x1 + (x2 - x1) * (jnp.sin(metal_angle) - jnp.sin(metal_angle1)) / (jnp.sin(metal_angle2) - jnp.sin(metal_angle1))
        y = y1 - (x2 - x1) * (jnp.cos(metal_angle) - jnp.cos(metal_angle1)) / (jnp.sin(metal_angle2) - jnp.sin(metal_angle1))
    else:
        x = x1 + u * (x2 - x1)
        y = y1 + (x - x1) * jnp.tan(stagger)
    dydx = jnp.tan(metal_angle)
    return x, y, stagger, dydx


def _compute_camberline_NACA(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    stagger = (metal_angle1 + metal_angle2) / 2.0
    denom = jnp.tan(metal_angle2 - stagger) - jnp.tan(metal_angle1 - stagger)
    p = jnp.tan(metal_angle2 - stagger) / (denom + 1e-12)
    m = p / 2.0 * jnp.tan(metal_angle1 - stagger)
    x_c = u

    left = (x_c <= p)
    y_left = m / (p**2 + 1e-12) * (2.0 * p * x_c - x_c**2)
    y_right = m / ((1 - p)**2 + 1e-12) * (1.0 - 2.0 * p + 2.0 * p * x_c - x_c**2)
    y_c = jnp.where(left, y_left, y_right)

    dy_left = 2.0 * m / (p**2 + 1e-12) * (p - x_c)
    dy_right = 2.0 * m / ((1 - p)**2 + 1e-12) * (p - x_c)
    dy_c = jnp.where(left, dy_left, dy_right)

    chord = c_ax / jnp.cos(stagger)
    R = jnp.array([[jnp.cos(stagger), -jnp.sin(stagger)], [jnp.sin(stagger), jnp.cos(stagger)]])
    coords = jnp.array([[x1], [y1]]) + chord * R @ jnp.vstack((x_c, y_c))
    x = coords[0, :]
    y = coords[1, :]
    metal_angle = jnp.arctan(dy_c) + stagger
    dydx = jnp.tan(metal_angle)
    return x, y, stagger, dydx


def _compute_camberline_linear_angle_change_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    x2 = x1 + c_ax
    if float(jnp.abs(metal_angle1 - metal_angle2)) > 1e-6:
        stagger = jnp.arctan(-jnp.log(jnp.cos(metal_angle2) / jnp.cos(metal_angle1)) / (metal_angle2 - metal_angle1 + 1e-6))
        metal_angle = metal_angle1 + u * (metal_angle2 - metal_angle1)
        x = x1 + (metal_angle - metal_angle1) / (metal_angle2 - metal_angle1) * (x2 - x1)
        y = y1 - (x2 - x1) / (metal_angle2 - metal_angle1) * jnp.log(jnp.cos(metal_angle) / jnp.cos(metal_angle1))
    else:
        stagger = metal_angle1
        x = x1 + u * (x2 - x1)
        y = y1 + (x - x1) * jnp.tan(stagger)
    metal_angle_x = metal_angle1 + (metal_angle2 - metal_angle1) * (x - x1) / (x2 - x1)
    dydx = jnp.tan(metal_angle_x)
    return x, y, stagger, dydx


def _compute_camberline_linear_slope_change_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    x2 = x1 + c_ax
    x = x1 + u * (x2 - x1)
    temp = 0.5 * jnp.tan(metal_angle1) * (1.0 - ((x2 - x) / (x2 - x1))**2) + 0.5 * jnp.tan(metal_angle2) * ((x - x1) / (x2 - x1))**2
    y = y1 + temp * (x2 - x1)
    stagger = jnp.arctan(0.5 * (jnp.tan(metal_angle1) + jnp.tan(metal_angle2)))
    dydx = jnp.tan(metal_angle1) * (x2 - x) / (x2 - x1) + jnp.tan(metal_angle2) * (x - x1) / (x2 - x1)
    return x, y, stagger, dydx


# ---- Conformal mapping (linear -> radial) ----------------------------------

def apply_conformal_mapping(x, y, x1, y1, r1, r2, c_ax, theta0):
    r = r1 * jnp.exp(jnp.log(r2 / r1) * (x - x1) / c_ax)
    theta = theta0 + jnp.log(r2 / r1) / c_ax * (y - y1)
    X = r * jnp.cos(theta)
    Y = r * jnp.sin(theta)
    return X, Y


def create_camberline_conformal(camberline_type: str, r1, r2, metal_angle1, metal_angle2, theta0, u):
    x1 = 0.0; y1 = 0.0; c_ax = 1.0
    x_lin, y_lin, dydx_lin, stagger, _ = compute_camberline_cartesian(camberline_type, x1, y1, metal_angle1, metal_angle2, c_ax, u)
    x_rad, y_rad = apply_conformal_mapping(x_lin, y_lin, x1, y1, r1, r2, c_ax, theta0)
    r = jnp.sqrt(x_rad**2 + y_rad**2)
    theta = jnp.arctan2(y_rad, x_rad)
    metal_angle = jnp.arctan(dydx_lin)
    phi = metal_angle + theta
    d_theta = theta[-1] - theta[0]
    stagger_pol = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    return x_rad, y_rad, r, theta, metal_angle, phi, stagger_pol


# Convenience wrappers

def create_camberline_circular_arc_conformal(r1, r2, metal_angle1, metal_angle2, theta0, u):
    return create_camberline_conformal("circular_arc", r1, r2, metal_angle1, metal_angle2, theta0, u)

def compute_camberline_linear_angle_change_conformal(r1, r2, metal_angle1, metal_angle2, theta0, u):
    return create_camberline_conformal("linear_angle_change", r1, r2, metal_angle1, metal_angle2, theta0, u)

def compute_camberline_linear_slope_change_conformal(r1, r2, metal_angle1, metal_angle2, theta0, u):
    return create_camberline_conformal("linear_slope_change", r1, r2, metal_angle1, metal_angle2, theta0, u)
