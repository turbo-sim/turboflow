"""Quartic B-spline thickness distribution using the T-Blade3 paper strategy.

This module implements the "implicit LE via quartic thickness spline constraints"
approach described in the paper and used in T-Blade3 (`splinethick.f90`).

Key points of the strategy:
- Solve for quartic B-spline control points of top and bottom thickness curves.
- Enforce LE/TE/max-thickness constraints through a square linear system.
- Evaluate top/bottom thickness as parametric quartic B-splines y=f(x), then
  compute half-thickness as 0.5 * (y_top - y_bot).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp

from .bspline import open_uniform_knot_vector
from .thickness_naca import naca_modified_thickness


def _quartic_basis(t: float) -> np.ndarray:
    """Quartic segment basis functions used by T-Blade3."""
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    return np.asarray(
        [
            ((1.0 - t) ** 4) / 24.0,
            ((-4.0 * t4) + (12.0 * t3) - (6.0 * t2) - (12.0 * t) + 11.0) / 24.0,
            ((6.0 * t4) - (12.0 * t3) - (6.0 * t2) + (12.0 * t) + 11.0) / 24.0,
            ((-4.0 * t4) + (4.0 * t3) + (6.0 * t2) + (4.0 * t) + 1.0) / 24.0,
            t4 / 24.0,
        ],
        dtype=float,
    )


def _quartic_basis_d1(t: float) -> np.ndarray:
    """First derivative of quartic segment basis functions."""
    t2 = t * t
    t3 = t2 * t
    return np.asarray(
        [
            (t3 / 6.0) - (t2 / 2.0) + (t / 2.0) - (1.0 / 6.0),
            (-2.0 * t3 / 3.0) + (3.0 * t2 / 2.0) - (t / 2.0) - 0.5,
            t3 - (3.0 * t2 / 2.0) - (t / 2.0) + 0.5,
            (-2.0 * t3 / 3.0) + (t2 / 2.0) + (t / 2.0) + (1.0 / 6.0),
            t3 / 6.0,
        ],
        dtype=float,
    )


def _bspline4_eval(cp5: np.ndarray, t: float) -> float:
    return float(np.dot(cp5, _quartic_basis(t)))


def _bspline4_eval_d1(cp5: np.ndarray, t: float) -> float:
    return float(np.dot(cp5, _quartic_basis_d1(t)))


def _bspline4_t_newton(cp5: np.ndarray, x_target: float, max_iter: int = 80, tol: float = 1e-12) -> float:
    """Newton solve for local parameter t in [0, 1] where bspline4(cp5,t)=x_target."""
    t = 0.5
    for _ in range(max_iter):
        x_t = _bspline4_eval(cp5, t)
        dxdt = _bspline4_eval_d1(cp5, t)
        if abs(dxdt) < 1e-14:
            break
        t_new = t + ((x_target - x_t) / dxdt)
        if t_new < 0.0:
            t_new = 0.0
        elif t_new > 1.0:
            t_new = 1.0
        if abs(t_new - t) <= tol:
            t = t_new
            break
        t = t_new
    return float(t)


def _bspline_y_of_x(
    ncp: int,
    degree: int,
    xcp: np.ndarray,
    ycp: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Compute y=f(x) for a segment-wise quartic B-spline, mirroring T-Blade3."""
    if degree != 4:
        raise ValueError("Only degree=4 is supported in the T-Blade3 quartic thickness solver.")
    if xcp.shape[0] != ncp or ycp.shape[0] != ncp:
        raise ValueError("xcp/ycp size mismatch.")

    nseg = ncp - degree
    if nseg < 1:
        raise ValueError("Invalid control-point count for quartic spline.")

    x_spl_end = np.zeros(nseg + 1, dtype=float)
    x_spl_end[0] = _bspline4_eval(xcp[0 : degree + 1], 0.0)
    for j in range(nseg):
        x_spl_end[j + 1] = _bspline4_eval(xcp[j : j + degree + 1], 1.0)

    if x_spl_end[1] > x_spl_end[0]:
        seg_1 = 0
        seg_end = nseg - 1
    else:
        seg_1 = nseg - 1
        seg_end = 0

    y = np.zeros_like(x, dtype=float)
    tol = 1e-14

    for i, xi in enumerate(x):
        assigned = False
        for j in range(nseg):
            x0 = x_spl_end[j]
            x1 = x_spl_end[j + 1]
            in_range = ((xi >= x0) and (xi <= x1)) or ((xi <= x0) and (xi >= x1))
            boundary_ok = ((i == 0) and (j == seg_1)) or ((i == x.shape[0] - 1) and (j == seg_end))
            if not (in_range or boundary_ok):
                continue

            cp_seg_x = xcp[j : j + degree + 1]
            cp_seg_y = ycp[j : j + degree + 1]

            if abs(xi - x0) <= tol:
                t = 0.0
            elif abs(xi - x1) <= tol:
                t = 1.0
            else:
                t = _bspline4_t_newton(cp_seg_x, xi)

            y[i] = _bspline4_eval(cp_seg_y, t)
            assigned = True
            break

        if not assigned:
            # Fallback for tiny extrapolation due to floating-point mismatch.
            if xi <= min(x_spl_end[0], x_spl_end[-1]):
                cp_seg_x = xcp[0 : degree + 1]
                cp_seg_y = ycp[0 : degree + 1]
                t = 0.0
            else:
                cp_seg_x = xcp[-(degree + 1) :]
                cp_seg_y = ycp[-(degree + 1) :]
                t = 1.0
            _ = cp_seg_x
            y[i] = _bspline4_eval(cp_seg_y, t)

    return y


def _hermite_cubic_scalar(
    s: np.ndarray,
    *,
    y0: float,
    m0: float,
    y1: float,
    m1: float,
) -> np.ndarray:
    """Cubic Hermite interpolation on s in [0,1]."""
    h00 = (2.0 * s**3) - (3.0 * s**2) + 1.0
    h10 = s**3 - (2.0 * s**2) + s
    h01 = (-2.0 * s**3) + (3.0 * s**2)
    h11 = s**3 - s**2
    return (h00 * y0) + (h10 * m0) + (h01 * y1) + (h11 * m1)


def _apply_te_monotone_cap(
    u: np.ndarray,
    thickness_half: np.ndarray,
    te_half: float,
) -> np.ndarray:
    """Enforce smooth monotone approach to blunt TE thickness.

    The paper/T-Blade3 strategy enforces TE constraints. In discrete sampling,
    small undershoots can still appear before u=1. This cap removes that artifact
    using a monotone cubic-Hermite tail replacement.
    """
    n = u.shape[0]
    if n < 16:
        out = thickness_half.copy()
        out[-1] = te_half
        return out

    out = thickness_half.copy()
    out[-1] = te_half

    tail_start = max(0, int(0.75 * (n - 1)))
    idx_tail = np.arange(tail_start, n - 1)
    if idx_tail.size == 0:
        return out

    undershoot = idx_tail[out[idx_tail] < (te_half - 1e-12)]
    if undershoot.size == 0:
        return out

    i_cross = int(undershoot[0])
    i_join = max(0, i_cross - max(6, int(0.05 * n)))
    while i_join > 0 and out[i_join] <= te_half:
        i_join -= 1

    u0 = float(u[i_join])
    u1 = float(u[-1])
    if u1 <= u0 + 1e-12:
        out[-1] = te_half
        return out

    y0 = float(max(out[i_join], te_half))
    y1 = float(te_half)
    # Secant slope in physical u-units; should be <= 0.
    delta = (y1 - y0) / (u1 - u0)
    d = np.gradient(out, u)
    d0_raw = float(d[i_join])
    # Monotone cubic bound: m0,m1 in [3*delta, 0] for decreasing profile.
    m0 = float(np.clip(d0_raw, 3.0 * delta, 0.0)) * (u1 - u0)
    m1 = 0.0

    s = (u[i_join:] - u0) / (u1 - u0)
    tail = _hermite_cubic_scalar(s, y0=y0, m0=m0, y1=y1, m1=m1)
    tail = np.maximum(tail, te_half)
    # Enforce non-increasing tail to avoid any numerical wiggle.
    for k in range(1, tail.shape[0]):
        if tail[k] > tail[k - 1]:
            tail[k] = tail[k - 1]
    tail[-1] = te_half

    out[i_join:] = tail
    return out


def _solve_tblade3_quartic_control_points(
    *,
    max_thickness_ratio: float,
    max_thickness_location: float,
    trailing_edge_thickness_ratio: float,
    leading_edge_thickness_ratio: float,
    thick_distr: int = 1,
) -> dict:
    """Port of the quartic constraint system in T-Blade3 `splinethick.f90`.

    This implementation keeps the finite-thickness trailing-edge branch
    (`thick_distr=1`) and uses geometric TE closure in section assembly.
    """
    if thick_distr not in (1,):
        raise ValueError("Only thick_distr=1 is supported.")

    degree = 4
    side_segments = 7
    ncp_side = side_segments + degree  # 11
    ncp = (2 * side_segments) + degree  # 18
    xrhs = ncp + 6  # 24
    yrhs = ncp + 4  # 22

    ix_tete_float = xrhs - 5
    ix_te_float = xrhs - 4
    ix_le_float = xrhs - 3
    ile_ee = xrhs - 2
    ite_ee = xrhs - 1
    # In the y-system, there are three explicit float unknowns corresponding to
    # TE-TE, TE and LE control locations. These map to the last three unknown
    # columns (excluding the RHS column).
    iy_tete_float = yrhs - 3
    iy_te_float = yrhs - 2
    iy_le_float = yrhs - 1

    Ax = np.zeros((xrhs - 1, xrhs), dtype=float)
    Ay = np.zeros((yrhs - 1, yrhs), dtype=float)

    ax_rhs_col = xrhs - 1
    ay_rhs_col = yrhs - 1

    def set_ax(row: int, col: int, val: float) -> None:
        Ax[row - 1, col - 1] = val

    def set_ay(row: int, col: int, val: float) -> None:
        Ay[row - 1, col - 1] = val

    def set_ax_rhs(row: int, val: float) -> None:
        Ax[row - 1, ax_rhs_col] = val

    def set_ay_rhs(row: int, val: float) -> None:
        Ay[row - 1, ay_rhs_col] = val

    # Base rows used by T-Blade3 to represent spline endpoint relationships.
    coeff = np.asarray([1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0], dtype=float)
    for i in range(1, ncp - degree + 2):  # 1..15
        Ax[i - 1, (i - 1) : (i - 1 + 4)] = coeff
        Ay[i - 1, (i - 1) : (i - 1 + 4)] = coeff

    irow = 1
    iTE = irow
    set_ax_rhs(irow, 1.0)
    set_ay_rhs(irow, 0.0)

    irow += 1
    set_ax(irow, ite_ee, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay_rhs(irow, 0.5 * trailing_edge_thickness_ratio)

    if ix_tete_float > ncp:
        irow += 1
        ix_tete = irow
        set_ax(irow, ix_tete_float, -1.0)
        set_ax_rhs(irow, 0.0)
        set_ay(irow, iy_tete_float, -1.0)
        set_ay_rhs(irow, 0.0)
    else:
        ix_tete = -1

    irow += 1
    ix_te = irow
    set_ax(irow, ix_te_float, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay(irow, iy_te_float, -1.0)
    set_ay_rhs(irow, 0.0)

    irow += 1
    imxthk = irow
    set_ax_rhs(irow, max_thickness_location)
    set_ay_rhs(irow, 0.5 * max_thickness_ratio)

    irow += 1
    ix_le = irow
    set_ax(irow, ix_le_float, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay(irow, iy_le_float, -1.0)
    set_ay_rhs(irow, 0.0)

    irow += 1
    set_ax(irow, ile_ee, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay_rhs(irow, 0.5 * leading_edge_thickness_ratio)

    irow += 1
    iLE = irow
    set_ax_rhs(irow, 0.0)
    set_ay_rhs(irow, 0.0)

    irow += 1
    set_ax(irow, ile_ee, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay_rhs(irow, -0.5 * leading_edge_thickness_ratio)

    irow += 1
    set_ax(irow, ix_le_float, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay(irow, iy_le_float, 1.0)
    set_ay_rhs(irow, 0.0)

    irow += 1
    set_ax_rhs(irow, max_thickness_location)
    set_ay_rhs(irow, -0.5 * max_thickness_ratio)

    irow += 1
    set_ax(irow, ix_te_float, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay(irow, iy_te_float, 1.0)
    set_ay_rhs(irow, 0.0)

    if ix_tete_float > ncp:
        irow += 1
        set_ax(irow, ix_tete_float, -1.0)
        set_ax_rhs(irow, 0.0)
        set_ay(irow, iy_tete_float, 1.0)
        set_ay_rhs(irow, 0.0)

    irow += 1
    set_ax(irow, ite_ee, -1.0)
    set_ax_rhs(irow, 0.0)
    set_ay_rhs(irow, -0.5 * trailing_edge_thickness_ratio)

    irow += 1
    set_ax_rhs(irow, 1.0)
    set_ay_rhs(irow, 0.0)

    irow += 1
    # TE periodic condition (finite-thickness branch).
    set_ax(irow, 1, -1.0 / 6.0)
    set_ax(irow, 2, -1.0 / 2.0)
    set_ax(irow, 3, 1.0 / 2.0)
    set_ax(irow, 4, 1.0 / 6.0)
    set_ax(irow, ncp - 3, 1.0 / 6.0)
    set_ax(irow, ncp - 2, 1.0 / 2.0)
    set_ax(irow, ncp - 1, -1.0 / 2.0)
    set_ax(irow, ncp, -1.0 / 6.0)
    set_ax_rhs(irow, 0.0)
    Ay[irow - 1, :ncp] = Ax[irow - 1, :ncp]
    set_ay_rhs(irow, 0.0)

    irow += 1
    # First derivative equality near TE (finite-thickness branch).
    set_ax(irow, 1, 1.0 / 2.0)
    set_ax(irow, 2, -1.0 / 2.0)
    set_ax(irow, 3, -1.0 / 2.0)
    set_ax(irow, 4, 1.0 / 2.0)
    set_ax(irow, ncp - 3, -1.0 / 2.0)
    set_ax(irow, ncp - 2, 1.0 / 2.0)
    set_ax(irow, ncp - 1, 1.0 / 2.0)
    set_ax(irow, ncp, -1.0 / 2.0)
    set_ax_rhs(irow, 0.0)
    Ay[irow - 1, :ncp] = Ax[irow - 1, :ncp]
    set_ay_rhs(irow, 0.0)

    irow += 1
    # Quartic-order continuity control at TE.
    set_ax(irow, 1, -1.0)
    set_ax(irow, 2, 3.0)
    set_ax(irow, 3, -3.0)
    set_ax(irow, 4, 1.0)
    set_ax(irow, ncp - 3, -1.0)
    set_ax(irow, ncp - 2, 3.0)
    set_ax(irow, ncp - 1, -3.0)
    set_ax(irow, ncp, 1.0)
    set_ax_rhs(irow, 0.0)
    Ay[irow - 1, :ncp] = Ax[irow - 1, :ncp]
    set_ay_rhs(irow, 0.0)

    ixrow = irow
    iyrow = irow

    # LE thickness-location curvature constraint (from T-Blade3).
    ixrow += 1
    set_ax(ixrow, iLE, 1.0 / 2.0)
    set_ax(ixrow, iLE + 1, -1.0 / 2.0)
    set_ax(ixrow, iLE + 2, -1.0 / 2.0)
    set_ax(ixrow, iLE + 3, 1.0 / 2.0)
    set_ax(ixrow, ile_ee, 0.0)
    set_ax_rhs(ixrow, 0.04)

    # TE thickness-location curvature constraint (finite-thickness branch).
    ixrow += 1
    set_ax(ixrow, iTE, 1.0 / 2.0)
    set_ax(ixrow, iTE + 1, -1.0 / 2.0)
    set_ax(ixrow, iTE + 2, -1.0 / 2.0)
    set_ax(ixrow, iTE + 3, 1.0 / 2.0)
    set_ax(ixrow, ite_ee, 0.0)
    set_ax_rhs(ixrow, -0.0)

    # Max-thickness control by LE.
    ixrow += 1
    set_ax(ixrow, ix_le, -1.0)
    set_ax(ixrow, ix_le + 1, 3.0)
    set_ax(ixrow, ix_le + 2, -3.0)
    set_ax(ixrow, ix_le + 3, 1.0)
    set_ax(ixrow, ix_le_float, 0.0)
    set_ax_rhs(ixrow, 0.0)

    iyrow += 1
    set_ay(iyrow, ix_le, 1.0)
    set_ay(iyrow, ix_le + 1, -4.0)
    set_ay(iyrow, ix_le + 2, 6.0)
    set_ay(iyrow, ix_le + 3, -4.0)
    set_ay(iyrow, ix_le + 4, 1.0)
    set_ay(iyrow, iy_le_float, 0.0)
    set_ay_rhs(iyrow, 0.0)

    # Max-thickness control by TE.
    ixrow += 1
    set_ax(ixrow, ix_te, 1.0)
    set_ax(ixrow, ix_te + 1, -4.0)
    set_ax(ixrow, ix_te + 2, 6.0)
    set_ax(ixrow, ix_te + 3, -4.0)
    set_ax(ixrow, ix_te + 4, 1.0)
    set_ax(ixrow, ix_te_float, 0.0)
    set_ax_rhs(ixrow, 0.0)

    iyrow += 1
    set_ay(iyrow, imxthk, -1.0 / 6.0)
    set_ay(iyrow, imxthk + 1, -1.0 / 2.0)
    set_ay(iyrow, imxthk + 2, 1.0 / 2.0)
    set_ay(iyrow, imxthk + 3, 1.0 / 6.0)
    set_ay(iyrow, iy_te_float, 0.0)
    set_ay_rhs(iyrow, 0.0)

    if ix_tete_float > ncp:
        ixrow += 1
        set_ax(ixrow, ix_tete, 1.0)
        set_ax(ixrow, ix_tete + 1, -4.0)
        set_ax(ixrow, ix_tete + 2, 6.0)
        set_ax(ixrow, ix_tete + 3, -4.0)
        set_ax(ixrow, ix_tete + 4, 1.0)
        set_ax(ixrow, ix_tete_float, 0.0)
        set_ax_rhs(ixrow, 0.0)

        iyrow += 1
        set_ay(iyrow, ix_tete, -1.0)
        set_ay(iyrow, ix_tete + 1, 3.0)
        set_ay(iyrow, ix_tete + 2, -3.0)
        set_ay(iyrow, ix_tete + 3, 1.0)
        set_ay(iyrow, iy_tete_float, 0.0)
        set_ay_rhs(iyrow, 0.0)

    if ixrow != (xrhs - 1):
        raise RuntimeError("Internal mismatch: ixrow != xrhs - 1 in quartic thickness solver.")
    if iyrow != (yrhs - 1):
        raise RuntimeError("Internal mismatch: iyrow != yrhs - 1 in quartic thickness solver.")

    A_x = Ax[:, : xrhs - 1]
    b_x = Ax[:, ax_rhs_col]
    A_y = Ay[:, : yrhs - 1]
    b_y = Ay[:, ay_rhs_col]

    # T-Blade3 solves these as square systems. For robustness in Python, we
    # fall back to least-squares if a matrix is (near-)singular.
    try:
        x_sol = np.linalg.solve(A_x, b_x)
    except np.linalg.LinAlgError:
        x_sol, *_ = np.linalg.lstsq(A_x, b_x, rcond=None)

    try:
        y_sol = np.linalg.solve(A_y, b_y)
    except np.linalg.LinAlgError:
        y_sol, *_ = np.linalg.lstsq(A_y, b_y, rcond=None)

    rx = A_x @ x_sol - b_x
    ry = A_y @ y_sol - b_y
    residual_norm = float(np.sqrt(np.dot(rx, rx) + np.dot(ry, ry)))

    xcp_full = x_sol[:ncp]
    ycp_full = y_sol[:ncp]

    ucp_top = xcp_full[:ncp_side][::-1]
    vcp_top = ycp_full[:ncp_side][::-1]
    ucp_bot = xcp_full[ncp - ncp_side :]
    vcp_bot = ycp_full[ncp - ncp_side :]

    return {
        "degree": degree,
        "ncp_side": ncp_side,
        "ucp_top": ucp_top,
        "vcp_top": vcp_top,
        "ucp_bot": ucp_bot,
        "vcp_bot": vcp_bot,
        "residual_norm": residual_norm,
    }


def _default_leading_edge_thickness_ratio(
    *,
    max_thickness_ratio: float,
    trailing_edge_thickness_ratio: float,
    leading_edge_radius_ratio: Optional[float],
    leading_edge_thickness_ratio: Optional[float],
) -> float:
    """Infer LE thickness ratio for T-Blade3-style quartic constraints."""
    if leading_edge_thickness_ratio is not None:
        le = float(leading_edge_thickness_ratio)
    elif leading_edge_radius_ratio is not None and float(leading_edge_radius_ratio) > 0.0:
        # Practical mapping from LE radius to a near-LE thickness control level.
        le = 2.0 * float(leading_edge_radius_ratio)
    else:
        le = 0.25 * float(max_thickness_ratio)

    le_min = max(1e-8, 1.5 * float(trailing_edge_thickness_ratio))
    le_max = 0.95 * float(max_thickness_ratio)
    return float(np.clip(le, le_min, le_max))


def quartic_bspline_thickness(
    u: jnp.ndarray,
    *,
    max_thickness_ratio: float,
    max_thickness_location: float,
    trailing_edge_thickness_ratio: float = 0.0,
    leading_edge_radius_ratio: Optional[float] = None,
    leading_edge_thickness_ratio: Optional[float] = None,
    trailing_edge_wedge_angle_deg: Optional[float] = None,
    n_control: int = 11,
    lambda_smooth_d2: float = 5e-4,
    lambda_smooth_d1: float = 1e-5,
    enable_le_blend: bool = False,
) -> dict:
    """Compute quartic B-spline half-thickness profile using paper/T-Blade3 constraints.

    Notes:
    - This port supports the canonical T-Blade3 quartic setup
      (`n_control=11`, degree=4 on each side), using the finite-thickness
      trailing-edge branch.
    - `lambda_smooth_*`, `enable_le_blend`, and `trailing_edge_wedge_angle_deg`
      are accepted for API compatibility.
    """
    _ = trailing_edge_wedge_angle_deg
    _ = lambda_smooth_d2
    _ = lambda_smooth_d1
    _ = enable_le_blend

    u = jnp.asarray(u, dtype=jnp.float64)
    if u.ndim != 1:
        raise ValueError("u must be a 1D array.")
    if u.shape[0] < 8:
        raise ValueError("u must have at least 8 samples.")
    if not bool(jnp.all(u[1:] >= u[:-1])):
        raise ValueError("u must be sorted ascending.")

    if int(n_control) != 11:
        raise ValueError("Paper/T-Blade3 quartic strategy currently requires n_control=11.")

    umax = float(max_thickness_location)
    if not (0.02 <= umax <= 0.98):
        raise ValueError("max_thickness_location must be in [0.02, 0.98].")

    tmax = float(max_thickness_ratio)
    tte = float(trailing_edge_thickness_ratio)
    if tmax <= 0.0:
        raise ValueError("max_thickness_ratio must be > 0.")
    if tte < 0.0:
        raise ValueError("trailing_edge_thickness_ratio must be >= 0.")
    if tte >= tmax:
        raise ValueError("trailing_edge_thickness_ratio must be smaller than max_thickness_ratio.")

    le_thk = _default_leading_edge_thickness_ratio(
        max_thickness_ratio=tmax,
        trailing_edge_thickness_ratio=tte,
        leading_edge_radius_ratio=leading_edge_radius_ratio,
        leading_edge_thickness_ratio=leading_edge_thickness_ratio,
    )

    cp = _solve_tblade3_quartic_control_points(
        max_thickness_ratio=tmax,
        max_thickness_location=umax,
        trailing_edge_thickness_ratio=tte,
        leading_edge_thickness_ratio=le_thk,
        thick_distr=1,
    )

    u_np = np.asarray(u, dtype=float)
    top = _bspline_y_of_x(
        ncp=cp["ncp_side"],
        degree=cp["degree"],
        xcp=cp["ucp_top"],
        ycp=cp["vcp_top"],
        x=u_np,
    )
    bot = _bspline_y_of_x(
        ncp=cp["ncp_side"],
        degree=cp["degree"],
        xcp=cp["ucp_bot"],
        ycp=cp["vcp_bot"],
        x=u_np,
    )

    thk_half_np = 0.5 * (top - bot)
    thk_half_np = np.maximum(thk_half_np, 0.0)
    te_half = 0.5 * tte
    thk_half_np[0] = 0.0

    # Follow T-Blade3 strategy: identify where thickness crosses target TE half-thickness.
    # Do not force endpoint to te_half here; closure is handled downstream.
    i_te = -1
    u_te = float(u_np[-1])
    for i in range(u_np.shape[0] - 1):
        if (thk_half_np[i] > te_half) and (thk_half_np[i + 1] <= te_half):
            i_te = int(i)
            t0 = float(thk_half_np[i])
            t1 = float(thk_half_np[i + 1])
            if abs(t1 - t0) < 1e-14:
                u_te = float(u_np[i])
            else:
                frac = (te_half - t0) / (t1 - t0)
                u_te = float(u_np[i] + frac * (u_np[i + 1] - u_np[i]))
            break

    thk_half = jnp.asarray(thk_half_np, dtype=jnp.float64)
    thk_full = 2.0 * thk_half
    dthk_du = jnp.asarray(np.gradient(thk_half_np, u_np), dtype=jnp.float64)

    i_max = int(np.argmax(thk_half_np))
    u_max_actual = float(u_np[i_max])
    max_thickness_ratio_actual = float(2.0 * thk_half_np[i_max])

    # Keep API-compatible fields.
    knots = open_uniform_knot_vector(n_control=11, degree=4, dtype=jnp.float64)

    return {
        "u": u,
        "thickness_half_ratio": thk_half,
        "thickness_full_ratio": thk_full,
        "dthickness_half_du": dthk_du,
        "top_thickness_ratio": jnp.asarray(top, dtype=jnp.float64),
        "bottom_thickness_ratio": jnp.asarray(bot, dtype=jnp.float64),
        "degree": int(cp["degree"]),
        "n_control": 11,
        "control_points": jnp.asarray(cp["vcp_top"], dtype=jnp.float64),
        "control_points_top_u": jnp.asarray(cp["ucp_top"], dtype=jnp.float64),
        "control_points_top_v": jnp.asarray(cp["vcp_top"], dtype=jnp.float64),
        "control_points_bottom_u": jnp.asarray(cp["ucp_bot"], dtype=jnp.float64),
        "control_points_bottom_v": jnp.asarray(cp["vcp_bot"], dtype=jnp.float64),
        "knots": knots,
        "target_max_thickness_ratio": float(tmax),
        "target_max_thickness_location": float(umax),
        "target_trailing_edge_thickness_ratio": float(tte),
        "target_leading_edge_thickness_ratio": float(le_thk),
        "max_thickness_ratio_actual": max_thickness_ratio_actual,
        "max_thickness_location_actual": u_max_actual,
        "weighted_constraint_residual_norm": float(cp["residual_norm"]),
        "te_half_ratio_target": float(te_half),
        "te_crossing_index": int(i_te),
        "te_crossing_u": float(u_te),
    }


def thickness_from_geometry(
    geometry: dict,
    *,
    u: Optional[jnp.ndarray] = None,
    n_points: int = 161,
    n_control: int = 11,
    chord_override: Optional[float] = None,
    enable_le_blend: bool = False,
    lambda_smooth_d2: float = 5e-4,
    lambda_smooth_d1: float = 1e-5,
) -> dict:
    """Build thickness profile from geometry using explicit model selection.

    Required geometry key:
    - ``thickness_model``: one of ``"NACA"``, ``"B_spline"``
    """
    chord = chord_override
    if chord is None:
        chord = geometry.get("chord", None)
    if chord is None:
        chord = geometry.get("chord_axial", geometry.get("meridional_chord", None))
    if chord is None:
        raise KeyError(
            "geometry must contain 'chord' or 'chord_axial' or 'meridional_chord', "
            "or chord_override must be provided."
        )
    chord = float(chord)
    if chord <= 0.0:
        raise ValueError("chord must be > 0.")

    model = geometry.get("thickness_model", None)
    if model is None:
        raise KeyError("geometry must include 'thickness_model' with value 'NACA' or 'B_spline'.")
    model_norm = str(model).strip().lower()
    if model_norm in ("b_spline", "bspline", "quartic_bspline"):
        model_norm = "b_spline"
    elif model_norm == "naca":
        model_norm = "naca"
    else:
        raise ValueError("Unsupported thickness_model. Allowed: 'NACA', 'B_spline'.")

    max_t = float(geometry["maximum_thickness"])
    umax = float(geometry["maximum_thickness_location_fraction"])
    tte = geometry.get("trailing_edge_thickness", None)
    if tte is None:
        if "trailing_edge_radius" not in geometry:
            raise KeyError("geometry must contain trailing_edge_radius or trailing_edge_thickness.")
        tte = 2.0 * float(geometry["trailing_edge_radius"])
    tte = float(tte)

    le_r = geometry.get("leading_edge_radius", None)
    le_t = geometry.get("leading_edge_thickness", None)
    te_wedge = geometry.get("trailing_edge_wedge_angle", None)

    if u is None:
        u = jnp.linspace(0.0, 1.0, n_points, dtype=jnp.float64)
    else:
        u = jnp.asarray(u, dtype=jnp.float64)

    if model_norm == "b_spline":
        out = quartic_bspline_thickness(
            u=u,
            max_thickness_ratio=max_t / chord,
            max_thickness_location=umax,
            trailing_edge_thickness_ratio=tte / chord,
            leading_edge_radius_ratio=None if le_r is None else float(le_r) / chord,
            leading_edge_thickness_ratio=None if le_t is None else float(le_t) / chord,
            trailing_edge_wedge_angle_deg=None if te_wedge is None else float(te_wedge),
            n_control=n_control,
            enable_le_blend=enable_le_blend,
            lambda_smooth_d2=lambda_smooth_d2,
            lambda_smooth_d1=lambda_smooth_d1,
        )
    else:
        out = naca_modified_thickness(
            u=u,
            max_thickness_ratio=max_t / chord,
            max_thickness_location=umax,
            trailing_edge_thickness_ratio=tte / chord,
            leading_edge_radius_ratio=None if le_r is None else float(le_r) / chord,
            trailing_edge_wedge_angle_deg=None if te_wedge is None else float(te_wedge),
        )

    out["chord"] = chord
    out["thickness_model"] = "B_spline" if model_norm == "b_spline" else "NACA"
    out["thickness_half"] = out["thickness_half_ratio"] * chord
    out["thickness_full"] = out["thickness_full_ratio"] * chord
    return out
