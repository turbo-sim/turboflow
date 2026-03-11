"""2D blade-section assembly from camberline and thickness distributions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp


def _as_1d(name: str, arr) -> jnp.ndarray:
    out = jnp.asarray(arr, dtype=jnp.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return out


def _interp_to_u(u_src: jnp.ndarray, y_src: jnp.ndarray, u_tgt: jnp.ndarray) -> jnp.ndarray:
    xs = np.asarray(u_src)
    ys = np.asarray(y_src)
    xt = np.asarray(u_tgt)
    return jnp.asarray(np.interp(xt, xs, ys), dtype=jnp.float64)


def _interp_scalar(u_src: jnp.ndarray, y_src: jnp.ndarray, u_val: float) -> float:
    xs = np.asarray(u_src)
    ys = np.asarray(y_src)
    return float(np.interp([float(u_val)], xs, ys)[0])


def _trailing_edge_semicircle_points(
    *,
    x_top_te: float,
    y_top_te: float,
    x_bot_te: float,
    y_bot_te: float,
    angle_te: float,
    n_points: int,
) -> tuple[np.ndarray, float]:
    """Build a downstream semicircle from top TE point to bottom TE point."""
    if int(n_points) < 3:
        return np.empty((0, 2), dtype=float), 0.0

    p_top = np.asarray([x_top_te, y_top_te], dtype=float)
    p_bot = np.asarray([x_bot_te, y_bot_te], dtype=float)
    center = 0.5 * (p_top + p_bot)
    radius = 0.5 * float(np.linalg.norm(p_top - p_bot))
    if radius <= 1e-14:
        return np.empty((0, 2), dtype=float), radius

    t_hat = np.asarray([np.cos(angle_te), np.sin(angle_te)], dtype=float)
    n_hat = np.asarray([-np.sin(angle_te), np.cos(angle_te)], dtype=float)
    # Ensure normal points from center to top endpoint.
    if float(np.dot(p_top - center, n_hat)) < 0.0:
        n_hat = -n_hat

    phi = np.linspace(0.0, np.pi, int(n_points), dtype=float)
    pts = center[None, :] + radius * (
        (np.cos(phi)[:, None] * n_hat[None, :]) + (np.sin(phi)[:, None] * t_hat[None, :])
    )
    pts[0, :] = p_top
    pts[-1, :] = p_bot
    return pts, radius


def _trailing_edge_hermite_bridge_points(
    *,
    x_top_te: float,
    y_top_te: float,
    x_bot_te: float,
    y_bot_te: float,
    tx_top: float,
    ty_top: float,
    tx_bot: float,
    ty_bot: float,
    n_points: int,
) -> tuple[np.ndarray, float]:
    """Build a C1 bridge from top TE point to bottom TE point.

    Uses cubic Hermite interpolation with endpoint tangents taken from
    the local top/bottom surface directions to avoid visible kinks.
    """
    if int(n_points) < 3:
        return np.empty((0, 2), dtype=float), 0.0

    p0 = np.asarray([x_top_te, y_top_te], dtype=float)
    p1 = np.asarray([x_bot_te, y_bot_te], dtype=float)
    d = float(np.linalg.norm(p1 - p0))
    if d <= 1e-14:
        return np.empty((0, 2), dtype=float), 0.0

    t0 = np.asarray([tx_top, ty_top], dtype=float)
    t1 = np.asarray([tx_bot, ty_bot], dtype=float)
    n0 = float(np.linalg.norm(t0))
    n1 = float(np.linalg.norm(t1))
    if n0 <= 1e-14 or n1 <= 1e-14:
        # Fallback to semicircle if endpoint tangents are degenerate.
        angle_te = float(np.arctan2((p1 - p0)[1], (p1 - p0)[0])) + 0.5 * np.pi
        return _trailing_edge_semicircle_points(
            x_top_te=x_top_te,
            y_top_te=y_top_te,
            x_bot_te=x_bot_te,
            y_bot_te=y_bot_te,
            angle_te=angle_te,
            n_points=n_points,
        )

    t0_hat = t0 / n0
    t1_hat = t1 / n1

    # Moderate tangent magnitudes for a stable smooth bridge.
    m0 = 0.35 * d * t0_hat
    m1 = 0.35 * d * t1_hat

    s = np.linspace(0.0, 1.0, int(n_points), dtype=float)
    h00 = (2.0 * s**3) - (3.0 * s**2) + 1.0
    h10 = s**3 - (2.0 * s**2) + s
    h01 = (-2.0 * s**3) + (3.0 * s**2)
    h11 = s**3 - s**2

    pts = (
        (h00[:, None] * p0[None, :])
        + (h10[:, None] * m0[None, :])
        + (h01[:, None] * p1[None, :])
        + (h11[:, None] * m1[None, :])
    )
    pts[0, :] = p0
    pts[-1, :] = p1
    return pts, 0.5 * d


def build_section_from_camber_and_thickness(
    u: jnp.ndarray,
    camber: jnp.ndarray,
    slope: jnp.ndarray,
    thickness_half: jnp.ndarray,
    *,
    clip_negative_thickness: bool = True,
    return_closed_contour: bool = True,
    close_te_with_semicircle: bool = True,
    te_arc_points: int = 31,
) -> dict:
    """Build upper/lower section coordinates in (u, v) from camber/slope/thickness.

    Uses the same geometric construction pattern as T-Blade3:
      angle = atan(slope)
      x_bot = u + t*sin(angle)
      y_bot = camber - t*cos(angle)
      x_top = u - t*sin(angle)
      y_top = camber + t*cos(angle)
    where `t` is half-thickness.
    """
    u = _as_1d("u", u)
    camber = _as_1d("camber", camber)
    slope = _as_1d("slope", slope)
    t_half = _as_1d("thickness_half", thickness_half)

    n = u.shape[0]
    if camber.shape[0] != n or slope.shape[0] != n or t_half.shape[0] != n:
        raise ValueError("u, camber, slope and thickness_half must have the same length.")

    if clip_negative_thickness:
        t_half = jnp.maximum(t_half, 0.0)

    angle = jnp.arctan(slope)

    x_bot = u + t_half * jnp.sin(angle)
    y_bot = camber - t_half * jnp.cos(angle)
    x_top = u - t_half * jnp.sin(angle)
    y_top = camber + t_half * jnp.cos(angle)

    # Geometric distance between top and bottom points at same u index.
    thickness_geom = jnp.sqrt((x_top - x_bot) ** 2 + (y_top - y_bot) ** 2)

    i_max = int(jnp.argmax(thickness_geom))
    u_max = float(u[i_max])

    out = {
        "u": u,
        "camber_u": u,
        "camber_v": camber,
        "slope": slope,
        "angle": angle,
        "thickness_half": t_half,
        "thickness_full_geom": thickness_geom,
        "x_bot": x_bot,
        "y_bot": y_bot,
        "x_top": x_top,
        "y_top": y_top,
        "max_thickness_full_geom": float(thickness_geom[i_max]),
        "max_thickness_location_u": u_max,
        "leading_edge": (float(0.5 * (x_top[0] + x_bot[0])), float(0.5 * (y_top[0] + y_bot[0]))),
        "trailing_edge": (
            float(0.5 * (x_top[-1] + x_bot[-1])),
            float(0.5 * (y_top[-1] + y_bot[-1])),
        ),
    }

    te_arc_pts = np.empty((0, 2), dtype=float)
    te_radius_geom = 0.0
    if close_te_with_semicircle:
        te_arc_pts, te_radius_geom = _trailing_edge_semicircle_points(
            x_top_te=float(x_top[-1]),
            y_top_te=float(y_top[-1]),
            x_bot_te=float(x_bot[-1]),
            y_bot_te=float(y_bot[-1]),
            angle_te=float(angle[-1]),
            n_points=int(te_arc_points),
        )
        if te_arc_pts.shape[0] > 0:
            out["x_te_arc"] = jnp.asarray(te_arc_pts[:, 0], dtype=jnp.float64)
            out["y_te_arc"] = jnp.asarray(te_arc_pts[:, 1], dtype=jnp.float64)
    out["trailing_edge_radius_geom"] = float(te_radius_geom)

    if return_closed_contour:
        # TE -> LE on bottom, then LE -> TE on top (without duplicate LE point).
        x_closed = jnp.concatenate([x_bot[::-1], x_top[1:]])
        y_closed = jnp.concatenate([y_bot[::-1], y_top[1:]])
        if close_te_with_semicircle and te_arc_pts.shape[0] > 0:
            # Append TE arc from top to bottom to close contour smoothly.
            x_closed = jnp.concatenate([x_closed, jnp.asarray(te_arc_pts[1:, 0], dtype=jnp.float64)])
            y_closed = jnp.concatenate([y_closed, jnp.asarray(te_arc_pts[1:, 1], dtype=jnp.float64)])
        out["x_closed"] = x_closed
        out["y_closed"] = y_closed

    return out


def build_section_from_results(
    camber_result: dict,
    thickness_result: dict,
    *,
    return_closed_contour: bool = True,
    close_te_with_semicircle: bool = True,
    te_arc_points: int = 31,
) -> dict:
    """Build section from existing camber and thickness result dictionaries.

    Expected camber_result keys:
    - u, camber, slope
    Expected thickness_result keys:
    - u and one of: thickness_half_ratio, thickness_half
    """
    if "u" not in camber_result or "camber" not in camber_result or "slope" not in camber_result:
        raise KeyError("camber_result must include keys: 'u', 'camber', 'slope'.")

    u = _as_1d("camber_result['u']", camber_result["u"])
    camber = _as_1d("camber_result['camber']", camber_result["camber"])
    slope = _as_1d("camber_result['slope']", camber_result["slope"])

    if "thickness_half_ratio" in thickness_result:
        t_half = _as_1d("thickness_result['thickness_half_ratio']", thickness_result["thickness_half_ratio"])
    elif "thickness_half" in thickness_result:
        chord = thickness_result.get("chord", camber_result.get("chord", None))
        if chord is None:
            # If dimensional thickness is provided and no chord, use it directly.
            t_half = _as_1d("thickness_result['thickness_half']", thickness_result["thickness_half"])
        else:
            t_half = _as_1d("thickness_result['thickness_half']", thickness_result["thickness_half"]) / float(chord)
    else:
        raise KeyError("thickness_result must include 'thickness_half_ratio' or 'thickness_half'.")

    if "u" in thickness_result:
        u_t = _as_1d("thickness_result['u']", thickness_result["u"])
        if u_t.shape[0] != t_half.shape[0]:
            raise ValueError("thickness_result['u'] and thickness array must have same length.")
        if u_t.shape[0] != u.shape[0] or not bool(jnp.allclose(u_t, u)):
            t_half = _interp_to_u(u_t, t_half, u)
    else:
        if t_half.shape[0] != u.shape[0]:
            raise ValueError(
                "Thickness array length does not match camber u and no thickness u is provided."
            )

    # T-Blade3-style TE handling:
    # the quartic thickness distribution provides the crossing location with target TE thickness.
    # Build the section up to this crossing and use a geometric TE semicircle closure from there.
    te_u = thickness_result.get("te_crossing_u", None)
    te_half_ratio_target = thickness_result.get("te_half_ratio_target", None)
    if te_u is not None and te_half_ratio_target is not None:
        te_u = float(te_u)
        te_half_ratio_target = float(te_half_ratio_target)
        if te_u < float(u[-1]) - 1e-12:
            mask = np.asarray(u) <= te_u
            u_cut = np.asarray(u)[mask]
            c_cut = np.asarray(camber)[mask]
            s_cut = np.asarray(slope)[mask]
            t_cut = np.asarray(t_half)[mask]

            if u_cut.size == 0:
                u_cut = np.asarray([float(u[0])], dtype=float)
                c_cut = np.asarray([float(camber[0])], dtype=float)
                s_cut = np.asarray([float(slope[0])], dtype=float)
                t_cut = np.asarray([float(t_half[0])], dtype=float)

            if abs(u_cut[-1] - te_u) > 1e-12:
                c_te = _interp_scalar(u, camber, te_u)
                s_te = _interp_scalar(u, slope, te_u)
                u_cut = np.concatenate([u_cut, np.asarray([te_u], dtype=float)])
                c_cut = np.concatenate([c_cut, np.asarray([c_te], dtype=float)])
                s_cut = np.concatenate([s_cut, np.asarray([s_te], dtype=float)])
                t_cut = np.concatenate([t_cut, np.asarray([te_half_ratio_target], dtype=float)])
            else:
                t_cut[-1] = te_half_ratio_target

            u = jnp.asarray(u_cut, dtype=jnp.float64)
            camber = jnp.asarray(c_cut, dtype=jnp.float64)
            slope = jnp.asarray(s_cut, dtype=jnp.float64)
            t_half = jnp.asarray(t_cut, dtype=jnp.float64)

    section = build_section_from_camber_and_thickness(
        u=u,
        camber=camber,
        slope=slope,
        thickness_half=t_half,
        return_closed_contour=return_closed_contour,
        close_te_with_semicircle=close_te_with_semicircle,
        te_arc_points=te_arc_points,
    )

    section["camber_result"] = camber_result
    section["thickness_result"] = thickness_result
    return section


def dimensionalize_section(section_uv: dict, chord: float, *, x_offset: float = 0.0, y_offset: float = 0.0) -> dict:
    """Scale a nondimensional (u, v) section by chord and optional offsets."""
    c = float(chord)
    if c <= 0.0:
        raise ValueError("chord must be > 0.")

    out = dict(section_uv)
    for k in ("camber_u", "camber_v", "x_bot", "y_bot", "x_top", "y_top", "x_closed", "y_closed"):
        if k in out:
            arr = jnp.asarray(out[k], dtype=jnp.float64)
            if k.endswith("_u") or k.startswith("x_"):
                out[k] = (arr * c) + x_offset
            elif k.endswith("_v") or k.startswith("y_"):
                out[k] = (arr * c) + y_offset
            else:
                out[k] = arr * c

    out["thickness_half_dimensional"] = jnp.asarray(out["thickness_half"]) * c
    out["thickness_full_geom_dimensional"] = jnp.asarray(out["thickness_full_geom"]) * c
    out["chord"] = c
    return out
