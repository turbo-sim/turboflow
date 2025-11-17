# plot_radial_outflow.py
# ------------------------------------------------------------
# Plotting helpers for axial & radial-outflow turbine geometry
# (depends on: turboflow.radial_outflow_turbine as rt,
#              blade_parametrization_polar_jax as bp,
#              geometry_model as gm)
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, Sequence

import warnings
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml

from turboflow.radial_outflow_turbine import blade_parametrization as bp
from turboflow.radial_outflow_turbine import geometry_model as radial_gm
from turboflow.axial_turbine import geometry_model2 as axial_gm

# ===============================================================
# Public: loader
# ===============================================================


def get_full_geometry_from_yaml(yaml_path: str) -> Dict[str, Dict[str, Any]]:
    """Load YAML and compute the full-geometry dictionary (row_name -> dict)."""
    return radial_gm.run_radial_outflow_pipeline_from_yaml(yaml_path)


# ===============================================================
# Public: original radial row plot
# ===============================================================


def plot_row_from_full_geometry(
    full_geom_by_name: Dict[str, Dict[str, Any]],
    row_name: str | Sequence[str],
    N_points: int = 800,
    title: str = "Radial outflow cascade",
):
    """
    Plot one or multiple rows on the SAME plot using the full geometry dictionary.

    Note about angles:
      - If your YAML angles are **already w.r.t tangential**, set:
            metal_angle1_deg = geom["metal_angle_in"]
            metal_angle2_deg = geom["metal_angle_out"]
        (i.e., remove the '90 -' conversion below).
      - The code below uses the previous convention (convert from meridional).
    """
    rows = [row_name] if isinstance(row_name, str) else list(row_name)
    missing = [r for r in rows if r not in full_geom_by_name]
    if missing:
        raise KeyError(
            f"Rows not found: {missing}. Available: {list(full_geom_by_name.keys())}"
        )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    r_all = []

    for rname in rows:
        geom = full_geom_by_name[rname]

        camberline_type = geom["camberline_type"]
        r1 = float(geom["r_in"])
        r2 = float(geom["r_out"])

        # If YAML angles are meridional, convert to tangential:
        metal_angle1_deg = 90.0 - float(geom["metal_angle_in"])
        metal_angle2_deg = 90.0 - float(geom["metal_angle_out"])
        # If YAML angles are already tangential, use:
        # metal_angle1_deg = float(geom["metal_angle_in"])
        # metal_angle2_deg = float(geom["metal_angle_out"])

        theta0_deg = float(geom.get("theta0", 0.0))
        N_blades = int(geom["N_blades"])

        # Rotor orientation: mirror vs stator
        sgn = +1.0 if geom["cascade_type"].lower() == "stator" else -1.0
        metal_angle1_deg *= sgn
        metal_angle2_deg *= sgn

        # Thickness / profile (absolute LE radius)
        loc_max = float(geom["maximum_thickness_location_fraction"])
        t_max = float(geom["maximum_thickness"])
        t_te = float(geom["trailing_edge_thickness"])
        wedge = jnp.deg2rad(float(geom["trailing_edge_wedge"]))
        r_le = float(geom["leading_edge_radius"])

        # Camberline
        u = jnp.linspace(0.0, 1.0, max(2, N_points))
        metal_angle1 = jnp.deg2rad(metal_angle1_deg)
        metal_angle2 = jnp.deg2rad(metal_angle2_deg)
        theta0 = jnp.deg2rad(theta0_deg)

        x_c, y_c, *_rest, chord = bp.compute_camberline_radial(
            camberline_type, r1, r2, metal_angle1, metal_angle2, theta0, u
        )

        # Full blade
        x_b, y_b, *_ = bp.compute_blade_coordinates_radial(
            camberline_type,
            r1,
            r2,
            metal_angle1,
            metal_angle2,
            theta0,
            loc_max,
            t_max,
            t_te,
            wedge,
            r_le,
            N_points,
        )

        # Annulus circles for this row
        ang = jnp.linspace(0.0, 2.0 * jnp.pi, 512)
        ax.plot(r1 * jnp.cos(ang), r1 * jnp.sin(ang), linewidth=1.0, alpha=0.85)
        ax.plot(r2 * jnp.cos(ang), r2 * jnp.sin(ang), linewidth=1.0, alpha=0.85)

        # Tile around circumference
        d_theta = 2.0 * jnp.pi / float(N_blades)
        for i in range(N_blades):
            th = d_theta * i
            Xc, Yc = bp.rotate_counterclockwise_2D(x_c, y_c, th)
            Xb, Yb = bp.rotate_counterclockwise_2D(x_b, y_b, th)
            ax.plot(Xc, Yc, ":", lw=0.9)
            ax.plot(Xb, Yb, lw=0.9)

        r_all.extend([abs(r1), abs(r2)])

    r_max = max(r_all) * 1.15 if r_all else 1.0
    ax.set_xlim([-r_max, r_max])
    ax.set_ylim([-r_max, r_max])
    ax.set_title(f"{title}\n{', '.join(rows)}", fontsize=10)

    plt.tight_layout()
    plt.show()


# ===============================================================
# Helpers
# ===============================================================


def _is_axial_geom(G: dict) -> bool:
    """Predicate: does G look like the axial geometry dict (arrays per cascade)?"""
    required = [
        "cascade_type",
        "radius_hub_in",
        "radius_hub_out",
        "radius_tip_in",
        "radius_tip_out",
        "pitch",
        "chord",
        "stagger_angle",
        "tip_clearance",
        "throat_location_fraction",
    ]
    return all(k in G for k in required) and hasattr(G["cascade_type"], "__len__")


def _axial_rows_from_geom(G: dict):
    """
    Normalize axial geometry dict -> list of per-row records with synthetic axial placement.
    Uses calculate_full_geometry outputs if available (e.g., meridional_chord, radius_mean_in).
    """
    n = len(G["cascade_type"])
    meridional_chord = jnp.asarray(G.get("meridional_chord", G["chord"]), dtype=float)
    gap_frac = 0.15  # spacing between cascades as fraction of local axial chord
    gaps = gap_frac * meridional_chord
    x_starts = jnp.cumsum(jnp.concatenate([[0.0], (meridional_chord + gaps)[:-1]]))
    x_ends = x_starts + meridional_chord

    rows = []
    for i in range(n):
        row = dict(
            name=f"Row{i+1}",
            cascade_type=str(G["cascade_type"][i]),
            r_h_in=float(G["radius_hub_in"][i]),
            r_h_out=float(G["radius_hub_out"][i]),
            r_t_in=float(G["radius_tip_in"][i]),
            r_t_out=float(G["radius_tip_out"][i]),
            tip_clearance=float(G["tip_clearance"][i]),
            chord=float(G["chord"][i]),
            pitch=float(G["pitch"][i]),
            stagger_deg=float(G["stagger_angle"][i]),
            le_angle_deg=float(G["leading_edge_angle"][i]),
            N_blades=int(
                round(
                    2.0
                    * jnp.pi
                    * float(G.get("radius_mean_in", G["radius_tip_in"])[i])
                    / float(G["pitch"][i])
                )
            ),
            x_in=float(x_starts[i]),
            x_out=float(x_ends[i]),
            throat_frac=float(G["throat_location_fraction"][i]),
        )
        r_sh_in = row["r_t_in"] + row["tip_clearance"]
        r_sh_out = row["r_t_out"] + row["tip_clearance"]
        row["r_sh_in"], row["r_sh_out"] = float(r_sh_in), float(r_sh_out)
        rows.append(row)
    return rows


def _is_radial_geom(geom: dict) -> bool:
    """Predicate: does the per-row dict look radial?"""
    return ("r_in" in geom and "r_out" in geom) or (
        geom.get("geometry_family", "").lower() == "radial"
    )


def _deg2rad(x):
    return jnp.deg2rad(float(x))


def _safe_array(a, name: str):
    try:
        return jnp.asarray(a, dtype=float)
    except Exception:
        raise ValueError(f"Expected array-like for '{name}'")


# ===============================================================
# New plot: Meridional–Tangential (Blade-to-Blade) view
# ===============================================================


def plot_meridional_tangential(
    full_geom,  # EITHER dict-of-rows (radial) OR axial geometry dict
    row_name=None,  # For radial: str | list[str]. For axial: ignored.
    N_points: int = 800,
    title: str = "Meridional–Tangential (Blade-to-Blade) view",
):
    """
    Axial: x (axial) vs y (tangential). Draws a clean placeholder blade per cascade using chord+stagger and repeats by ±pitch.
    Radial: circular tiling (r–θ mapped to x–y), using the JAX radial parametrization.
           Now uses tight layout and a consistent color per row.
    """

    # Slightly larger, higher-DPI figure for clarity
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Meridional (x or r·θ)")
    ax.set_ylabel("Tangential")
    # ---------------- AXIAL INPUT (JAX-only) ----------------
    x1_global = 0.0
    if _is_axial_geom(full_geom):

        # number of blade rows
        n = len(full_geom["cascade_type"])

        # global geometry
        y1_global = 0.0

        # camberline type (global)
        camberline_type = "linear_angle_change"

        # thickness parameters (global)
        loc_max = full_geom.get("maximum_thickness_location_fraction", 0.25)
        t_max_array = full_geom["maximum_thickness"]
        t_te_array = full_geom["trailing_edge_thickness"]
        wedge = full_geom.get("trailing_edge_wedge", 5.0)
        r_le_array = full_geom["leading_edge_diameter"] * 0.5

        # Leading / trailing metal angles arrays (global but row-dependent sign)
        cascade_types = full_geom["cascade_type"]  # list of strings
        sgn_array = jnp.array(
            [+1.0 if str(ct).lower() == "stator" else -1.0 for ct in cascade_types]
        )

        sgn_array = 1.0
        m1_array = sgn_array * (
            jnp.asarray(full_geom["leading_edge_angle"], dtype=float)
        )
        m2_array = sgn_array * (jnp.asarray(full_geom["gauging_angle"], dtype=float))

        # chord_ax may be absent → fallback to chord
        chord_ax_array = jnp.asarray(
            full_geom.get("meridional_chord", full_geom["chord"]),
            dtype=float,
        )

        pitch_array = jnp.asarray(full_geom["pitch"], dtype=float)
        chord_array = jnp.asarray(full_geom["chord"], dtype=float)
        stagger_array = jnp.deg2rad(
            jnp.asarray(full_geom["stagger_angle"], dtype=float)
        )

        # gaps and axial placement
        gap_frac = 0.15
        gaps = gap_frac * chord_ax_array

        x_starts = jnp.cumsum(
            jnp.concatenate(
                [jnp.array([0.0], dtype=float), (chord_ax_array + gaps)[:-1]]
            )
        )
        x_centers = x_starts + 0.5 * chord_ax_array

        # plot colors
        colors = ["darkorange", "steelblue"]

        def row_color(i):
            return colors[i % len(colors)]

        # =========================================================
        # Loop over blade rows
        # =========================================================
        for i in range(n):

            # -----------------------------------------
            # Read per-row scalar parameters uniformly
            # -----------------------------------------
            c_ax = float(chord_ax_array[i])  # axial chord
            phi = float(stagger_array[i])  # stagger
            P = float(pitch_array[i])  # pitch
            xmid = float(x_centers[i])  # axial offset

            beta1 = jnp.deg2rad(m1_array[i])
            beta2 = jnp.deg2rad(m2_array[i])
            r_le = float(r_le_array[i])
            t_te = float(t_te_array[i])
            t_max = float(t_max_array[i])

            color = row_color(i)

            # -----------------------------------------
            # Compute blade geometry for this row
            # -----------------------------------------
            x_b, y_b, _, _ = bp.compute_blade_coordinates_cartesian(
                camberline_type,
                x1=xmid - 0.5 * c_ax,
                y1=y1_global,
                beta1=beta1,
                beta2=beta2,
                chord_ax=c_ax,
                loc_max=loc_max,
                thickness_max=t_max,
                thickness_trailing=t_te,
                wedge_trailing=jnp.deg2rad(wedge),
                radius_leading=r_le,
                N_points=200,
            )

            # -----------------------------------------
            # Plot periodic copies
            # -----------------------------------------
            for k in (-1, 0, 1, 2, 3, 4, 5):
                label = f"Row {i+1}" if k == 0 else None
                ax.plot(x_b, y_b + k * P, color=color, lw=1.2, label=label)

            x1_global += 1.1 * c_ax

        # finalize
        ax.set_ylim([0, 3.5 * P])
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout(pad=1)

        return fig, ax

    # ---------------- RADIAL INPUT (dict of named rows) ----------------
    if not isinstance(full_geom, dict):
        raise TypeError(
            "For radial use, pass the full-geometry dict-of-rows as in your current pipeline."
        )

    rows = (
        [row_name]
        if isinstance(row_name, str)
        else (list(row_name) if row_name is not None else list(full_geom.keys()))
    )
    missing = [r for r in rows if r not in full_geom]
    if missing:
        raise KeyError(
            f"Rows not found: {missing}. Available: {list(full_geom.keys())}"
        )

    # Color: consistent per row (use tab10 cycle deterministically)
    # cmap = plt.get_cmap("tab10")
    # def row_color(i): return cmap(i % 10)
    colors = ["darkorange", "steelblue"]

    def row_color(i):
        return colors[i % 2]

    xy_max = 0.0
    for idx, rname in enumerate(rows):
        geom = full_geom[rname]
        if not _is_radial_geom(geom):
            warnings.warn(
                f"[{rname}] does not look radial; skipping in this plot.", stacklevel=1
            )
            continue

        color = row_color(idx)

        camberline_type = geom["camberline_type"]
        r1 = float(geom["r_in"])
        r2 = float(geom["r_out"])
        N_blades = int(geom["N_blades"])

        theta = jnp.linspace(0.0, 2.0 * jnp.pi, 200)
        x_circ = r1 * jnp.cos(theta)
        y_circ = r1 * jnp.sin(theta)
        ax.plot(x_circ, y_circ, "k-", linewidth=0.5)
        x_circ = r2 * jnp.cos(theta)
        y_circ = r2 * jnp.sin(theta)
        ax.plot(x_circ, y_circ, "k-", linewidth=0.5)

        # YAML angles meridional → tangential; then sign per cascade_type
        # sgn = +1.0 if str(geom.get("cascade_type","stator")).lower() == "stator" else -1.0
        # m1 = sgn * (90.0 - float(geom["metal_angle_in"]))
        # m2 = sgn * (90.0 - float(geom["metal_angle_out"]))
        m1 = float(geom["metal_angle_in"])
        m2 = float(geom["metal_angle_out"])
        theta0 = float(geom.get("theta0", 0.0))

        # thickness params
        loc_max = float(geom["maximum_thickness_location_fraction"])
        t_max = float(geom["maximum_thickness"])
        t_te = float(geom["trailing_edge_thickness"])
        wedge = _deg2rad(geom["trailing_edge_wedge"])
        r_le_array = float(geom["leading_edge_radius"])

        x_b, y_b, *_ = bp.compute_blade_coordinates_radial(
            camberline_type,
            r1,
            r2,
            _deg2rad(m1),
            _deg2rad(m2),
            _deg2rad(theta0),
            loc_max,
            t_max,
            t_te,
            wedge,
            r_le_array,
            N_points,
        )

        d_theta = 2.0 * jnp.pi / float(N_blades)
        for i in range(N_blades):
            th = d_theta * i
            Xb, Yb = bp.rotate_counterclockwise_2D(x_b, y_b, th)
            # Use SAME color for all blades of this row
            # ax.fill(Xb, Yb, color=color, alpha=0.15)
            ax.plot(Xb, Yb, lw=1.1, color=color, label=rname if i == 0 else None)

        # Keep reasonable bounds
        xy_max = max(xy_max, float(max(abs(r1), abs(r2)) * 1.1))

    # ax.set_title(title)
    ax.set_xlim(0.0, +xy_max)
    ax.set_ylim(0.0, +xy_max)
    # ax.legend(loc="best", fontsize=8)

    # Make it fill the available canvas nicely
    # ax.margins(x=0.05*xy_max, y=0.05*xy_max)
    plt.tight_layout(pad=1)
    # plt.show()
    return fig, ax


# ===============================================================
# New plot: Meridional
# ===============================================================


def plot_meridional(
    full_geom,  # EITHER axial dict OR radial dict-of-rows
    row_name=None,  # For radial: str | list[str]. For axial: ignored.
    title: str = "Meridional (side) view",
    N_curve: int = 80,  # points along r for smooth band edges (radial)
    z_center: float = 0.0,  # single shared axial center for ALL radial stages
    label_kwargs=None,  # optional: text appearance settings
):
    """
    Unified meridional (side) view plot for axial and radial-outflow turbines.

    Axial geometry:
        - Plots x (axial) vs r (radius)
        - Fills hub–tip band per cascade
        - Places row name inside each band (no legends)

    Radial-outflow geometry:
        - Plots z (axial) vs r (radius)
        - All stages share a single z_center (aligned view)
        - Each row drawn as a filled band with axial thickness equal to blade height b(r)
        - Places row name inside the band (no legends)
    """

    # Default label style (applied if user doesn’t provide one)
    if label_kwargs is None:
        label_kwargs = {
            "fontsize": 9,
            "ha": "center",
            "va": "center",
            "bbox": dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        }

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel("Axial (x or z)")
    ax.set_ylabel("Radius r")

    # ---------------- AXIAL GEOMETRY (JAX-only) ----------------
    if _is_axial_geom(full_geom):
        # Default label style if your function signature doesn't pass one
        try:
            _ = label_kwargs
        except NameError:
            label_kwargs = None
        if label_kwargs is None:
            label_kwargs = {
                "fontsize": 9,
                "ha": "center",
                "va": "center",
                "bbox": dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.25",
                ),
            }

        # JAX arrays for radii
        r_h_in = jnp.asarray(full_geom["radius_hub_in"], dtype=float)
        r_h_out = jnp.asarray(full_geom["radius_hub_out"], dtype=float)
        r_t_in = jnp.asarray(full_geom["radius_tip_in"], dtype=float)
        r_t_out = jnp.asarray(full_geom["radius_tip_out"], dtype=float)
        r_sh_in = jnp.asarray(full_geom.get("radius_shroud_in", r_t_in), dtype=float)
        r_sh_out = jnp.asarray(full_geom.get("radius_shroud_out", r_t_out), dtype=float)

        chord_ax = jnp.asarray(
            full_geom.get("meridional_chord", full_geom["chord"]), dtype=float
        )
        n = len(full_geom["cascade_type"])

        # Axial placement with small gaps (JAX-safe concat)
        gap_frac = 0.15
        gaps = gap_frac * chord_ax
        x_starts = jnp.cumsum(
            jnp.concatenate([jnp.array([0.0], dtype=float), (chord_ax + gaps)[:-1]])
        )
        x_ends = x_starts + chord_ax
        x_mids = 0.5 * (x_starts + x_ends)

        # Consistent color per row
        cmap = plt.get_cmap("tab10")

        def row_color(i):
            return cmap(i % 10)

        for i in range(n):
            color = row_color(i)

            # Linear variation inlet→outlet
            x = jnp.linspace(float(x_starts[i]), float(x_ends[i]), 3)
            r_h = jnp.linspace(float(r_h_in[i]), float(r_h_out[i]), 3)
            r_t = jnp.linspace(float(r_t_in[i]), float(r_t_out[i]), 3)
            r_sh = jnp.linspace(float(r_sh_in[i]), float(r_sh_out[i]), 3)

            # Filled hub–tip band + outlines (+ optional shroud)
            ax.fill_between(x, r_h, r_t, alpha=0.12, color=color)
            ax.plot(x, r_h, lw=1.3, color=color)
            ax.plot(x, r_t, lw=1.3, color=color)
            ax.plot(x, r_sh, lw=1.0, ls="--", color=color)

            # In-band label at mid-axial, mid-radius
            r_mid = 0.5 * (
                0.5 * (float(r_h_in[i]) + float(r_h_out[i]))
                + 0.5 * (float(r_t_in[i]) + float(r_t_out[i]))
            )
            row_name = f"{str(full_geom['cascade_type'][i]).capitalize()} {i+1}"
            ax.text(float(x_mids[i]), r_mid, row_name, **label_kwargs)

        ax.set_title(title)
        ax.margins(x=0.03, y=0.03)
        plt.tight_layout()
        plt.show()
        return

    # ---------------- RADIAL GEOMETRY ----------------
    if not isinstance(full_geom, dict):
        raise TypeError(
            "For radial use, pass the full-geometry dict-of-rows produced by your radial pipeline."
        )

    # Collect selected rows
    rows = (
        [row_name]
        if isinstance(row_name, str)
        else (list(row_name) if row_name is not None else list(full_geom.keys()))
    )
    missing = [r for r in rows if r not in full_geom]
    if missing:
        raise KeyError(
            f"Rows not found: {missing}. Available: {list(full_geom.keys())}"
        )

    for rname in rows:
        geom = full_geom[rname]
        if not _is_radial_geom(geom):
            warnings.warn(f"[{rname}] does not look radial; skipping.", stacklevel=1)
            continue

        # Required parameters
        r_in = float(geom["r_in"])
        r_out = float(geom["r_out"])
        b_in = float(geom["blade_height_in"])
        b_out = float(geom["blade_height_out"])

        # Blade height distribution (linear)
        s = jnp.linspace(0.0, 1.0, int(max(3, N_curve)))
        r = r_in + s * (r_out - r_in)
        b = b_in + s * (b_out - b_in)

        # Shared z_center for all rows
        z_front = z_center - 0.5 * b
        z_back = z_center + 0.5 * b

        # Draw band
        ctype = str(geom.get("cascade_type", "")).lower()
        color = "C0" if ctype == "stator" else "C1" if ctype == "rotor" else "C2"
        ax.fill_betweenx(r, z_front, z_back, alpha=0.15, color=color)
        ax.plot(z_front, r, lw=1.3, color=color)
        ax.plot(z_back, r, lw=1.3, color=color)

        # Label at mid-radius, centered axially
        r_mid = 0.5 * (r_in + r_out)
        ax.text(z_center, r_mid, rname, color="black", **label_kwargs)

    ax.set_title(title)
    ax.set_xlabel("Axial (z)")
    plt.tight_layout()
    plt.show()


def plot_meridional_hybrid(
    yaml_path: str,
    title: str = "Hybrid turbine - meridional view",
    z_center: float = 0.0,  # shared axial center for ALL radial rows (unchanged)
    N_curve: int = 80,  # smoothness for radial bands
    label_kwargs: dict | None = None,
    gap_frac: float = 0.15,  # axial gap fraction between *axial* rows only
    axial_fallback_chord_factor: float = 0.6,  # used if axial chord unavailable
    gap_multiplier: float = 0.8,  # EXTRA spacing multiplier between radial and axial blocks
):
    """
    Hybrid meridional plot from a single YAML:
      • RADIAL rows (r_in/out + blade_height_in/out): identical rendering to your good radial plot
      • AXIAL rows: plotted to the RIGHT of the radial block, after the last radial stage,
        with an extra spacing controlled by 'gap_multiplier'.
        Axial handling:
          - If hub/tip radii are provided: calculate_full_geometry(row-wise list)
          - If design variables are provided: prepare_geometry(..., radius_type) -> calculate_full_geometry(...)
          - If only mean+span are provided: plot via mean±span/2 with safe axial-chord fallback
    """

    if label_kwargs is None:
        label_kwargs = {
            "fontsize": 9,
            "ha": "center",
            "va": "center",
            "bbox": dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        }

    # --- Load YAML ---
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    rows = cfg.get("geometry", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("YAML must contain a 'geometry' list with row dictionaries.")

    # radius_type for axial prepare() if present
    radius_type = (
        cfg.get("design_optimization", {}).get("radius_type")
        or cfg.get("performance_analysis", {}).get("radius_type")
        or "constant_mean"
    )

    # --- Classifiers ---
    def is_radial_row(d: dict) -> bool:
        return all(
            k in d for k in ("r_in", "r_out", "blade_height_in", "blade_height_out")
        )

    def is_axial_std_row(d: dict) -> bool:
        return all(
            k in d
            for k in (
                "radius_hub_in",
                "radius_hub_out",
                "radius_tip_in",
                "radius_tip_out",
            )
        )

    def is_axial_mean_row(d: dict) -> bool:
        return all(
            k in d
            for k in ("r_mean_in", "r_mean_out", "blade_height_in", "blade_height_out")
        )

    def is_axial_design_row(d: dict) -> bool:
        needed = [
            "radius",
            "hub_tip_ratio_in",
            "hub_tip_ratio_out",
            "aspect_ratio",
            "pitch_chord_ratio",
            "gauging_angle",
            "leading_edge_angle",
            "trailing_edge_thickness_opening_ratio",
            "leading_edge_diameter",
            "leading_edge_wedge_angle",
            "tip_clearance",
            "throat_location_fraction",
            "cascade_type",
        ]
        return all(k in d for k in needed)

    # --- Partition while preserving order ---
    ordered = []  # ('radial'|'axial_std'|'axial_mean'|'axial_design', name)
    radial_items = []
    axial_std_items = []
    axial_mean_items = []
    axial_design_items = []

    for item in rows:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(
                "Each entry in 'geometry' must be a single-key dict, e.g. {'stator_1': {...}}"
            )
        name, data = next(iter(item.items()))
        if is_radial_row(data):
            ordered.append(("radial", name))
            radial_items.append({name: data})
        elif is_axial_std_row(data):
            ordered.append(("axial_std", name))
            axial_std_items.append({name: data})
        elif is_axial_design_row(data):
            ordered.append(("axial_design", name))
            axial_design_items.append({name: data})
        elif is_axial_mean_row(data):
            ordered.append(("axial_mean", name))
            axial_mean_items.append({name: data})
        else:
            warnings.warn(
                f"[{name}] row not recognized as radial/axial; skipping.", stacklevel=1
            )

    # --- Build full geometry for RADIAL subset (unchanged behavior) ---
    full_radial = {}
    meridional_half_spans = []  # half of meridional chord per radial row

    if radial_items:
        cfg_r = {"geometry": radial_items}
        prepared_r = radial_gm.prepare_all_rows(cfg_r)
        full_radial = radial_gm.calculate_full_geometries(
            prepared_r
        )  # {row_name: full_row_dict}

        for item in radial_items:
            nm = next(iter(item.keys()))
            row = full_radial[nm]
            # meridional chord: use provided or chord*cos(stagger)
            if "meridional_chord" in row:
                c_mer = float(row["meridional_chord"])
            else:
                c_mer = float(row["chord"] * jnp.cos(jnp.deg2rad(row["stagger_angle"])))
            meridional_half_spans.append(0.5 * max(c_mer, 1e-12))

    half_span_radial = max(meridional_half_spans) if meridional_half_spans else 0.0

    # --- Build full geometry for AXIAL subsets ---
    full_axial_std = {}
    axial_std_names = []
    if axial_std_items:
        # geometry_model accepts row-wise list (your updated model coerce routine handles it)
        full_axial_std = axial_gm.calculate_full_geometry(axial_std_items)
        axial_std_names = [next(iter(d.keys())) for d in axial_std_items]

    full_axial_design = {}
    axial_design_names = []
    if axial_design_items:
        # flatten to dict-of-arrays for prepare_geometry
        acc = {}
        ctypes = []
        for item in axial_design_items:
            nm, dat = next(iter(item.items()))
            axial_design_names.append(nm)
            ctype = str(dat["cascade_type"]).lower()
            if ctype in ("0", "0.0"):
                ctype = "stator"
            if ctype in ("1", "1.0"):
                ctype = "rotor"
            ctypes.append(ctype)
            for k, v in dat.items():
                if k == "cascade_type":
                    continue
                acc.setdefault(k, []).append(float(v))
        design_block = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in acc.items()}
        design_block["cascade_type"] = ctypes  # keep as list[str]
        prepared_ax = axial_gm.prepare_geometry(design_block, radius_type=radius_type)
        full_axial_design = axial_gm.calculate_full_geometry(prepared_ax)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=140)
    ax.set_aspect("equal")
    # cmap = plt.get_cmap("tab10")
    # def row_color(i): return cmap(i % 10)
    colors = ["darkorange", "steelblue"]

    def row_color(i):
        return colors[i % 2]

    # Axial block starts AFTER the last radial stage (+ extra spacing)
    radial_width = 2.0 * half_span_radial if half_span_radial > 0 else 1.0
    z_last_radial_edge = z_center + half_span_radial
    base_gap = gap_multiplier * radial_width
    s_axial = z_last_radial_edge + base_gap
    gap_eps = 1e-12

    color_idx = 0
    axial_std_counter = -1
    axial_design_counter = -1

    for kind, name in ordered:
        color = row_color(color_idx)

        if kind == "radial":
            color_idx += 1
            geom = full_radial.get(name)
            if geom is None:
                continue

            r_in = float(geom["r_in"])
            r_out = float(geom["r_out"])
            b_in = float(geom["blade_height_in"])
            b_out = float(geom["blade_height_out"])

            s = jnp.linspace(0.0, 1.0, int(max(3, N_curve)))
            r = r_in + s * (r_out - r_in)
            b = b_in + s * (b_out - b_in)

            z_front = z_center - 0.5 * b
            z_back = z_center + 0.5 * b

            ax.fill_betweenx(r, z_front, z_back, alpha=0.15, color=color)
            ax.plot(z_front, r, lw=1.3, color=color)
            ax.plot(z_back, r, lw=1.3, color=color)

            r_mid = 0.5 * (r_in + r_out)
            ax.text(z_center, r_mid, name.replace("_", " ").title(), **label_kwargs)

        elif kind == "axial_std":
            color_idx += 1
            axial_std_counter += 1

            row_a = {
                k: (v[axial_std_counter] if isinstance(v, jnp.ndarray) else v)
                for k, v in full_axial_std.items()
                if k != "cascade_type"
            }

            hub_in = float(row_a["radius_hub_in"])
            hub_out = float(row_a["radius_hub_out"])
            tip_in = float(row_a["radius_tip_in"])
            tip_out = float(row_a["radius_tip_out"])

            if "meridional_chord" in row_a:
                c_ax = float(row_a["meridional_chord"])
            elif "chord" in row_a and "stagger_angle" in row_a:
                c_ax = float(
                    row_a["chord"] * jnp.cos(jnp.deg2rad(row_a["stagger_angle"]))
                )
            else:
                mean_span = 0.5 * ((tip_in - hub_in) + (tip_out - hub_out))
                c_ax = max(axial_fallback_chord_factor * float(mean_span), 1e-3)

            gap = gap_frac * max(c_ax, gap_eps)
            x0, x1 = s_axial, s_axial + c_ax

            x = jnp.linspace(x0, x1, 3)
            r_h = jnp.linspace(hub_in, hub_out, 3)
            r_t = jnp.linspace(tip_in, tip_out, 3)

            ax.fill_between(x, r_h, r_t, alpha=0.12, color=color)
            ax.plot(x, r_h, lw=1.3, color=color)
            ax.plot(x, r_t, lw=1.3, color=color)

            x_mid = 0.5 * (x0 + x1)
            r_mid = 0.5 * (0.5 * (hub_in + hub_out) + 0.5 * (tip_in + tip_out))
            ax.text(x_mid, r_mid, name.replace("_", " ").title(), **label_kwargs)

            s_axial = x1 + gap

        elif kind == "axial_design":
            color_idx += 1
            axial_design_counter += 1

            row_a = {
                k: (v[axial_design_counter] if isinstance(v, jnp.ndarray) else v)
                for k, v in full_axial_design.items()
                if k != "cascade_type"
            }

            hub_in = float(row_a["radius_hub_in"])
            hub_out = float(row_a["radius_hub_out"])
            tip_in = float(row_a["radius_tip_in"])
            tip_out = float(row_a["radius_tip_out"])

            if "meridional_chord" in row_a:
                c_ax = float(row_a["meridional_chord"])
            elif "chord" in row_a and "stagger_angle" in row_a:
                c_ax = float(
                    row_a["chord"] * jnp.cos(jnp.deg2rad(row_a["stagger_angle"]))
                )
            else:
                mean_span = 0.5 * ((tip_in - hub_in) + (tip_out - hub_out))
                c_ax = max(axial_fallback_chord_factor * float(mean_span), 1e-3)

            gap = gap_frac * max(c_ax, gap_eps)
            x0, x1 = s_axial, s_axial + c_ax

            x = jnp.linspace(x0, x1, 3)
            r_h = jnp.linspace(hub_in, hub_out, 3)
            r_t = jnp.linspace(tip_in, tip_out, 3)

            ax.fill_between(x, r_h, r_t, alpha=0.12, color=color)
            ax.plot(x, r_h, lw=1.3, color=color)
            ax.plot(x, r_t, lw=1.3, color=color)

            x_mid = 0.5 * (x0 + x1)
            r_mid = 0.5 * (0.5 * (hub_in + hub_out) + 0.5 * (tip_in + tip_out))
            ax.text(x_mid, r_mid, name.replace("_", " ").title(), **label_kwargs)

            s_axial = x1 + gap

        else:  # 'axial_mean' fallback (mean ± span/2)
            color_idx += 1
            data = next(d[name] for d in axial_mean_items if name in d)

            rmin = float(data["r_mean_in"])
            rmax = float(data["r_mean_out"])
            b_in = float(data["blade_height_in"])
            b_out = float(data["blade_height_out"])

            hub_in = rmin - 0.5 * b_in
            tip_in = rmin + 0.5 * b_in
            hub_out = rmax - 0.5 * b_out
            tip_out = rmax + 0.5 * b_out

            if ("chord" in data) and ("stagger_angle" in data):
                c_ax = float(
                    data["chord"] * jnp.cos(jnp.deg2rad(data["stagger_angle"]))
                )
            else:
                mean_span = 0.5 * (b_in + b_out)
                c_ax = max(axial_fallback_chord_factor * float(mean_span), 1e-3)

            gap = gap_frac * max(c_ax, gap_eps)
            x0, x1 = s_axial, s_axial + c_ax

            x = jnp.linspace(x0, x1, 3)
            r_h = jnp.linspace(hub_in, hub_out, 3)
            r_t = jnp.linspace(tip_in, tip_out, 3)

            ax.fill_between(x, r_h, r_t, alpha=0.12, color=color)
            ax.plot(x, r_h, lw=1.3, color=color)
            ax.plot(x, r_t, lw=1.3, color=color)

            x_mid = 0.5 * (x0 + x1)
            r_mid = 0.5 * (0.5 * (hub_in + hub_out) + 0.5 * (tip_in + tip_out))
            ax.text(x_mid, r_mid, name.replace("_", " ").title(), **label_kwargs)

            s_axial = x1 + gap

    # ==== Axis limits ====
    radial_width = 2.0 * half_span_radial if half_span_radial > 0 else 1.0
    left_edge = z_center - half_span_radial
    right_edge = max(s_axial, z_center + half_span_radial)
    ax.set_xlim(left_edge - 2.25 * radial_width, right_edge + 2.25 * radial_width)
    ax.set_ylim(0.0, None)  # auto top

    ax.set_title(title)
    ax.set_xlabel("Station coordinate (z for radial, x for axial) [m]")
    ax.set_ylabel("Radius r [m]")
    plt.tight_layout(pad=1)
    # plt.show()
    return fig, ax
