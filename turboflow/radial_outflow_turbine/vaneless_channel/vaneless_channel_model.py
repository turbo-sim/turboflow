import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import jaxprop as jxp
import optimistix as optx
import nurbspy.jax as nrb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolor
import matplotlib.cm as cm

from typing import Any
from jaxtyping import Array, Float, Scalar

from .friction_models import FrictionModel, make_friction_model
from .heat_models import HeatTransferModel, make_heat_model
jxp.set_plot_options(grid=False)


# -------------------------
# Define Equinox modules
# -------------------------
class Geometry(eqx.Module):
    z_in: Float[Array, ""]  # Scalar JAX array
    z_out: Float[Array, ""]
    r_in: Float[Array, ""]
    r_out: Float[Array, ""]
    b_in: Float[Array, ""]
    b_out: Float[Array, ""]
    phi_in: Float[Array, ""]
    phi_out: Float[Array, ""]
    td_in: Float[Array, ""]
    td_out: Float[Array, ""]


class OperatingConditions(eqx.Module):
    # always stored in static form internally
    p_in: Float[Array, ""]
    h_in: Float[Array, ""]
    v_in: Float[Array, ""]
    alpha_in: Float[Array, ""]

    @classmethod
    def from_dict(cls, config: dict, fluid):
        """
        Initialize either from static or stagnation inputs.
        The 'operating_conditions' section of the config must
        contain either (p_in, h_in, v_in, alpha_in) or
        (p0_in, T0_in, Ma_in, alpha_in).
        """

        oc = config["operating_conditions"]

        # Option 1: static inputs
        if all(k in oc for k in ["p_in", "h_in", "v_in", "alpha_in"]):
            return cls(
                p_in=jnp.array(oc["p_in"]),
                h_in=jnp.array(oc["h_in"]),
                v_in=jnp.array(oc["v_in"]),
                alpha_in=jnp.array(oc["alpha_in"]),
            )

        # Option 2: stagnation inputs
        elif all(k in oc for k in ["p0_in", "T0_in", "Ma_in", "alpha_in"]):
            p, h, v = convert_stagnation_to_static(
                jnp.array(oc["p0_in"]),
                jnp.array(oc["T0_in"]),
                jnp.array(oc["Ma_in"]),
                fluid,
            )
            return cls(
                p_in=p,
                h_in=h,
                v_in=v,
                alpha_in=jnp.array(oc["alpha_in"]),
            )

        else:
            provided = list(oc.keys())
            raise ValueError(
                "Invalid operating conditions configuration.\n"
                "Expected either:\n"
                "  - (p_in, h_in, v_in, alpha_in)  [static form], or\n"
                "  - (p0_in, T0_in, Ma_in, alpha_in)  [stagnation form].\n"
                f"Provided keys: {provided}"
            )


class SolverOptions(eqx.Module):
    solver_name: str = eqx.field(static=True)
    adjoint_name: str = eqx.field(static=True)
    throw: bool = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    n_points: int = eqx.field(static=True)


class ModelOptions(eqx.Module):
    """Holds all model-specific configuration (heat + friction)."""

    heat_transfer: HeatTransferModel
    friction: FrictionModel


class VanelessChannel(eqx.Module):
    name: str = eqx.field(static=True)
    geometry: Geometry
    operating_conditions: OperatingConditions
    model_options: ModelOptions
    solver_options: SolverOptions
    fluid: Any

    @classmethod
    def from_dict(cls, config: dict, fluid):
        """Create VanelessDiffuser from a nested dictionary."""

        # Helper to convert dict values to JAX arrays
        def to_jax_dict(d: dict) -> dict:
            return {
                k: jnp.array(v) if isinstance(v, (int, float)) else v
                for k, v in d.items()
            }

        # Create friction and heat transfer models
        friction_cfg = config["model_options"]["friction_model"]
        friction_model = make_friction_model(friction_cfg)
        heat_cfg = config["model_options"]["heat_model"]
        heat_model = make_heat_model(heat_cfg)
        model_options = ModelOptions(
            heat_transfer=heat_model,
            friction=friction_model,
        )

        return cls(
            name=config["name"],
            geometry=Geometry(**to_jax_dict(config["geometry"])),
            operating_conditions=OperatingConditions.from_dict(config, fluid),
            model_options=model_options,
            solver_options=SolverOptions(**config.get("solver_options", {})),
            fluid=fluid,
        )

    def solve(self):
        """Solve the vaneless diffuser flow problem."""
        return solve_vaneless_channel_model(
            self.geometry,
            self.operating_conditions,
            self.model_options,
            self.solver_options,
            self.fluid,
        )
    
    def make_geometry(self):
        geom_handle = make_vaneless_channel_geometry(self.geometry)
        return geom_handle
    
    def plot_geometry(
        self,
        fig=None,
        ax=None,
        n_points: int = 200,
        plot_midline: bool = True,
        plot_hub: bool = True,
        plot_shroud: bool = True,
        plot_inlet: bool = True,
        plot_outlet: bool = True,
        plot_control_points: bool = False,
    ):
        """
        Plot the meridional channel geometry using a geometry handle from
        make_vaneless_channel_geometry().

        Parameters
        ----------
        geom_handle : callable
            Geometry evaluator function geom_handle(s) → dict of geometry properties.
        fig, ax : matplotlib Figure and Axes, optional
            Existing figure/axes to plot on. If None, new ones are created.
        n_points : int, optional
            Number of evaluation points along the channel (default 200).
        plot_midline : bool, optional
            Whether to plot the channel midline (default True).
        plot_hub : bool, optional
            Whether to plot the hub wall (default True).
        plot_shroud : bool, optional
            Whether to plot the shroud wall (default True).
        plot_inlet : bool, optional
            Whether to plot the inlet edge (default True).
        plot_outlet : bool, optional
            Whether to plot the outlet edge (default True).
        plot_control_points : bool, optional
            Whether to to plot the control points of the midline NURBS (default False).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes objects
        """

        # Create geometry
        geom_handle = make_vaneless_channel_geometry(self.geometry)

        # Sample the geometry along the midline
        s = jnp.linspace(0.0, geom_handle(s=0.0)["s_total"], n_points)
        geom = geom_handle(s)

        # Extract geometry data
        z_mid = geom["z"]
        r_mid = geom["r"]
        z_shroud = geom["z_shroud"]
        r_shroud = geom["r_shroud"]
        z_hub = geom["z_hub"]
        r_hub = geom["r_hub"]

        # Compute y-axis (radial) limits
        r_max = jnp.max(r_shroud)
        r_top = 1.1 * r_max  # 10% margin above the shroud radius
        z_min = jnp.max(z_shroud)

        # Create figure and axes if needed
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$z$ $-$ Axial coordinate  [m]")
            ax.set_ylabel(r"$r$ $-$ Radial coordinate [m]")

            decimals = 2
            fmt = mticker.FormatStrFormatter(f"%.{decimals}f")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

            r_max = float(jnp.max(r_shroud))
            r_top = 1.1 * r_max
            z_min = float(jnp.min(jnp.concatenate([z_hub, z_shroud])))
            z_max = float(jnp.max(jnp.concatenate([z_hub, z_shroud])))
            dz = 0.2 * (z_max - z_min)
            z_left = z_min - dz
            z_right = z_max + dz
            ax.set_xlim(left=z_left, right=z_right)
            ax.set_ylim(bottom=0.0, top=r_top)

        # Plot requested items
        if plot_hub:
            ax.plot(z_hub, r_hub, color="black", linestyle="-")

        if plot_shroud:
            ax.plot(z_shroud, r_shroud, color="black", linestyle="-")

        if plot_midline:
            ax.plot(z_mid, r_mid, color="black", linestyle="-.")

        if plot_inlet:
            ax.plot(
                [z_hub[0], z_shroud[0]],
                [r_hub[0], r_shroud[0]],
                color="black",
                linestyle="-",
            )
        if plot_outlet:
            ax.plot(
                [z_hub[-1], z_shroud[-1]],
                [r_hub[-1], r_shroud[-1]],
                color="black",
                linestyle="-",
            )

        if plot_control_points:
            P = geom_handle.nurbs_midline.P
            zc, rc = P[0], P[1]
            ax.plot(zc, rc, "ro", linestyle="-.")

        fig.tight_layout(pad=1)

        return fig, ax

    def plot_solution_contour(
        self,
        solution: dict,
        var_name: str,
        fig=None,
        ax=None,
        levels=100,
        cmap=None,
        cbar: bool = True,
        label: str | None = None,
        show_channel: bool = True,
        channel_kwargs: dict | None = None,
    ):
        """
        Plot a 2D contour of a 1D solution variable along the meridional channel walls.

        Parameters
        ----------
        geom_handle : callable
            Geometry evaluator function geom_handle(s) → dict of geometry properties.
        solution : dict
            Output dictionary from the diffuser solver or postprocessing function.
            Must contain 'm' (meridional coordinate) and the requested variable.
        var_name : str
            Name of the variable in the solution dictionary to visualize
            (e.g. "p", "Ma", "Cp", "v_m").
        fig, ax : matplotlib Figure and Axes, optional
            Existing figure/axes to plot on. If None, new ones are created.
        cmap : matplotlib Colormap, optional
            Colormap object to use for the contour fill (default: plt.cm.magma).
        label : str, optional
            Label for the colorbar.
        cbar : bool, optional
            Whether to include a colorbar (default True).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Figure and axes with the plotted contours.
        """

        # Create geometry
        geom_handle = make_vaneless_channel_geometry(self.geometry)

        # Extract data from the solution dictionary
        if "m" not in solution:
            raise KeyError(
                "Solution dictionary must contain key 'm' (meridional coordinate)."
            )
        if var_name not in solution:
            raise KeyError(f"Variable '{var_name}' not found in solution dictionary.")

        if cmap is None:
            cmap = mcolor.LinearSegmentedColormap.from_list(
                "soft_Blues", plt.cm.Blues(jnp.linspace(0.3, 0.9, 256))
            )

        s_vec = solution["m"]
        var_vec = solution[var_name]

        # Evaluate geometry
        geom = geom_handle(s_vec)
        z_hub, r_hub = geom["z_hub"], geom["r_hub"]
        z_shroud, r_shroud = geom["z_shroud"], geom["r_shroud"]

        # Build 2D interpolation grid between hub and shroud
        spanwise = jnp.linspace(0.0, 1.0, 50)  # increase for smoother interpolation
        z_grid = jnp.outer(z_hub, 1 - spanwise) + jnp.outer(z_shroud, spanwise)
        r_grid = jnp.outer(r_hub, 1 - spanwise) + jnp.outer(r_shroud, spanwise)
        var_grid = jnp.outer(var_vec, jnp.ones_like(spanwise))

        # Create figure and axes if needed
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$z$ $-$ Axial coordinate  [m]")
            ax.set_ylabel(r"$r$ $-$ Radial coordinate [m]")

            r_max = jnp.max(r_shroud)
            r_top = 1.1 * r_max  # 10% margin above the shroud radius
            z_min = jnp.max(z_shroud)
            decimals = 2
            fmt = mticker.FormatStrFormatter(f"%.{decimals}f")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

            r_max = float(jnp.max(r_shroud))
            r_top = 1.1 * r_max
            z_min = float(jnp.min(jnp.concatenate([z_hub, z_shroud])))
            z_max = float(jnp.max(jnp.concatenate([z_hub, z_shroud])))
            dz = 0.2 * (z_max - z_min)
            z_left = z_min - dz
            z_right = z_max + dz
            ax.set_xlim(left=z_left, right=z_right)
            ax.set_ylim(bottom=0.0, top=r_top)

        # Plot the filled contour
        cf = ax.contourf(
            z_grid,
            r_grid,
            var_grid,
            levels=levels,
            cmap=cmap,
        )

        # Optionally overlay the channel
        if show_channel:
            channel_kwargs = channel_kwargs or {}
            self.plot_geometry(
                fig=fig,
                ax=ax,
                **channel_kwargs,
            )

        # Optional colorbar
        if cbar:
            cb = fig.colorbar(cf, ax=ax)
            cb.set_label(label if label else var_name)

        fig.tight_layout(pad=1)

        return fig, ax

    def plot_streamlines(
        self,
        solution,
        fig=None,
        ax=None,
        color="black",
        label="",
        number_of_streamlines=5,
    ):
        """
        Plot streamlines and circular inlet/outlet boundaries.

        Parameters
        ----------
        out : dict
            Contains the radial and angular coordinates of the flow field:
            - 'r' : array of radii
            - 'theta' : array of angles
        params : dict
            Geometry parameters including:
            - params["geometry"]["r_in"]
            - params["geometry"]["r_out"]
        number_of_streamlines : int, optional
            Number of streamlines to plot (default 5).
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on. Creates a new one if None.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """

        # Create figure and axes if needed
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("$x$ coordinate")
            ax.set_ylabel("$y$ coordinate")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)

        # Inlet and outlet circles
        r_in = self.geometry.r_in
        r_out = self.geometry.r_out
        theta = jnp.linspace(0, 2 * jnp.pi, 100)
        ax.plot(r_in * jnp.cos(theta), r_in * jnp.sin(theta), "k")
        ax.plot(r_out * jnp.cos(theta), r_out * jnp.sin(theta), "k")

        # Streamlines
        theta_stream = jnp.linspace(0, 2 * jnp.pi, number_of_streamlines + 1)
        for i, t in enumerate(theta_stream):
            x = solution["r"] * jnp.cos(solution["theta"] + t)
            y = solution["r"] * jnp.sin(solution["theta"] + t)
            if i == 0:
                ax.plot(x, y, color=color, label=label)
            else:
                ax.plot(x, y, color=color)

        # Axis limits
        limit = 1.1 * r_out
        ax.axis([-limit, limit, -limit, limit])

        fig.tight_layout(pad=1)
        return fig, ax

    def plot_efficiency_breakdown(self, solution, cmap_name=None):
        """
        Plot the pressure recovery and loss breakdown.

        Parameters
        ----------
        out : dict
            Contains:
            - 'm'
            - 'efficiency'
            - 'efficiency_kinetic'
            - 'efficiency_loss_wall'
            - 'efficiency_loss_diffusion'
            - 'efficiency_loss_curvature'
        cmap_name : str or None, optional
            Name of the colormap (e.g. 'Blues', 'magma', 'viridis').
            If None, a black-and-white hatched style is used.
        """
        m = solution["m"]
        eff = solution["efficiency"]
        eff_kin = solution["efficiency_kinetic"]
        eff_loss_wall = solution["efficiency_loss_wall"]
        eff_loss_diff = solution["efficiency_loss_diffusion"]
        eff_loss_curv = solution["efficiency_loss_curvature"]

        # Prepare colors depending on mode
        if cmap_name:
            cmap = cm.get_cmap(cmap_name)
            colors = [
                cmap(0.2),  # wall loss
                cmap(0.4),  # diffusion
                cmap(0.6),  # curvature
                cmap(0.8),  # kinetic
            ]
            hatches = [None] * 4  # no hatching when using color
        else:
            # grayscale mode for print
            # colors = ["lightgray", "gainsboro", "lightgray", "white"]
            colors = ["white"] * 4
            hatches = ["///", "\\\\", "xxxxx", ""]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlabel("Meridional coordinate")
        ax.set_ylabel("Recovery efficiency breakdown")

        bottom = jnp.zeros_like(m)

        # 1) Static-to-total efficiency
        ax.fill_between(
            m,
            bottom,
            eff,
            facecolor="lightgray",
            edgecolor="k",
            linewidth=0.6,
            label="Recovery efficiency",
            zorder=3,
        )
        bottom = eff

        # 2) Wall loss
        ax.fill_between(
            m,
            bottom,
            bottom + eff_loss_wall,
            facecolor=colors[0],
            edgecolor="k",
            linewidth=0.5,
            hatch=hatches[0],
            label="Wall loss",
            zorder=3,
        )
        bottom += eff_loss_wall

        # 3) Diffusion loss
        ax.fill_between(
            m,
            bottom,
            bottom + eff_loss_diff,
            facecolor=colors[1],
            edgecolor="k",
            linewidth=0.5,
            hatch=hatches[1],
            label="Diffusion loss",
            zorder=3,
        )
        bottom += eff_loss_diff

        # 4) Curvature loss
        ax.fill_between(
            m,
            bottom,
            bottom + eff_loss_curv,
            facecolor=colors[2],
            edgecolor="k",
            linewidth=0.5,
            hatch=hatches[2],
            label="Curvature loss",
            zorder=3,
        )
        bottom += eff_loss_curv

        # 5) Kinetic contribution
        ax.fill_between(
            m,
            bottom,
            bottom + eff_kin,
            facecolor=colors[3],
            edgecolor="k",
            linewidth=0.5,
            hatch=hatches[3],
            label="Kinetic energy",
            zorder=3,
        )
        bottom += eff_kin

        ax.set_xlim(0.0, m[-1])
        ax.set_ylim(0.0, 1.00)
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        fig.tight_layout(pad=1)
        return fig, ax

    def plot_skin_friction_distribution(self, solution, fig=None, ax=None):
        """
        Plot the distribution of the skin friction coefficient along the meridional length.

        Parameters
        ----------
        out : dict
            Dictionary containing:
                - 'm' : meridional coordinate [m]
                - 'cf_total' : total Fanning friction coefficient
                - 'cf_wall' : wall friction contribution
                - 'cf_diffusion' : diffusion contribution
                - 'cf_curvature' : curvature contribution
        fig, ax : matplotlib Figure and Axes, optional
            Existing figure/axes to plot on. If None, new ones are created.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """

        # Validate input
        required_keys = ["m", "cf_total", "cf_wall", "cf_diffusion", "cf_curvature"]
        for key in required_keys:
            if key not in solution:
                raise KeyError(f"Missing key '{key}' in results dictionary")

        m = solution["m"]

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        ax.grid(True)
        ax.set_xlabel("Meridional coordinate [m]")
        ax.set_ylabel("Fanning friction coefficient $c_f$")
        ax.set_xlim(float(m[0]), float(m[-1]))

        ax.plot(m, solution["cf_total"], label="Total $c_f$", color="k", linewidth=2.0)
        ax.plot(
            m,
            solution["cf_wall"],
            label="Wall $c_{f,W}$",
            linestyle="--",
            color="tab:blue",
        )
        ax.plot(
            m,
            solution["cf_diffusion"],
            label="Diffusion $c_{f,D}$",
            linestyle=":",
            color="tab:orange",
        )
        ax.plot(
            m,
            solution["cf_curvature"],
            label="Curvature $c_{f,C}$",
            linestyle="-.",
            color="tab:green",
        )

        ax.legend(loc="upper right")
        fig.tight_layout(pad=1)

        return fig, ax


# -----------------------------------------------------------------------------
# Geometry generator
# -----------------------------------------------------------------------------
def make_vaneless_channel_geometry(geometry: Geometry, tol=1e-6):
    """
    Construct a parametric representation of a 2D axisymmetric vaneless channel
    (e.g., diffuser, inter-blade passage, U-turn, 90-degree bend) using two coupled NURBS
    curves:

      - a channel midline curve defining the (z-r) coordinates in the meridional plane,
      - a channel width curve defining the local width b(s) along the channel.

    The function returns a callable geometric evaluator `geom_handle(s)` that
    provides all local geometric quantities as continuous functions of the
    meridional arclength coordinate `s`.

    ---------------------------------------------------------------------------
    Construction procedure
    ---------------------------------------------------------------------------
    1. The **channel midline** is constructed from four control points:
           (z_in, r_in),
           (z_in + td_in*cos(phi_in), r_in + td_in*sin(phi_in)),
           (z_out - td_out*cos(phi_out), r_out - td_out*sin(phi_out)),
           (z_out, r_out)
       where `td_in` and `td_out` are pseudo-tangential control distances and
       `phi_in`, `phi_out` are inlet and outlet inclination angles.

    2. The midline curve is reparametrized by meridional **arclength** s using
       numerical ODE integration of ds/du = ||C'(u)||, yielding the mapping
       u(s) and total arclength s_total.

    3. The **channel width** curve is constructed in (s, b)
       space, with the first coordinate representing the physical meridional
       distance along the midline, and the second the full channel height b(s).

       For each s, b(s) is evaluated by inverting the NURBS parameterization
       via the mapping u_b(s).

    4. The final evaluator returns geometric quantities such as coordinates,
       normal vector, cross-sectional area, and local slope angle as functions
       of arclength.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    geometry : Geometry
        Equinox module containing geometric parameters:
            {
                "z_in", "z_out" : inlet/outlet axial coordinates [m],
                "r_in", "r_out" : inlet/outlet radii [m],
                "b_in", "b_out" : inlet/outlet channel heights [m],
                "phi_in", "phi_out" : inlet/outlet wall angles [deg],
                "td_in", "td_out" : inlet/outlet tangent distances [m]
            }

    tol : float, optional
        Relative and absolute tolerance for the arclength ODE integration
        (default: 1e-6).

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    geom_handle : callable
        A JAX-compatible function `geom_handle(s)` that returns a dictionary
        of geometric quantities evaluated at arclength positions `s`
        (scalar or array-like):

            {
                "z"        : axial coordinate z(s),
                "r"        : radial coordinate r(s),
                "b"        : local channel height b(s),
                "A"        : cross-sectional area A(s) = 2π·r·b,
                "phi"      : local meridional inclination atan(dr/ds, dz/ds),
                "dzds"     : derivative dz/ds,
                "drds"     : derivative dr/ds,
                "dbds"     : derivative db/ds,
                "dAds"     : derivative dA/ds,
                "z_shroud" : shroud wall axial coordinate z + n_z·b/2,
                "r_shroud" : should wall radial coordinate r + n_r·b/2,
                "z_hub"    : hub wall axial coordinate z - n_z·b/2,
                "r_hub"    : hub wall radial coordinate r - n_r·b/2,
                "s_total"  : total meridional arclength
            }

        The callable is fully compatible with `jax.jit`, `jax.vmap`,
        and automatic differentiation.
    """
    # Extract geometry parameters with short names
    z_in = geometry.z_in
    z_out = geometry.z_out
    r_in = geometry.r_in
    r_out = geometry.r_out
    b_in = geometry.b_in
    b_out = geometry.b_out
    phi_in = geometry.phi_in
    phi_out = geometry.phi_out
    td_in = geometry.td_in
    td_out = geometry.td_out

    # Compute inlet area
    A_in = 2.0 * jnp.pi * r_in * b_in

    # Ensure minimum tangent distances
    td_in = jnp.maximum(1e-3, td_in)
    td_out = jnp.maximum(1e-3, td_out)

    # Construct the channel midline curve control points
    z = jnp.array(
        [
            z_in,
            z_in + td_in * jnp.cos(jnp.deg2rad(phi_in)),
            z_out - td_out * jnp.cos(jnp.deg2rad(phi_out)),
            z_out,
        ]
    )

    r = jnp.array(
        [
            r_in,
            r_in + td_in * jnp.sin(jnp.deg2rad(phi_in)),
            r_out - td_out * jnp.sin(jnp.deg2rad(phi_out)),
            r_out,
        ]
    )

    # Create midline NURBS curve
    P_midline = jnp.vstack([z, r])
    channel_midline = nrb.NurbsCurve(control_points=P_midline, degree=3)

    # Reparametrize midline by arclength
    u_of_s, s_total = channel_midline.reparametrize_by_arclength(tol=tol)
    s_total = s_total*(1.0 - 100.*tol)  # Prevent NURBS extrapolation!!!

    # Construct the channel width curve control points
    P_width = jnp.array(
        [
            [0.00 * s_total, b_in],
            [1.00 * s_total, b_out],
        ]
    ).T
    channel_width = nrb.NurbsCurve(control_points=P_width)

    # Reparametrize channel width by coordinate
    u_of_x = channel_width.reparametrize_by_coordinate(dim=0)

    # Define geometry evaluation function
    def geom_handle(s):
        """Return geometric properties at given arclength s (scalar or array)."""

        # Parameterize back to u-coordinate
        s = jnp.atleast_1d(s)
        u_midline = u_of_s(s)
        u_width = u_of_x(s)

        # Compute channel midline geometry and derivatives
        zr = channel_midline.get_value(u_midline)
        n = channel_midline.get_normal(u_midline)
        curvature = channel_midline.get_curvature(u_midline)
        dzr_du = channel_midline.get_derivative(u_midline, order=1)
        dzr_du_mag = jnp.linalg.norm(dzr_du, axis=0)
        dzr_ds = dzr_du / dzr_du_mag
        z, r = zr[0], zr[1]
        dzds, drds = dzr_ds[0], dzr_ds[1]
        phi = jnp.rad2deg(jnp.arctan2(drds, dzds))

        # Compute channel width and derivative
        _, b = channel_width.get_value(u_width)
        dsb_du = channel_width.get_derivative(u_width, order=1)
        dsdu = dsb_du[0, :]  # first coordinate derivative
        dbdu = dsb_du[1, :]  # second coordinate derivative
        dbds = dbdu / dsdu

        # Compute area and its derivative
        A = 2.0 * jnp.pi * r * b
        dAds = 2.0 * jnp.pi * (drds * b + r * dbds)

        # Compute hub/shroud wall coordinates
        z_hub = z - 0.5 * n[0] * b
        r_hub = r - 0.5 * n[1] * b
        z_shroud = z + 0.5 * n[0] * b
        r_shroud = r + 0.5 * n[1] * b

        # Combine results
        geom_dict = {
            "z": z,
            "r": r,
            "b": b,
            "A": A,
            "phi": phi,
            "dzds": dzds,
            "drds": drds,
            "dbds": dbds,
            "dAds": dAds,
            "z_shroud": z_shroud,
            "r_shroud": r_shroud,
            "z_hub": z_hub,
            "r_hub": r_hub,
            "curvature": curvature,
            "s_total": s_total,
            "m_total": s_total,
            "area_ratio": A / A_in,
            "radius_ratio": r / r_in,
        }

        # If s was scalar, return scalar values instead of 1-element arrays
        geom_dict = {
            k: (v.squeeze() if v.shape == (1,) else v) for k, v in geom_dict.items()
        }
        return geom_dict

    geom_handle.nurbs_midline = channel_midline
    geom_handle.nurbs_width = channel_width

    return jax.jit(geom_handle)


# -----------------------------------------------------------------------------
# Main API to the vaneless diffuser model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def solve_vaneless_channel_model(
    geometry: Geometry,
    operating_conditions: OperatingConditions,
    model_options: ModelOptions,
    solver_options: SolverOptions,
    fluid,
):
    r"""
    Integrate the one-dimensional steady flow equations through a vaneless
    diffuser or return channel using the enthalpy-pressure formulation.

    The model solves the coupled differential system governing the mass, meridional
    momentum, tangential momentum, and total energy balances.

    The solver performs numerical integration from the inlet to the diffuser exit
    using an adaptive high-order Diffrax ODE integrator. At each integration point,
    derived flow quantities (Mach number, density, entropy, geometric parameters, etc.)
    are also evaluated and stored.

    Parameters
    ----------
    params : dict
        Flow and boundary condition parameters, including:
        - ``p_in`` : inlet static pressure [Pa]
        - ``h_in`` : inlet static enthalpy [J/kg]
        - ``v_in`` : inlet velocity magnitude [m/s]
        - ``alpha_in`` : inlet flow angle (radians)
        - ``C_f`` : wall friction coefficient [-]
        - ``q_w`` : wall heat flux [W/m²]
    fluid : jaxprop.Fluid
        Thermodynamic fluid model exposing the method
        ``get_state(INPUT_TYPE, var1, var2)`` for evaluating
        fluid properties such as density, enthalpy, entropy, and speed of sound.
    geom_handle : callable
        Geometry handle returning a dictionary of local geometric quantities
        (radius, width, area, derivatives, inclination) as a function of the
        meridional coordinate. Obtained from function `make_vaneless_channel_geometry()`
    solver_params : object
        Configuration for the Diffrax integrator, including the solver name,
        adjoint type, tolerances (``rtol``, ``atol``), number of save points,
        and maximum iteration count.

    Returns
    -------
    solution : dict of ndarrays
        Integrated solution array containing all computed variables
    """
    # Create the channel geometry
    geom_handle = make_vaneless_channel_geometry(geometry)

    # Compute velocity components
    p_in = operating_conditions.p_in
    h_in = operating_conditions.h_in
    v_in = operating_conditions.v_in
    alpha_in = operating_conditions.alpha_in
    v_m_in = v_in * jnp.cos(jnp.deg2rad(alpha_in))
    v_t_in = v_in * jnp.sin(jnp.deg2rad(alpha_in))

    # Compute inlet stagnation quantities
    state_in = fluid.get_state(jxp.HmassP_INPUTS, h_in, p_in)
    h0_in = h_in + 0.5 * v_in**2
    s0_in = state_in.s
    p0_in = fluid.get_state(jxp.HmassSmass_INPUTS, h0_in, s0_in).p

    # Define the initial condition for integration
    # fmt: off
    y0 = jnp.array([
        v_m_in,      #  0: meridional velocity [m/s]
        v_t_in,      #  1: tangential velocity [m/s]
        h_in,        #  2: static enthalpy [J/kg]
        p_in,        #  3: static pressure [Pa]
        0.0,         #  4: cumulative static-to-total efficiency (η) 
        1.0,         #  5: cumulative kinetic efficiency contribution (η_kinetic) (starts at 1)
        0.0,         #  6: cumulative total loss contribution (η_loss)
        0.0,         #  7: cumulative wall-friction loss contribution (η_loss_wall)
        0.0,         #  8: cumulative diffusion loss contribution (η_loss_diff)
        0.0,         #  9: cumulative curvature loss contribution (η_loss_curv)
        s0_in,       # 10: integrated entropy generation (s_int) [J/kg·K]
        0.0,         # 11: streamline circumferential angle (theta) [rad]
    ])
    # fmt: on

    # Define the upper integration limit
    m_total = geom_handle(s=0.0)["s_total"]

    # Build params dict for internal use
    params = {
        "b_in": geometry.b_in,  # Only pass what you need
        "p_in": p_in,
        "h_in": h_in,
        "v_in": v_in,
        "alpha_in": alpha_in,
        "p0_in": p0_in,
        "h0_in": h0_in,
        "s0_in": s0_in,
        "m_total": m_total,
    }

    # Group the ODE system constant parameters
    args = (params, fluid, geom_handle, model_options)

    # Create and configure the solver
    func_rhs = lambda t, y, args: evaluate_vaneless_channel_ode(t, y, args)[0]
    func_all = lambda t, y, args: evaluate_vaneless_channel_ode(t, y, args)[1]
    solver = jxp.make_diffrax_solver(solver_options.solver_name)
    adjoint = jxp.make_diffrax_adjoint(solver_options.adjoint_name)
    term = dfx.ODETerm(func_rhs)
    ctrl = dfx.PIDController(rtol=solver_options.rtol, atol=solver_options.atol)
    ts = jnp.linspace(0.0, m_total, solver_options.n_points)
    saveat = dfx.SaveAt(ts=ts, dense=False, fn=func_all)

    # Solve the ODE system
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=m_total,
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=ctrl,
        args=args,
        max_steps=solver_options.max_steps,
        adjoint=adjoint,
        throw=solver_options.throw,
    )

    return solution.ys


# -----------------------------------------------------------------------------
# Right hand side of the diffuser ODE system
# -----------------------------------------------------------------------------
def evaluate_vaneless_channel_ode(t, y, args):
    r"""
    Evaluate the right-hand side of the one-dimensional vaneless diffuser model
    using the enthalpy-pressure formulation.

    This function computes the differential evolution of the meridional,
    tangential, thermodynamic, and geometric flow variables along the channel
    arclength coordinate :math:`m` (or :math:`s`). It represents the local
    steady-flow momentum and energy balances in axisymmetric coordinates,
    following the model described in:

        R. Agromayor, B. Müller, and L. O. Nord, “One-dimensional annular diffuser model
        for preliminary turbomachinery design,” International Journal of Turbomachinery,
        Propulsion and Power, vol. 4, no. 3, p. 31, 2019, doi: 10.3390/ijtpp4030031.

    The governing equations are identical to those in the cited paper but
    reformulated in terms of the thermodynamic variables :math:`(h, p)` instead
    of :math:`(\rho, p)`.

    The system can be written compactly as:

    .. math::

        A\,\frac{dU}{dm} = S

    where

    .. math::
        A =
        \begin{bmatrix}
        \tfrac{\rho}{v_m} & 0 & -\,\tfrac{\rho G}{a^{2}} & \tfrac{1+G}{a^{2}} \\
        \rho v_m & 0 & 0 & 1 \\
        0 & \rho v_m & 0 & 0 \\
        v_m & v_\theta & 1 & 0
        \end{bmatrix},
        \quad
        U =
        \begin{bmatrix}
        v_m \\ v_\theta \\ h \\ p
        \end{bmatrix},
        \quad
        S =
        \begin{bmatrix}
        -\,\tfrac{\rho}{A}\tfrac{dA}{dm} \\[0.6em]
        \tfrac{\rho v_\theta^2}{r}\sin\phi - \tfrac{2\tau_w}{b}\cos\alpha \\[0.6em]
        -\,\tfrac{\rho v_\theta v_m}{r}\sin\phi - \tfrac{2\tau_w}{b}\sin\alpha \\[0.6em]
        \tfrac{2\dot q_w}{b}
        \end{bmatrix}.

    Here,
    - :math:`v_m, v_\theta` are the meridional and tangential velocities,
    - :math:`h, p` are the static enthalpy and pressure,
    - :math:`A = 2\pi r b` is the meridional flow area,
    - :math:`\tau_w` is the wall shear stress,
    - :math:`\dot q_w` is the wall heat flux,
    - :math:`a` is the local speed of sound,
    - :math:`G` is the Grüneisen parameter

    The dependent variables evolve according to:

    .. math::
        \frac{dU}{dm}
        =
        A^{-1} S,

    which is solved numerically at each spatial location.

    The last two terms of the returned state vector represent:
    - :math:`\dot{s}_{gen}` — entropy generation rate per unit mass flow,
    - :math:`\dot{\theta}` — circumferential wrapping rate of the streamline.

    Parameters
    ----------
    t : float
        Meridional coordinate :math:`m` or :math:`s` (integration variable).
    y : array_like
        State vector `[v_m, v_t, h, p, s_gen, theta]`.
    args : tuple
        Model parameters `(C_f, q_w, p0_in, p_in, fluid, geom_handle)`.

    Returns
    -------
    rhs : ndarray
        Time derivative of the state vector
        `[dv_m/dm, dv_t/dm, dh/dm, dp/dm, ds_gen/dm, dtheta/dm]`.
    out : dict
        Dictionary containing derived thermodynamic, kinematic, and geometric
        quantities at the current meridional location, including Mach numbers,
        entropy, total enthalpy, static-to-total pressure ratio, and local
        geometry descriptors such as area, radius, and inclination angle.
    """

    # Rename from ODE terminology to physical variables
    params, fluid, geom_handle, model_options = args
    m_total = params["m_total"]
    # m_coord = jnp.minimum(t, m_total*0.9999)  # Prevent NURBS extrapolation
    m_coord = t
    (
        v_m,
        v_t,
        h,
        p,
        eta,
        efficiency_kinetic,
        efficiency_loss,
        efficiency_loss_wall,
        efficiency_loss_diff,
        efficiency_loss_curv,
        s_int,
        theta,
    ) = y

    # # --- Debug print section ---
    # jax.debug.print(
    #     "t = {t:.4e}, m = {m:.4e}, m_tot = {m_tot:.4e}, "
    #     "v_m = {v_m:.4e}, v_t = {v_t:.4e}, h = {h:.4e}, p = {p:.4e}, eta = {eta:.4e}",
    #     t=t,
    #     m=m_coord,
    #     m_tot= m_total,
    #     v_m=v_m,
    #     v_t=v_t,
    #     h=h,
    #     p=p,
    #     eta=eta,
    # )

    # Calculate velocity magnitude and direction
    v = jnp.sqrt(v_t**2 + v_m**2)
    alpha = jnp.rad2deg(jnp.arctan2(v_t, v_m))

    # Evaluate geometry at local meridional coordinate
    geom = geom_handle(m_coord)
    r = geom["r"]
    b = geom["b"]
    A = geom["A"]
    dAds = geom["dAds"]
    phi = geom["phi"]
    curvature = geom["curvature"]

    # Calculate static thermodynamic state
    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    a = state["a"]
    d = state["d"]
    T = state["T"]
    s = state["s"]
    cp = state["cp"]
    k = state["conductivity"]
    mu = state["viscosity"]
    G = state["gruneisen"]

    # Calculate stagnation thermodynamic state
    h0 = h + 0.5 * v**2
    state0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    p0 = state0.p
    d0 = state0.d
    T0 = state0.T

    # Calculate isentropic thermodynamic state
    state_s = fluid.get_state(jxp.PSmass_INPUTS, p, params["s0_in"])
    h_s = state_s.h

    # Compute Reynolds and Prandtl numbers
    D_h = 2.0 * b
    Re = jnp.maximum(d * v * D_h / mu, 1.0)
    Pr = jnp.maximum(cp * mu / k, 1e-6)  # cp, k from fluid if available

    # Compute skin friction according to Aungier loss model
    cf_wall, cf_diff, cf_curv, E = model_options.friction.get_cf_components(
        m_coord, m_total, b, params["b_in"], A, dAds, curvature, alpha, params["alpha_in"], Re
    )

    # Original Augier formulation with an asymmetrical loss distribution
    tau_m = 0.5 * d * v**2 * (cf_wall * jnp.cos(jnp.deg2rad(alpha)) + cf_diff + cf_curv)
    tau_t = 0.5 * d * v**2 * (cf_wall * jnp.sin(jnp.deg2rad(alpha)))

    # # Alternative formulation with a symmetrical loss distribution
    # tau_m = 0.5 * d * v**2 * (cf_wall  + cf_diff + cf_curv) * jnp.cos(alpha)
    # tau_t = 0.5 * d * v**2 * (cf_wall  + cf_diff + cf_curv) * jnp.sin(alpha)

    # Compute heat transfer at the walls
    q_w, htc = model_options.heat_transfer.compute_heat_transfer(
        T0_fluid=T0,
        rho=d,
        v=v,
        cp=state["cp"],
        k=state["conductivity"],
        mu=mu,
        Cf=cf_wall,
        Dh=D_h,
    )
    
    # Compute coefficient matrix
    M = jnp.asarray(
        [
            [d / v_m, 0.0, -G / a**2, (1.0 + G) / a**2],
            [d * v_m, 0.0, 0.0, 1.0],
            [0.0, d * v_m, 0.0, 0.0],
            [v_m, v_t, 1.0, 0.0],
        ]
    )

    # Compute source term
    sin_phi = jnp.sin(jnp.deg2rad(phi))
    S = jnp.asarray(
        [
            -d * (dAds / A),
            d * v_t**2 / r * sin_phi - 2 * tau_m / b,
            -d * v_t * v_m / r * sin_phi - 2 * tau_t / b,
            (2 / b) * q_w,
        ]
    )

    # Compute solution
    dv_m, dv_t, dh, dp = jnp.linalg.solve(M, S)

    # Compute entropy generation check
    s_gen = 2.0 / b * (tau_m * v_m + tau_t * v_t)
    ds_int = s_gen / (d * v_m * T)

    # Compute the streamline wrapping angle derivative
    dtheta = (v_t / v_m) / r

    # Kinetic energy analysis
    h_in = params["h_in"]
    h0_in = params["h0_in"]
    dEff, dEff_kinetic, dEff_loss, (dEff_loss_wall, dEff_loss_diff, dEff_loss_curv) = (
        compute_efficiency_derivatives(
            dh, dp, d, alpha, h0_in, h_in, cf_wall, cf_diff, cf_curv
        )
    )

    # Dynamic pressure analysis (Cp + KE + Y = 1)
    p_in = params["p_in"]
    p0_in = params["p0_in"]
    Cp = (p - p_in) / (p0_in - p_in)
    KE = (p0 - p) / (p0_in - p_in)
    Y = (p0_in - p0) / (p0_in - p_in)

    # Prepare right hand side of the ODE system
    rhs_core = jnp.array([dv_m, dv_t, dh, dp])
    rhs_eff = jnp.array(
        [dEff, dEff_kinetic, dEff_loss, dEff_loss_wall, dEff_loss_diff, dEff_loss_curv]
    )
    rhs_extra = jnp.asarray([ds_int, dtheta])
    rhs = jnp.concatenate([rhs_core, rhs_eff, rhs_extra])

    # Store esults
    out = {
        **geom,
        "m": t,
        "v_t": v_t,
        "v_m": v_m,
        "v": v,
        "alpha": alpha,
        "Ma": v / a,
        "Ma_m": v_m / a,
        "Ma_t": v_t / a,
        "Re": Re,
        "mu": mu,
        "p": p,
        "T": T,
        "d": d,
        "s": s,
        "s_int": s_int,
        "h": h,
        "h_s": h_s,
        "h0": h0,
        "p0": p0,
        "T0": T0,
        "d0": d0,
        "theta": theta,
        "Cp": Cp,
        "KE": KE,
        "Y": Y,
        "efficiency": eta,
        "efficiency_kinetic": efficiency_kinetic,
        "efficiency_loss": efficiency_loss,
        "efficiency_loss_wall": efficiency_loss_wall,
        "efficiency_loss_diffusion": efficiency_loss_diff,
        "efficiency_loss_curvature": efficiency_loss_curv,
        "cf_total": cf_wall + cf_diff + cf_curv,
        "cf_wall": cf_wall,
        "cf_diffusion": cf_diff,
        "cf_curvature": cf_curv,
        "E_diffusion": E,
        "q_w": q_w,
        "htc": htc,
    }

    return rhs, out


def compute_efficiency_derivatives(
    dh, dp, d, alpha, h0_in, h_in, cf_wall, cf_diff, cf_curv
):
    """Return incremental efficiency components (total, kinetic, and losses)."""

    # Isentropic enthalpy change
    dh_s = dp / d

    # Overall efficiency derivatives
    dEff_total = dh_s / (h0_in - h_in)
    dEff_kinetic = -dh / (h0_in - h_in)
    dEff_loss = (dh - dh_s) / (h0_in - h_in)

    # Fractional decomposition
    # TODO: Loss split is currently based on the Aungier's formulation
    loss_total = cf_wall + jnp.cos(jnp.deg2rad(alpha)) * (cf_diff + cf_curv) + 1e-12
    weights = jnp.array(
        [
            cf_wall / loss_total,
            jnp.cos(jnp.deg2rad(alpha)) * cf_diff / loss_total,
            jnp.cos(jnp.deg2rad(alpha)) * cf_curv / loss_total,
        ]
    )

    dEff_loss_split = weights * dEff_loss
    return dEff_total, dEff_kinetic, dEff_loss, dEff_loss_split


def convert_stagnation_to_static(p0, T0, Ma, fluid):
    r"""
    Compute the static thermodynamic state corresponding to a given Mach number
    assuming an isentropic process from a known stagnation (total) state.

    The function determines the static pressure :math:`p`, static enthalpy
    :math:`h`, and flow velocity :math:`v` such that both the total energy and
    entropy are conserved between the stagnation and static states.

    The governing relations are:

    .. math::

        \begin{aligned}
        h_0 &= h + \tfrac{1}{2}v^2, \\
        s_0 &= s,
        \end{aligned}

    where :math:`h_0, s_0` are the stagnation (total) enthalpy and entropy,
    and :math:`v = a\,M_a` is the local flow velocity expressed in terms of the
    Mach number :math:`M_a` and the local speed of sound :math:`a`.

    By substituting :math:`v = a\,M_a`, the system becomes two nonlinear equations
    in :math:`(p, T)`:

    .. math::

        \begin{cases}
        f_1(p, T) = h_0 - \left[h(p, T) + \tfrac{1}{2}(a(p, T)\,M_a)^2\right] = 0, \\[0.5em]
        f_2(p, T) = s(p, T) - s_0 = 0,
        \end{cases}

    which are solved simultaneously using a Newton root-finding method.

    Parameters
    ----------
    p0 : float
        Stagnation (total) pressure [Pa].
    T0 : float
        Stagnation (total) temperature [K].
    Ma : float
        Local Mach number.
    fluid : object
        Thermodynamic fluid model providing the method
        ``get_state(INPUT_TYPE, p, T)`` returning properties such as
        speed of sound ``a``, enthalpy ``h``, and entropy ``s``.

    Returns
    -------
    p : float
        Static pressure [Pa].
    h : float
        Static enthalpy [J/kg].
    v : float
        Flow velocity [m/s].
    """
    # --- Reference stagnation state ---
    st0 = fluid.get_state(jxp.PT_INPUTS, p0, T0)
    h0, s0 = st0["h"], st0["s"]

    # --- Define residual system: f(p, h) = [f1, f2] ---
    def residual(x, _):
        p, T = x
        st = fluid.get_state(jxp.PT_INPUTS, p, T)
        a, s, h = st["a"], st["s"], st["h"]
        f1 = h0 - (h + 0.5 * (a * Ma) ** 2)  # energy balance
        f2 = s - s0  # isentropic condition
        return jnp.array([f1, f2])

    # --- Initial guess ---
    initial_guess = jnp.array([p0, T0])

    # --- Solve ---
    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    sol = optx.root_find(residual, solver, y0=initial_guess)
    p, T = sol.value

    # --- Evaluate final state ---
    st = fluid.get_state(jxp.PT_INPUTS, p, T)
    a = st["a"]
    h = st["h"]
    v = a * Ma

    return p, h, v


def get_analytical_Cp(alpha_in, AR, RR):
    r"""
    Compare the numerical diffuser solution against the analytical expression
    for the incompressible, inviscid pressure recovery coefficient.

    The analytical model assumes steady, axisymmetric, incompressible flow
    with negligible wall friction and no energy losses. Under these hypotheses,
    Bernoulli's equation applies along a streamline, and the pressure recovery
    coefficient between inlet (1) and outlet (2) can be written as

    $$
    C_p = 1
    - \cos^2(\alpha_1) \left( \frac{A_1}{A_2} \right)^2
    - \sin^2(\alpha_1) \left( \frac{r_1}{r_2} \right)^2
    $$

    where

    - $\alpha_1$ is the inlet flow angle (measured from the meridional direction),
    - $r_1$ and $r_2$ are the inlet and outlet radii,
    - $A_1$ and $A_2$ are the corresponding meridional flow areas.

    This expression represents the ideal pressure recovery in an annular vaneless diffuser.
    """
    alpha_rad = jnp.deg2rad(alpha_in)
    Cp = 1.0 - (jnp.cos(alpha_rad) ** 2) / AR**2 - (jnp.sin(alpha_rad) ** 2) / RR**2
    return Cp


# def evaluate_solution_dense(sol, n_points: int = 100):
#     """
#     Evaluate a Diffrax solution object at evenly spaced points between t0 and t1,
#     using vmapped interpolation.

#     Parameters
#     ----------
#     sol : diffrax.Solution
#         Solution object returned by `diffrax.diffeqsolve`.
#     n_points : int, optional
#         Number of evaluation points (default: 200).

#     Returns
#     -------
#     t_eval : jnp.ndarray
#         1D array of evaluation times (shape: [n_points]).
#     y_eval : jnp.ndarray
#         2D array of interpolated state values (shape: [n_points, n_state]).
#     """
#     t_eval = jnp.linspace(sol.t0, sol.t1, n_points)
#     y_eval = jax.vmap(sol.evaluate)(t_eval)
#     return t_eval, y_eval
