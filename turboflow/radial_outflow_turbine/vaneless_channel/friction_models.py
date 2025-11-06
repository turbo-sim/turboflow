import jax
import jax.numpy as jnp
import equinox as eqx


from jaxtyping import Array, Float, Scalar


# -------------------------
# Define Equinox modules
# -------------------------
class FrictionModel(eqx.Module):
    """Abstract base for wall friction and secondary loss models."""

    def get_cf_components(
        self,
        Re: Float[Array, ""],
        b: Float[Array, ""],
        A: Float[Array, ""],
        dA_dm: Float[Array, ""],
        alpha_in: Float[Array, ""],
        b_in: Float[Array, ""],
        L_total: Float[Array, ""],
        curvature: Float[Array, ""],
        alpha: Float[Array, ""],
    ):
        """Return cf_wall, cf_diffusion, cf_curvature, E"""
        raise NotImplementedError


class ZeroFriction(FrictionModel):
    """Idealized frictionless case."""
    def get_cf_components(self, *args, **kwargs):
        cf_wall = jnp.array(0.0)
        cf_diff = jnp.array(0.0)
        cf_curv = jnp.array(0.0)
        E = jnp.array(1.0)
        return cf_wall, cf_diff, cf_curv, E


class ConstantFriction(FrictionModel):
    """Constant skin friction coefficient."""
    Cf: Float[Array, ""]
    def get_cf_components(self, *args, **kwargs):
        cf_wall = self.Cf
        cf_diff = jnp.array(0.0)
        cf_curv = jnp.array(0.0)
        E = jnp.array(1.0)
        return cf_wall, cf_diff, cf_curv, E


class AungierFriction(FrictionModel):
    """Aungier (1993) empirical wall + diffusion + curvature losses."""

    roughness: Float[Array, ""]
    Re_transition: Float[Array, ""]
    Re_width: Float[Array, ""]

    def get_cf_components(
        self,
        Re: Float[Array, ""],
        b: Float[Array, ""],
        A: Float[Array, ""],
        dA_dm: Float[Array, ""],
        alpha_in: Float[Array, ""],
        b_in: Float[Array, ""],
        L_total: Float[Array, ""],
        curvature: Float[Array, ""],
        alpha: Float[Array, ""],
    ):
        # --- Call your existing helper functions directly ---
        D_h = 2.0 * b
        cf_wall = get_cf_wall(Re, self.roughness, D_h)
        cf_diff, E = get_cf_diffusion(b, A, dA_dm, alpha_in, b_in, L_total)
        cf_curv = get_cf_curvature(b, curvature, alpha)
        return cf_wall, cf_diff, cf_curv, E


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def make_friction_model(cfg: dict) -> FrictionModel:
    """Factory for friction submodels based on configuration dict.

    Supported model types:
        - 'zero_friction'             : Frictionless case (Cf = 0)
        - 'constant_friction_factor'  : Constant wall friction coefficient
        - 'aungier'                   : Aungier (1993) wall + diffusion + curvature losses
    """
    valid_models = ("zero_friction", "constant_friction_factor", "aungier")

    # Require explicit model type
    if "type" not in cfg:
        raise ValueError(
            "Missing required key 'type' in friction model configuration.\n"
            f"Valid options are: {', '.join(valid_models)}."
        )

    model_type = str(cfg["type"]).lower()

    def validate_keys(required_keys):
        """Check for missing or extra keys in cfg."""
        allowed_keys = set(required_keys) | {"type"}
        provided_keys = set(cfg.keys())

        missing = allowed_keys - provided_keys
        extra = provided_keys - allowed_keys

        if missing or extra:
            msg = [f"Issues found in configuration for '{model_type}':"]
            if missing:
                msg.append(f"  - Missing: {', '.join(sorted(missing))}")
            if extra:
                msg.append(f"  - Unexpected: {', '.join(sorted(extra))}")
            msg.append(f"  - Allowed keys: {sorted(allowed_keys)}")
            msg.append(f"  - Provided keys: {sorted(provided_keys)}")
            raise ValueError("\n".join(msg))

    # Model selection
    if model_type == "zero_friction":
        validate_keys([])  # only 'type' allowed
        return ZeroFriction()

    elif model_type == "constant_friction_factor":
        validate_keys(["Cf"])
        return ConstantFriction(Cf=jnp.array(cfg["Cf"]))

    elif model_type == "aungier":
        validate_keys(["roughness", "Re_transition", "Re_width"])
        return AungierFriction(
            roughness=jnp.array(cfg["roughness"]),
            Re_transition=jnp.array(cfg["Re_transition"]),
            Re_width=jnp.array(cfg["Re_width"]),
        )

    else:
        valid_str = ", ".join(f"'{m}'" for m in valid_models)
        raise ValueError(
            f"Unknown friction model type '{model_type}'.\n"
            f"Valid options are: {valid_str}."
        )
    

# -------------------------
# Define core functions
# -------------------------
def get_cf_wall(Re, roughness, diameter):
    """Compute the Fanning friction factor with a smooth laminar-turbulent transition.

    The function returns a continuous Fanning friction factor over all Reynolds numbers.
    In the laminar regime, it uses the analytical relation `Cf = 16 / Re`. In the turbulent
    regime, it applies the Haaland correlation, which provides an explicit and accurate
    approximation to the Colebrook-White equation for the Darcy friction factor.
    The result is then converted to the Fanning friction factor using `Cf = f_D / 4`.

    A smooth transition between laminar and turbulent regimes is achieved using a
    hyperbolic tangent blending function centered around `Re ≈ 2300` with a
    configurable transition width.

    Parameters
    ----------
    Re : float or array_like
        Reynolds number (dimensionless).
    roughness : float
        Absolute roughness of the pipe's inner surface (m).
    diameter : float
        Inner diameter of the pipe (m).

    Returns
    -------
    Cf : float or array_like
        Fanning friction factor (dimensionless), smoothly varying across flow regimes.
    """
    # Laminar friction factor (Cf = 16/Re for Fanning)
    Cf_laminar = 16.0 / jnp.maximum(Re, 1.0)

    # Turbulent friction factor (Haaland correlation)
    term = 6.9 / jnp.maximum(Re, 1.0) + (roughness / diameter / 3.7) ** 1.11
    f_D = (-1.8 * jnp.log10(term)) ** -2  # Darcy friction factor
    Cf_turbulent = f_D / 4.0  # Convert to Fanning friction factor

    # Smooth blending using tanh (transition centered at Re=2300)
    transition_width = 500.0  # Controls smoothness (smaller = sharper transition)
    Re_transition = 2300.0 + transition_width  # Typical transition Reynolds number
    blend = 0.5 * (1.0 + jnp.tanh((Re - Re_transition) / transition_width))

    # Blend between laminar and turbulent
    return Cf_laminar * (1.0 - blend) + Cf_turbulent * blend


def get_cf_diffusion(b, A, dA_dm, alpha_in, b_in, L_total):
    """Diffusion loss coefficient cf,D following Aungier (1993)."""
    D = (b / A) * dA_dm
    D_m = 0.4 * jnp.cos(jnp.deg2rad(alpha_in)) * (b_in / L_total) ** 0.35

    def E_piecewise(D, D_m):
        return jnp.where(
            D <= 0,
            1.0,
            jnp.where(
                D < D_m,
                1.0 - 0.2 * (D / D_m) ** 2,
                0.8 * jnp.sqrt(D_m / D),
            ),
        )

    E = E_piecewise(D, D_m)
    return D * (1.0 - E), E


def get_cf_curvature(b, curvature, alpha):
    """Curvature loss coefficient cf,C following Aungier (1993)."""
    return (b * curvature * jnp.cos(jnp.deg2rad(alpha))) / 26.0


