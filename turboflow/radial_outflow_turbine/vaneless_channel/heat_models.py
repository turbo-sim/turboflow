import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------
class HeatTransferModel(eqx.Module):
    """Abstract base class for wall heat transfer models."""

    def compute_heat_transfer(
        self,
        T0_fluid: Float[Array, ""],
        rho: Float[Array, ""],
        v: Float[Array, ""],
        cp: Float[Array, ""],
        k: Float[Array, ""],
        mu: Float[Array, ""],
        Cf: Float[Array, ""],
        Dh: Float[Array, ""],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """
        Compute wall heat flux and heat transfer coefficient.

        Parameters
        ----------
        T_wall : Wall temperature [K]
        T_fluid : Local stagnation fluid temperature [K]
        rho : Fluid density [kg/m³]
        v : Local velocity magnitude [m/s]
        cp : Specific heat capacity [J/(kg·K)]
        k : Thermal conductivity [W/(m·K)]
        mu : Dynamic viscosity [Pa·s]
        Cf : Local skin friction coefficient [-]
        Dh : Hydraulic diameter [m]

        Returns
        -------
        q_w : Wall heat flux [W/m²]
        htc : Heat transfer coefficient [W/(m²·K)]
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
# Adiabatic (zero heat flux)
# ---------------------------------------------------------------------
class Adiabatic(HeatTransferModel):
    """Idealized adiabatic wall (no heat transfer)."""
    def compute_heat_transfer(self, *args, **kwargs):
        return jnp.array(0.0), jnp.array(0.0)

# ---------------------------------------------------------------------
# Reynolds analogy model
# ---------------------------------------------------------------------
class ReynoldsAnalogy(HeatTransferModel):
    """Reynolds analogy (Stanitz) heat transfer model."""
    T_wall: Float[Array, ""]
    def compute_heat_transfer(self, T0_fluid, rho, v, cp, k, mu, Cf, Dh):
        Re = jnp.maximum(rho * v * Dh / mu, 1.0)
        Pr = jnp.maximum(cp * mu / k, 1e-6)
        Nu = 0.5 * Cf * Re * Pr
        htc = Nu * k / Dh
        q_w = htc * (self.T_wall - T0_fluid)
        return q_w, htc

# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def make_heat_model(cfg: dict) -> HeatTransferModel:
    """Factory for heat transfer submodels based on configuration dict.

    Supported model types:
        - 'adiabatic' : Idealized no-heat-transfer wall (htc = 0)
        # future:
        # - 'reynolds_analogy' : Reynolds analogy model
        # - 'dittus_boelter'   : Empirical turbulent correlation
    """
    valid_models = ("adiabatic", "reynolds_analogy", )

    # Require explicit model type
    if "type" not in cfg:
        raise ValueError(
            "Missing required key 'type' in heat transfer model configuration.\n"
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
    if model_type == "adiabatic":
        validate_keys([])  # only "type" allowed
        return Adiabatic()

    elif model_type == "reynolds_analogy":
        validate_keys(["T_wall"])
        return ReynoldsAnalogy(cfg["T_wall"])

    else:
        raise ValueError(
            f"Unknown heat transfer model type '{model_type}'. "
            f"Valid options: {', '.join(valid_models)}"
        )