from typing import Any, Dict

import jax
import jax.numpy as jnp
import equinox as eqx
import jaxprop as jxp
from jaxtyping import Array, Float


class Geometry(eqx.Module):
    z_in:  Float[Array, ""]
    z_out: Float[Array, ""]
    radius_mean_in: Float[Array, ""]
    radius_mean_out: Float[Array, ""]
    b_in:  Float[Array, ""]
    b_out: Float[Array, ""]
    phi_in:  Float[Array, ""]
    phi_out: Float[Array, ""]
    td_in:  Float[Array, ""]
    td_out: Float[Array, ""]


class Interspace(eqx.Module):
    """
    Simple algebraic interspace model between blade rows.

    - Owns a Geometry object (for consistency / plotting later).
    - Provides a cheap evaluate(...) that maps exit of row i to inlet of row i+1.
    - Does not introduce any solver unknowns (build_initial_guess returns {}).
    """

    name: str = eqx.field(static=True)
    geometry: Geometry
    fluid: Any

    # ------------------------------------------------------------------
    # Construction from YAML
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, config: Dict[str, Any], fluid: Any) -> "Interspace":
        """
        config example:

          name: interspace_1
          geometry:
            r_in:  0.1016
            r_out: 0.1016
            z_in: 0.0
            z_out: 0.05
            b_in: 0.03363
            b_out: 0.03363
            phi_in: 0.0
            phi_out: 0.0
            td_in: 0.01
            td_out: 0.01
        """
        # Helper to convert dict values to JAX arrays
        def to_jax_dict(d: dict) -> dict:
            return {
                k: jnp.array(v) if isinstance(v, (int, float)) else v
                for k, v in d.items()
            }
        
        return cls(
            name=config.get("name", "interspace"),
            geometry=Geometry(**to_jax_dict(config["geometry"])),
            fluid=fluid,
        )

    # ------------------------------------------------------------------
    # Algebraic mapping (exit row i → inlet row i+1)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        h0_exit,
        v_m_exit,
        v_t_exit,
        rho_exit,
        radius_exit,
        area_exit,
        blockage_exit,
        radius_inlet,
        area_inlet,
    ):
        """
        Propagate exit of i to inlet of i+1 (interspace), algebraic model:

        Inputs are the exit conditions of the upstream row + local geometry.
        Returns:
          h0_in, s_in, alpha_in [deg], v_in
        """

        h0_in, h_in, alpha_in, v_in, rho_in = _evaluate_interspace_core(
            h0_exit,
            v_m_exit,
            v_t_exit,
            rho_exit,
            radius_exit,
            area_exit,
            blockage_exit,
            radius_inlet,
            area_inlet,
        )

        # Entropy from fluid model (not jitted)
        st = self.fluid.get_state(jxp.DmassHmass_INPUTS, rho_in, h_in)
        s_in = st["s"]

        return h0_in, s_in, alpha_in, v_in

    # ------------------------------------------------------------------
    # Initial guess interface (no unknowns from interspace)
    # ------------------------------------------------------------------
    def build_initial_guess(self, inlet_state: Dict[str, Any], omega,
        row_index: int,):
        """
        Interspace contributes no variables to the nonlinear solver.
        Returning {} keeps compute_single_operation_point generic.
        """
        return {}, inlet_state


@jax.jit
def _evaluate_interspace_core(
    h0_exit,
    v_m_exit,
    v_t_exit,
    rho_exit,
    radius_exit,
    area_exit,
    blockage_exit,
    radius_inlet,
    area_inlet,
):
    """JIT-friendly numeric core for interspace mapping (no fluid calls)."""
    h0_in = h0_exit
    v_t_in = v_t_exit * radius_exit / radius_inlet
    v_m_in = v_m_exit * area_exit / area_inlet * (1.0 - blockage_exit)
    v_in = jnp.sqrt(v_t_in**2 + v_m_in**2)
    alpha_in = jnp.degrees(jnp.arctan2(v_t_in, v_m_in))
    h_in = h0_in - 0.5 * v_in**2
    rho_in = rho_exit
    return h0_in, h_in, alpha_in, v_in, rho_in
