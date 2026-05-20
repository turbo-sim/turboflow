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
        s_exit,
        mass_flow_exit,
        v_out_is,
        mass_flow_ref,
        alpha_target_deg=None,
    ):
        """
        Propagate exit of i to inlet of i+1 (interspace), algebraic model:

        Inputs are the exit conditions of the upstream row + local geometry.
        Returns:
          h0_in, s_in, alpha_in [deg], v_in
        """

        alpha_target = jnp.asarray(
            jnp.nan if alpha_target_deg is None else alpha_target_deg,
            dtype=jnp.float64,
        )

        h0_in, h_in, alpha_in, v_in, v_m_in = _evaluate_interspace_core(
            h0_exit=h0_exit,
            v_m_exit=v_m_exit,
            v_t_exit=v_t_exit,
            rho_exit=rho_exit,
            radius_exit=radius_exit,
            area_exit=area_exit,
            blockage_exit=blockage_exit,
            radius_inlet=radius_inlet,
            area_inlet=area_inlet,
            v_out_is=v_out_is,
            alpha_target_deg=alpha_target,   # <- NEW
        )

        # Entropy from fluid model (not jitted)
        s_in = s_exit

        st_in = self.fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)
        rho_in = st_in["d"]

        # Mass residual (same normalization philosophy as blade rows)
        m_in = rho_in * v_m_in * area_inlet * (1.0 - blockage_exit)
        mass_res = (m_in - mass_flow_exit) / jnp.maximum(mass_flow_ref, 1e-12)

        return h0_in, s_in, alpha_in, v_in, mass_res

    # ------------------------------------------------------------------
    # Initial guess interface (no unknowns from interspace)
    # ------------------------------------------------------------------
    # def build_initial_guess(self, inlet_state: Dict[str, Any], omega,
    #     row_index: int,):
    #     """
    #     Interspace contributes no variables to the nonlinear solver.
    #     Returning {} keeps compute_single_operation_point generic.
    #     """
    #     return {}, inlet_state
    
    def build_initial_guess(self, inlet_state: Dict[str, Any], omega, row_index: int):
        key = f"v_out_is_{self.name}"   # physical velocity [m/s]
        v_guess = jnp.asarray(inlet_state["v_in"], dtype=jnp.float64)
        return {key: v_guess}, inlet_state


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
    v_out_is,
    alpha_target_deg,   # <- NEW
):
    """JIT-friendly numeric core for interspace mapping (no fluid calls)."""
    # h0_in = h0_exit
    # v_t_in = v_t_exit * radius_exit / radius_inlet

    # # Solver DOF: interspace exit speed / next-row inlet speed
    # v_in = jnp.maximum(v_out_is, 1e-6)

    # # Keep velocity triangle physical
    # v_m_sq = jnp.maximum(v_in**2 - v_t_in**2, 1e-12)
    # v_m_in = jnp.sqrt(v_m_sq)

    # alpha_in = jnp.degrees(jnp.arctan2(v_t_in, v_m_in))
    

    h0_in = h0_exit

    # Default: carry swirl from upstream row
    v_t_swirl = v_t_exit * radius_exit / radius_inlet

    v_in = jnp.maximum(v_out_is, 1e-6)
    v_m_swirl = jnp.sqrt(jnp.maximum(v_in**2 - v_t_swirl**2, 1e-12))

    # Optional: force target inlet angle (e.g. stator_5 alignment)
    use_target = jnp.isfinite(alpha_target_deg)
    alpha_t = jnp.clip(alpha_target_deg, -89.0, 89.0)
    v_m_target = jnp.maximum(v_in * jnp.cos(jnp.deg2rad(alpha_t)), 1e-6)
    v_t_target = v_in * jnp.sin(jnp.deg2rad(alpha_t))

    v_m_in = jnp.where(use_target, v_m_target, v_m_swirl)
    v_t_in = jnp.where(use_target, v_t_target, v_t_swirl)

    # b = 0.3  # blending factor between swirl and target angle (tune 0.0 to 1.0)
    # alpha_cmd = jnp.nan_to_num(alpha_target_deg, nan=0.0)
    # alpha_cmd = jnp.clip(alpha_cmd, -89.0, 89.0)

    # v_m_target = jnp.maximum(v_in * jnp.cos(jnp.deg2rad(alpha_cmd)), 1e-6)
    # v_t_target = v_in * jnp.sin(jnp.deg2rad(alpha_cmd))

    # v_m_in = (1.0 - b) * v_m_swirl + b * v_m_target
    # v_t_in = (1.0 - b) * v_t_swirl + b * v_t_target

    alpha_in = jnp.degrees(jnp.arctan2(v_t_in, v_m_in))
    h_in = h0_in - 0.5 * v_in**2

    return h0_in, h_in, alpha_in, v_in, v_m_in
