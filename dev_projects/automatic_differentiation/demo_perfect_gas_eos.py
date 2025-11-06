"""
Demo: perfect gas API usage with air

What this shows:
- how to obtain constants (precomputed vs computed)
- how to evaluate states from different input pairs
- how to evaluate transport properties as vectorized functions of temperature
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jaxprop as jxp
import jaxprop.perfect_gas as pg

jxp.set_plot_options()

# --------------------------- get constants -------------------------- #
fluid = jxp.FluidPerfectGas("air", 298.15, 101_325.0)
print(f"\nPerfect-gas constants:")
print(fluid.constants)


# -------------------------- basic evaluations ----------------------- #

# Property calculation using (p, T)
P0 = 101_325.0  # Pa
T0 = 300.0      # K
state_PT = fluid.get_state(jxp.PT_INPUTS, P0, T0)
print(f"\nState from PT (p = {P0:.0f} Pa, T = {T0:.2f} K)")
print(state_PT)

# Property calculation using (h, s)
H = state_PT["h"]
S = state_PT["s"]
state_hs = fluid.get_state(jxp.HmassSmass_INPUTS, H, S)
print("\nState from HmassSmass (h, s)")
print(state_hs)

# Property calculation using (h, p)
state_hP = fluid.get_state(jxp.HmassP_INPUTS, H, P0)
print("\nState from HmassP (h, p)")
print(state_hP)

# Property calculation using (p, s)
state_Ps = fluid.get_state(jxp.PSmass_INPUTS, P0, S)
print("\nState from PSmass (p, s)")
print(state_Ps)

# Property calculation using (rho, h)
rho0 = state_PT["rho"]
state_rhoh = fluid.get_state(jxp.DmassHmass_INPUTS, rho0, H)
print("\nState from DmassHmass (rho, h)")
print(state_rhoh)

# ------------------- vectorized calculations and plotting ---------------- #

# Compute viscosity over the temperature range
T_range = np.linspace(300.0, 1000.0, 200)
mu_vals = pg.viscosity_from_T(T_range, fluid.constants)

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(False)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Dynamic viscosity [Pa·s]")
ax.set_xlim([T_range.min(), T_range.max()])

# Plot viscosity
ax.plot(T_range, mu_vals)
plt.tight_layout(pad=1)

# Show plot if not disabled
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
