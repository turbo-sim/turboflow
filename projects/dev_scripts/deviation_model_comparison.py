import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml

# Define case parameters
radius_curvature = np.inf
pitch = 1.00
opening = 0.40
Ma_crit = 1.00
Ma_exit = np.linspace(0.00, 1.00, 200)

# Compute exit flow angles
beta_metal = ml.meanline.get_subsonic_deviation(
    Ma_exit, Ma_crit, opening / pitch, deviation_model="metal_angle"
)
beta_aungier = ml.meanline.get_subsonic_deviation(
    Ma_exit, Ma_crit, opening / pitch, deviation_model="aungier"
)
beta_ainley = ml.meanline.get_subsonic_deviation(
    Ma_exit, Ma_crit, opening / pitch, deviation_model="ainley_mathieson"
)

# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Deviation model testing")
ax.set_xlabel(r"$\mathrm{Ma}_{out}$ - Exit Mach number")
ax.set_ylabel(r"$\beta_{\mathrm{out}}$ - Exit flow angle")
ax.set_xscale("linear")
ax.set_yscale("linear")
ax.plot(Ma_exit, beta_ainley, linewidth=1.25, label="Ainley-Mathieson")
ax.plot(Ma_exit, beta_aungier, linewidth=1.25, label="Aungier")
ax.plot(Ma_exit, beta_metal, linewidth=1.25, label="Metal")
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
plt.show()
