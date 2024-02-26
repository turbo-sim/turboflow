import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import meanline_axial as ml

# Define case parameters
Ma_crit = 1
Ma_exit = np.linspace(0.00, 1.00, 200)
geometry = {"A_throat" : 0.5,
            "A_out" : 1}

# Compute exit flow angles
beta_aungier = ml.deviation_model.get_subsonic_deviation(
    Ma_exit, Ma_crit, geometry, 'aungier'
)
beta_ainley = ml.deviation_model.get_subsonic_deviation(
    Ma_exit, Ma_crit, geometry, "ainley_mathieson",
)

# Plot comparison
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Deviation model testing")
ax.set_xlabel(r"$\mathrm{Ma}_{out}$ - Exit Mach number")
ax.set_ylabel(r"$\beta_{\mathrm{out}}$ - Exit flow angle")
ax.set_xscale("linear")
ax.set_yscale("linear")
ax.plot(Ma_exit, beta_ainley, linewidth=1.25, label="Ainley-Mathieson")
ax.plot(Ma_exit, beta_aungier, linewidth=1.25, label="Aungier")
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)

ax.set_ylim([58, 60.1])

plt.show()
