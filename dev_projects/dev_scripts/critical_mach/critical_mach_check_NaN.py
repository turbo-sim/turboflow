import os
import numpy as np
import matplotlib.pyplot as plt
import turboflow as tf
from matplotlib import cm

tf.set_plot_options()


# Plot the critical Mach number against the kinetic energy loss coefficient
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\Delta \phi^2$ - Kinetic energy loss coefficient")
ax.set_ylabel(r"$\text{Ma}_\text{crit}$ - Critical Mach number")
gamma = 1.4
eta = np.linspace(-0.1, 1.1, 101)
Ma_crit = tf.get_mach_crit(gamma, eta)
phi_crit = tf.get_flow_capacity(Ma_crit, gamma, eta)
ax.plot(eta, Ma_crit, linestyle='-')
# ax.legend(loc="lower right", ncol=1, fontsize=11)
plt.tight_layout(pad=1)

# Show figures
plt.show()