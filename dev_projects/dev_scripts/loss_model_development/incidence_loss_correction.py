import os
import numpy as np
import matplotlib.pyplot as plt
import turboflow.axial_turbine.loss_model_benner as tf


# Create the folder to save figures
fig_dir = "figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Define the distance parameter
chi = np.linspace(-20, 20, 1000)
# Define the loss coefficient_increment
loss_1 = tf.get_incidence_profile_loss_increment(
    chi, chi_extrapolation=10, loss_limit=None
)
loss_2 = tf.get_incidence_profile_loss_increment(
    chi, chi_extrapolation=5, loss_limit=None
)
loss_3 = tf.get_incidence_profile_loss_increment(
    chi, chi_extrapolation=5, loss_limit=0.5
)

# Create figure
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
# ax.set_title("Benner incidence loss model")
# ax.set_xlabel("$\chi$ - Distance parameter")
# ax.set_ylabel("$\Delta \phi_{p}^2$ - Loss coefficient increment")
ax.set_xlabel("Incidence angle")
ax.set_ylabel("Incidence losses")
ax.set_xscale("linear")
ax.set_yscale("linear")
ax.set_ylim([0, 0.1])
ax.set_xlim([-5,5])

# Plot simulation data
ax.plot(chi, loss_1, 'k', linewidth=1.25, label="Original correlation")
# ax.plot(chi, loss_2, linewidth=1.25, label="Linear extrapolation")
# ax.plot(chi, loss_3, linewidth=1.25, label="Extrapolation and limiting")

# Custom tick positions and labels
tick_positions = [-2, 0, 2]  # Specify the positions where you want ticks
tick_labels = ['Negative incidence', '0', 'Positive incidence']  # Corresponding labels
ax.grid(False)

# Set custom ticks and labels on the x-axis
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticklabels([])
# Remove ticks (set tick length to 0)
# ax.tick_params(axis='x', length=0)  # Remove x-axis ticks
# ax.tick_params(axis='y', length=0)  # Remove y-axis ticks (optional)
ax.tick_params(axis='x', which='both', bottom=False, top=False)  # Hide x-axis ticks
ax.tick_params(axis='y', which='both', left=False, right=False)  # Hide y-axis ticks (optional)

# Create legend
# leg = ax.legend(loc="upper left")

# Adjust PAD
fig.tight_layout(pad=1, w_pad=None, h_pad=None)

# Save plots
filename = os.path.join(fig_dir, "incidence_loss_modification")
# fig.savefig(filename + ".png", bbox_inches="tight")
# fig.savefig(filename + ".svg", bbox_inches="tight")

# Show figure
plt.show()
