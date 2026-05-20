import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import turboflow as tf

# -------------------------
# Paths and setup
# -------------------------
dir_figs = "figures"
os.makedirs(dir_figs, exist_ok=True)

tf.set_plot_options()

# NOTE: updated to match OUT_DIR = "output_axial" from your run script
df = pd.read_excel("./output_axial/one_stage_comp_wise_latest.xlsx", sheet_name="overall")
df_exp = pd.read_excel("./validation_cases_summary.xlsx")

# -------------------------
# Helper for repeated plotting pattern
# -------------------------
def plot_validation_curve(
    df,
    df_exp,
    y_col_sim: str,
    y_col_exp: str,
    y_label: str,
    filename: str,
    y_lim=None,
    x_lim=(1.4, 4.7),
):
    """
    Plot PR_ts vs selected y quantity (simulation + experiment) for several speeds.
    Uses same color + marker logic as your mass-flow figure.
    """
    fig, ax = plt.subplots()

    # Color palette (plasma_r as before)
    colors = plt.get_cmap("plasma_r")(np.linspace(0.2, 1.0, 4))

    unique_omegas = df["angular_speed"][df["angular_speed"] > 0.6 * 1627].unique()
    unique_omegas = np.sort(unique_omegas)

    # For building combined legend
    handles_sim = []
    labels_sim = []
    handles_exp = []
    labels_exp = []

    markers = ["o", "s", "^", "D"]

    for idx, omega in enumerate(unique_omegas):
        color = colors[idx % len(colors)]

        # Simulation data
        mask_sim = df["angular_speed"] == omega
        sim_x = df.loc[mask_sim, "PR_ts"]
        sim_y = df.loc[mask_sim, y_col_sim]

        line, = ax.plot(
            sim_x,
            sim_y,
            color=color,
            label=f"{int(np.round((omega / 1627) * 100))}",
        )
        handles_sim.append(line)
        labels_sim.append(f"{int(np.round((omega / 1627) * 100))}")

        # Experimental data
        mask_exp = np.isclose(df_exp["omega"], omega)
        exp_x = df_exp.loc[mask_exp, "pressure_ratio_ts"]
        exp_y = df_exp.loc[mask_exp, y_col_exp]

        pt, = ax.plot(
            exp_x,
            exp_y,
            linestyle="None",
            marker=markers[idx % len(markers)],
            color=color,
            label=f"{int(np.round((omega / 1627) * 100))}",
        )
        handles_exp.append(pt)
        labels_exp.append(f"{int(np.round((omega / 1627) * 100))}")

    # Labels / limits
    ax.set_xlabel(
        r"Total-to-static pressure ratio [$p_{0,\text{in}} / p_{\text{out}}$]",
        fontsize=22,
    )
    ax.set_ylabel(y_label, fontsize=22)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Combined legend (sim + exp)
    handles = handles_sim + handles_exp
    labels = labels_sim + labels_exp
    legend = ax.legend(
        handles=handles,
        labels=labels,
        title="Percent of design\nangular speed",
        loc="lower right",
        fontsize=10,
        ncol=2,
    )
    legend.get_title().set_horizontalalignment("center")

    plt.tight_layout()
    filepath = os.path.join(dir_figs, filename)
    tf.savefig_in_formats(fig, filepath)

    return fig, ax


# -------------------------
# 1) Efficiency vs PR_ts
# -------------------------
fig_eta, ax_eta = plot_validation_curve(
    df=df,
    df_exp=df_exp,
    y_col_sim="efficiency_ts",
    y_col_exp="efficiency_ts",
    y_label=r"Total-to-static efficiency $\eta_{ts}$ [-]",
    filename="efficiency_ts_validation",
    y_lim=(40, 90),  # set to e.g. (0.6, 1.0) if you want
)

# -------------------------
# 2) Torque vs PR_ts
# -------------------------
fig_torque, ax_torque = plot_validation_curve(
    df=df,
    df_exp=df_exp,
    y_col_sim="torque",
    y_col_exp="torque",
    y_label=r"Torque [N·m]",
    filename="torque_validation",
    y_lim=(40, 140),  # set if you want tighter range
)

# -------------------------
# 3) Mass flow vs PR_ts  (your original styled figure)
# -------------------------
fig_mdot, ax_mdot = plot_validation_curve(
    df=df,
    df_exp=df_exp,
    y_col_sim="mass_flow_rate",
    y_col_exp="mass_flow_rate",
    y_label=r"Mass flow rate [kg/s]",
    filename="mass_flow_rate_validation_updated",
    y_lim=(2.55, 2.85),
)

# Show all figures
plt.show()
