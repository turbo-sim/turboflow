import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set filenames
filename_exp = "experimental_data/experimental_data_kofskey1972_1stage_raw.xlsx"
filename_sim = "output/experimental_points.xlsx"

# Set design point angular speed
omega_des = 1627

# Define percentage limits
limits = [2.5, 5, 10]

def count(values, lim):
    below_lim_count = np.count_nonzero(abs(values) < lim)
    total_count = len(values)

    if total_count == 0:
        return 0

    percentage_below_lim = (below_lim_count / total_count) * 100
    return percentage_below_lim

# Load simulation data
speed_percent = np.flip([110, 100, 90, 70])
data_sim = pd.read_excel(filename_sim, sheet_name=["overall"])
data_sim = data_sim["overall"]
data_mass_flow = data_sim[0:37]
data_torque = data_sim[37:85]
data_eta = data_sim[85:170]
data_alpha = data_sim[170:]

# Load experimental data
sheets = ["Mass flow rate", "Torque", "alpha_out"]
data_exp = pd.read_excel(filename_exp, sheet_name=sheets)
for sheet in sheets:
    data_exp[sheet] = data_exp[sheet][data_exp[sheet]["omega"].isin(speed_percent)]

# Initalize dictionary to store results
mass_flow_rate_error = {}
torque_error = {}
alpha_error = {}
mass_flow_rate_count = {}
torque_count = {}
alpha_count = {}

for speed in speed_percent:

    # Quantify mass flow rate error
    mass_flow_rate_exp = data_exp["Mass flow rate"][
        data_exp["Mass flow rate"]["omega"] == speed
    ]["m"].values
    mass_flow_rate_sim = data_mass_flow[
        (data_mass_flow["angular_speed"] > speed/100*omega_des - 1)
        & (data_mass_flow["angular_speed"] < speed/100*omega_des + 1)
    ]["mass_flow_rate"].values
    mass_flow_rate_error[speed] = (
        (mass_flow_rate_exp - mass_flow_rate_sim) / mass_flow_rate_exp * 100
    )
    mass_flow_rate_count[speed] = [
        count(mass_flow_rate_error[speed], limits[0]),
        count(mass_flow_rate_error[speed], limits[1]),
        count(mass_flow_rate_error[speed], limits[2]),
    ]

    # Quantify torque error
    torque_exp = data_exp["Torque"][data_exp["Torque"]["omega"] == speed][
        "Torque"
    ].values
    torque_sim = data_torque[
        (data_torque["angular_speed"] > speed/100*omega_des - 1)
        & (data_torque["angular_speed"] < speed/100*omega_des + 1)
    ]["torque"].values
    torque_error[speed] = (torque_exp - torque_sim) / torque_exp * 100
    torque_count[speed] = [
        count(torque_error[speed], limits[0]),
        count(torque_error[speed], limits[1]),
        count(torque_error[speed], limits[2]),
    ]

    # Quantify alpha error
    alpha_exp = np.cos(
        data_exp["alpha_out"][data_exp["alpha_out"]["omega"] == speed][
            "alpha_out"
        ].values
        * np.pi
        / 180
    )
    alpha_sim = np.cos(
        data_alpha[
            (data_alpha["angular_speed"] > speed/100*omega_des - 1)
            & (data_alpha["angular_speed"] < speed/100*omega_des + 1)
        ]["exit_flow_angle"].values
        * np.pi
        / 180
    )
    alpha_error[speed] = (alpha_exp - alpha_sim) / alpha_exp * 100
    alpha_count[speed] = [
        count(alpha_error[speed], limits[0]),
        count(alpha_error[speed], limits[1]),
        count(alpha_error[speed], limits[2]),
    ]


# Count overall error
print(
    f"Mass flow rate error < {limits[0]}: {count(np.concatenate(list(mass_flow_rate_error.values())),limits[0])}"
)
print(
    f"Mass flow rate error < {limits[1]}: {count(np.concatenate(list(mass_flow_rate_error.values())),limits[1])}"
)
print(
    f"Mass flow rate error < {limits[2]}: {count(np.concatenate(list(mass_flow_rate_error.values())),limits[2])}"
)

print(
    f"Torque error < {limits[0]}: {count(np.concatenate(list(torque_error.values())),limits[0])}"
)
print(
    f"Torque error < {limits[1]}: {count(np.concatenate(list(torque_error.values())),limits[1])}"
)
print(
    f"Torque error < {limits[2]}: {count(np.concatenate(list(torque_error.values())),limits[2])}"
)

print(
    f"Alpha error < {limits[0]}: {count(np.concatenate(list(alpha_error.values())),limits[0])}"
)
print(
    f"Alpha error < {limits[1]}: {count(np.concatenate(list(alpha_error.values())),limits[1])}"
)
print(
    f"Alpha error < {limits[2]}: {count(np.concatenate(list(alpha_error.values())),limits[2])}"
)
