import functions
import matplotlib.pyplot as plt

def psig_to_Pa(x):
    return (0.0689476 * x) * 1e5 + 101325

def in_to_m(x):
    return 0.0254 * x


# Validate model for isentropic flow in converging nozzle
fluid_name = "hydrogen"
T_in = 50 + 273.15
T_ext = 5 + 273.15
p_in = psig_to_Pa(2500)
mach_in = 0.02
length = 100e3
diameter_in = in_to_m(20)
roughness = 10e-6

# Calculate solution
solution = functions.pipeline_steady_state_1D(
    fluid_name=fluid_name,
    pressure_in=p_in,
    temperature_in=T_in,
    temperature_external=T_ext,
    # mass_flow=mass_flow,
    roughness=roughness,
    mach_in=mach_in,
    diameter_in=diameter_in,
    length=length,
    include_friction=True,
    include_heat_transfer=True,
)

print(f"Mass flow rate: {solution['mass_flow'][-1]:0.2f} kg/s")


# Create figure and first axis
functions.set_plot_options(grid=True)
colors = functions.COLORS_MATLAB
figure, ax1 = plt.subplots(figsize=(6.0, 4.8))
ax1.set_title("Flow in a hydrogen pipeline")
ax1.set_xlabel("Pipeline distance [km]")
ax1.set_ylabel("Pressure, Density, and Temperature")
ax1.plot(
    solution["distance"] / 1e3,
    solution["pressure"] / 1e5,
    label="Pressure [bar]",
    color=colors[0],
)
ax1.plot(
    solution["distance"] / 1e3,
    solution["density"],
    label="Density [kg/m$^3$]",
    color=colors[1],
)
ax1.plot(
    solution["distance"] / 1e3,
    solution["temperature"] - 273.15,
    label="Temperature [$^\circ$C]",
    color=colors[2],
)

# Create second axis
ax2 = ax1.twinx()
ax2.grid(False)
ax2.set_ylabel("Mach Number")
ax2.plot(
    solution["distance"] / 1e3,
    solution["mach_number"],
    label="Mach Number",
    color=colors[3],
)

# Create combined legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')
ax1.set_ylim([0, 200])
ax2.set_ylim([0, 0.2])


# Plot evolution of flow variables
functions.set_plot_options()
figure, ax = plt.subplots(figsize=(6.0, 4.8))
ax.set_xlabel("Pipeline distance [km]")
ax.set_ylabel("Normalized flow variables")
ax.plot(
    solution["distance"] / 1e3,
    solution["pressure"] / solution["pressure"][0],
    linewidth=1.00,
    label="$p/p_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"] / 1e3,
    solution["velocity"] / solution["velocity"][0],
    linewidth=1.00,
    label="$v/v_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"] / 1e3,
    solution["temperature"] / solution["temperature"][0],
    linewidth=1.00,
    label="$T/T_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"] / 1e3,
    solution["density"] / solution["density"][0],
    linewidth=1.00,
    label=r"$\rho/\rho_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"] / 1e3,
    solution["mach_number"],
    linewidth=1.00,
    label="$Ma=v/a$",
)
ax.legend(loc="best")
figure.tight_layout(pad=1)


# Show figures
plt.show()
