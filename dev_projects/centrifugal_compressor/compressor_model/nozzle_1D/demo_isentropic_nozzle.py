import functions
import numpy as np
import matplotlib.pyplot as plt


# Validate model for isentropic flow in converging nozzle
fluid_name = "nitrogen"
fluid = functions.Fluid(fluid_name)
T_in = 300
p_in = 101325
mach_in = 0.3
length = 1.00
diameter_in = 0.01
roughness = 0.00
state_in = fluid.set_state(functions.PT_INPUTS, p_in, T_in)
v_in = mach_in * fluid.a
critical_area_ratio = functions.get_critical_area_ratio_isentropic(
    mach_in, state_in.gamma
)
area_ratio = 1.001 * 1 / critical_area_ratio


# Calculate solution
solution = functions.pipeline_steady_state_1D(
    fluid_name=fluid_name,
    pressure_in=p_in,
    temperature_in=T_in,
    # mass_flow=mass_flow,
    roughness=roughness,
    mach_in=mach_in,
    diameter_in=diameter_in,
    length=length,
    area_ratio=area_ratio,
    include_friction=False,
    include_heat_transfer=False,
    number_of_points=50,
)


# Plot evolution of flow variables
functions.set_plot_options()
figure, ax = plt.subplots(figsize=(6.0, 4.8))
ax.set_xlabel("Distance along pipe [m]")
ax.set_ylabel("Normalized flow variables")
ax.plot(
    solution["distance"],
    solution["pressure"] / solution["pressure"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="$p/p_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"],
    solution["velocity"] / solution["velocity"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="$v/v_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"],
    solution["temperature"] / solution["temperature"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="$T/T_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"],
    solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="$Ma=v/a$",
)
ax.legend(loc="best")
figure.tight_layout(pad=1)


# Plot numerical integration error
# The mass should always be conserved
# The total enthalpy is conserved if the heat transfer is zero
# The entropy is conserved if both heat transfer and friction are zero
m_error = solution["mass_flow"] / solution["mass_flow"][0] - 1
h_error = solution["total_enthalpy"] / solution["total_enthalpy"][0] - 1
s_error = solution["entropy"] / solution["entropy"][0] - 1
figure, ax = plt.subplots(figsize=(6.0, 4.8))
ax.set_xlabel("Pipeline distance [km]")
ax.set_ylabel("Integration error")
ax.set_yscale("log")
ax.plot(solution["distance"], np.abs(m_error), label="Mass flow error")
ax.plot(solution["distance"], np.abs(h_error), label="Total enthalpy error")
ax.plot(solution["distance"], np.abs(s_error), label="Entropy error")
ax.legend(loc="best", fontsize=9)
figure.tight_layout(pad=1)

# Show figures
plt.show()
