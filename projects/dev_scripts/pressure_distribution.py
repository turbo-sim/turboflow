import os
import numpy as np
import matplotlib.pyplot as plt
import turbo_flow as tf
import turbo_flow.axial_turbine.geometry_model as geom
from scipy.optimize import root_scalar
  
# Define auxiliaty functions
def get_enthalpy_residual_inlet(p_in, h0_in, s_in, mass_flow, area_in):
    state_in = Fluid.set_state(tf.PSmass_INPUTS, p_in, s_in)
    v_in = mass_flow / (state_in.d * area_in)
    residual = (h0_in - state_in.h - 0.5 * v_in ** 2) / h0_in
    return residual
 
def get_enthalpy_residual_outlet(p_out, h0_rel_in, s_in, mass_flow, area_out, angle_out):
    state_out = Fluid.set_state(tf.PSmass_INPUTS, p_out, s_in)
    velocity_out = mass_flow / (state_out.d * area_out * tf.cosd(angle_out))
    residual = (h0_rel_in - state_out.h - 0.5 * velocity_out ** 2) / h0_rel_in
    return residual
 
# Load configuration file
CONFIG_FILE = r"C:\Users\lasseba\Python Scripts\Mean line model\meanline-axial\projects\Kofskey1972_one_stage\Kofskey1972_1stage.yaml"
config = tf.read_configuration_file(CONFIG_FILE)
 
# Process geometry
geometry = config["geometry"]
geom.validate_input_geometry(geometry, display=True)
geometry = geom.calculate_full_geometry(config["geometry"])
 
# Get operating conditions
omega = config["operation_points"]["omega"]
p0_in = config["operation_points"]["p0_in"]
T0_in = config["operation_points"]["T0_in"]
fluid_name = config["operation_points"]["fluid_name"]
 
# Inlet state
Fluid = tf.Fluid(fluid_name)
state_01 = Fluid.set_state(tf.PT_INPUTS, p0_in, T0_in)
gamma = state_01.gamma
 
# Preallocate NumPy arrays
two_stage = True
if two_stage:
    mass_flows = np.linspace(1.5, 2.31, 100)  # Two-stage
else:
    mass_flows = np.linspace(1.5, 2.9, 100)  # One-stage
 
n_points = len(mass_flows)
p_1_array = np.zeros(n_points)
p_2_array = np.zeros(n_points)
p_3_array = np.zeros(n_points)
p_4_array = np.zeros(n_points)
p_5_array = np.zeros(n_points)
Ma_1_array = np.zeros(n_points)
Ma_2_array = np.zeros(n_points)
Ma_3_array = np.zeros(n_points)
Ma_4_array = np.zeros(n_points)
Ma_5_array = np.zeros(n_points)
mass_flux_crit_1 = np.zeros(n_points)
mass_flux_crit_2 = np.zeros(n_points)
mass_flux_crit_3 = np.zeros(n_points)
mass_flux_crit_4 = np.zeros(n_points)
mass_flow_crit_1 = np.zeros(n_points)
mass_flow_crit_2 = np.zeros(n_points)
mass_flow_crit_3 = np.zeros(n_points)
mass_flow_crit_4 = np.zeros(n_points)
 
# Loop over mass flows
# Not the best to program it (a lot of repeated code), but it works
for i, mass_flow in enumerate(mass_flows):
 
    # Stator 1 inlet
    area_1 = geometry["A_in"][0]
    res = lambda p: get_enthalpy_residual_inlet(p, state_01.h, state_01.s, mass_flow, area_1)
    result = root_scalar(res, x0=p0_in, method='newton')
    p_1 = result.root
    state_1 = Fluid.set_state(tf.PSmass_INPUTS, p_1, state_01.s)
    v_1 = mass_flow / (state_1.d * area_1)
    Ma_1 = v_1 / state_1.a
    p_1_array[i] = p_1
    Ma_1_array[i] = Ma_1
 
    # Stator 1 outlet
    area = geometry["A_out"][0]
    radius = geometry["radius_mean_out"][0]
    angle = geometry["metal_angle_te"][0]
    h0rel_in = state_01.h  # Inlet stagnation enthalpy
    res = lambda p: get_enthalpy_residual_outlet(p, h0rel_in, state_01.s, mass_flow, area, angle)
    result = root_scalar(res, x0=p0_in*0.9, method='newton')
    p_2 = result.root
    state_2 = Fluid.set_state(tf.PSmass_INPUTS, p_2, state_01.s)
    u_2 = omega * radius
    v_2 = mass_flow / (state_2.d * area * tf.cosd(angle))
    v_t_2 = v_2 * tf.sind(angle)
    v_m_2 = v_2 * tf.cosd(angle)
    w_t_2 = v_t_2 - u_2
    w_m_2 = v_m_2
    w_2 = np.sqrt(w_m_2 ** 2 + w_t_2 ** 2)
    Ma_2 = v_2 / state_2.a
    Ma_rel_2 = w_2 / state_2.a
    p_2_array[i] = p_2
    Ma_2_array[i] = Ma_2
    state_01 = Fluid.set_state(tf.HmassSmass_INPUTS, h0rel_in, state_01.s)
    d_crit = state_01.d * ((gamma + 1) / 2) ** (-1 / (gamma-1))
    a_crit = state_01.a * ((gamma + 1) / 2) ** (-0.5)
    mass_flux_crit_1[i] = d_crit*a_crit
    mass_flow_crit_1[i] = d_crit*a_crit*area*tf.cosd(angle)
 
    # Rotor 1 outlet
    area = geometry["A_out"][1]
    radius = geometry["radius_mean_out"][1]
    angle = geometry["metal_angle_te"][1]
    h0rel_in = state_2.h + 0.5 * w_2 ** 2  # Inlet stagnation enthalpy
    res = lambda p: get_enthalpy_residual_outlet(p, h0rel_in, state_01.s, mass_flow, area, angle)
    result = root_scalar(res, x0=p_2, method='newton')
    p_3 = result.root
    state_3 = Fluid.set_state(tf.PSmass_INPUTS, p_3, state_01.s)
    u_3 = omega * radius
    w_3 = mass_flow / (state_3.d * area * tf.cosd(angle))
    w_t_3 = w_3 * tf.sind(angle)
    w_m_3 = w_3 * tf.cosd(angle)
    v_t_3 = w_t_3 + u_3
    v_m_3 = w_m_3
    v_3 = np.sqrt(v_m_3 ** 2 + v_t_3 ** 2)
    Ma_3 = v_3 / state_3.a
    Ma_rel_3 = w_3 / state_3.a
    p_3_array[i] = p_3
    Ma_3_array[i] = Ma_rel_3
    state_02rel = Fluid.set_state(tf.HmassSmass_INPUTS, h0rel_in, state_01.s)
    d_crit = state_02rel.d * ((gamma + 1) / 2) ** (-1 / (gamma-1))
    a_crit = state_02rel.a * ((gamma + 1) / 2) ** (-0.5)
    mass_flux_crit_2[i] = d_crit*a_crit
    mass_flow_crit_2[i] = d_crit*a_crit*area*tf.cosd(angle)
 
    if two_stage:
        # Stator 2 outlet
        area = geometry["A_out"][0]
        radius = geometry["radius_mean_out"][0]
        angle = geometry["metal_angle_te"][0]
        h0rel_in = state_3.h + 0.5 * v_3 ** 2  # Inlet stagnation enthalpy
        res = lambda p: get_enthalpy_residual_outlet(p, h0rel_in, state_01.s, mass_flow, area, angle)
        result = root_scalar(res, x0=p0_in*0.9, method='newton')
        p_4 = result.root
        state_4 = Fluid.set_state(tf.PSmass_INPUTS, p_4, state_01.s)
        u_4 = omega * radius
        v_4 = mass_flow / (state_4.d * area * tf.cosd(angle))
        v_t_4 = v_4 * tf.sind(angle)
        v_m_4 = v_4 * tf.cosd(angle)
        w_t_4 = v_t_4 - u_4
        w_m_4 = v_m_4
        w_4 = np.sqrt(w_m_4 ** 2 + w_t_4 ** 2)
        Ma_4 = v_4 / state_4.a
        Ma_rel_4 = w_4 / state_4.a
        p_4_array[i] = p_4
        Ma_4_array[i] = Ma_4
        state_03 = Fluid.set_state(tf.HmassSmass_INPUTS, h0rel_in, state_01.s)
        d_crit = state_03.d * ((gamma + 1) / 2) ** (-1 / (gamma-1))
        a_crit = state_03.a * ((gamma + 1) / 2) ** (-0.5)
        mass_flux_crit_3[i] = d_crit*a_crit
        mass_flow_crit_3[i] = d_crit*a_crit*area*tf.cosd(angle)
 
        # Rotor 2 outlet
        area = geometry["A_out"][1]
        radius = geometry["radius_mean_out"][1]
        angle = geometry["metal_angle_te"][1]
        h0rel_in = state_4.h + 0.5 * w_4 ** 2  # Inlet stagnation enthalpy
        res = lambda p: get_enthalpy_residual_outlet(p, h0rel_in, state_01.s, mass_flow, area, angle)
        result = root_scalar(res, x0=p_2, method='newton')
        p_5 = result.root
        state_5 = Fluid.set_state(tf.PSmass_INPUTS, p_5, state_01.s)
        u_5 = omega * radius
        w_5 = mass_flow / (state_5.d * area * tf.cosd(angle))
        w_t_5 = w_5 * tf.sind(angle)
        w_m_5 = w_5 * tf.cosd(angle)
        v_t_5 = w_t_5 + u_5
        v_m_5 = w_m_5
        v_5 = np.sqrt(v_m_5 ** 2 + v_t_5 ** 2)
        Ma_5 = v_5 / state_5.a
        Ma_rel_5 = w_5 / state_5.a
        p_5_array[i] = p_5
        Ma_5_array[i] = Ma_rel_5
        state_04rel = Fluid.set_state(tf.HmassSmass_INPUTS, h0rel_in, state_01.s)
        d_crit = state_04rel.d * ((gamma + 1) / 2) ** (-1 / (gamma-1))
        a_crit = state_04rel.a * ((gamma + 1) / 2) ** (-0.5)
        mass_flux_crit_4[i] = d_crit*a_crit
        mass_flow_crit_4[i] = d_crit*a_crit*area*tf.cosd(angle)
 
# Compute overall total-to-static pressure ratip
if two_stage:
    PR = p0_in / p_5_array
else:
    PR = p0_in / p_3_array
 
 
marker = 'none'
color_map = "Reds"
colors = plt.get_cmap(color_map)(np.linspace(0.3, 0.9, 5))
 
# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Mach number vs pressure ratio")
ax.set_xlabel(r"Pressure ratio")
ax.set_ylabel(r"Mach number")
ax.set_xscale("linear")
ax.set_yscale("linear")
# ax.plot(PR, Ma_1_array, linewidth=1.25, label="Stator 1 inlet", marker=marker)
ax.plot(PR, Ma_2_array, linewidth=1.25, label="Stator 1 exit", marker=marker, color=colors[0])
ax.plot(PR, Ma_3_array, linewidth=1.25, label="Rotor 1 exit", marker=marker, color=colors[1])
ax.plot(PR, Ma_4_array, linewidth=1.25, label="Stator 2 exit", marker=marker, color=colors[2])
ax.plot(PR, Ma_5_array, linewidth=1.25, label="Rotor 2 exit", marker=marker, color=colors[3])
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# ax.set_ylim([58, 60.1])
 
# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Static pressure vs pressure ratio")
ax.set_xlabel(r"Pressure ratio")
ax.set_ylabel(r"Static pressure [Pa]")
ax.set_xscale("linear")
ax.set_yscale("linear")
# ax.plot(PR, p_1_array, linewidth=1.25, label="Stator 1 inlet", marker=marker)
ax.plot(PR, p_2_array, linewidth=1.25, label="Stator 1 exit", marker=marker, color=colors[0])
ax.plot(PR, p_3_array, linewidth=1.25, label="Rotor 1 exit", marker=marker, color=colors[1])
ax.plot(PR, p_4_array, linewidth=1.25, label="Stator 2 exit", marker=marker, color=colors[2])
ax.plot(PR, p_5_array, linewidth=1.25, label="Rotor 2 exit", marker=marker, color=colors[3])
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# ax.set_ylim([58, 60.1])
 
# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Mass flow rate vs pressure ratio")
ax.set_xlabel(r"Pressure ratio")
ax.set_ylabel(r"Mass flow rate [kg/s]")
ax.set_xscale("linear")
ax.set_yscale("linear")
ax.plot(PR, mass_flows, linewidth=1.25, label="Actual", marker=marker)
ax.plot(PR, mass_flow_crit_1, linewidth=1.25, label="Critical stator 1", marker=marker, color=colors[0])
ax.plot(PR, mass_flow_crit_2, linewidth=1.25, label="Critical rotor 1", marker=marker, color=colors[1])
ax.plot(PR, mass_flow_crit_3, linewidth=1.25, label="Critical stator 2", marker=marker, color=colors[2])
ax.plot(PR, mass_flow_crit_4, linewidth=1.25, label="Critical rotor 2", marker=marker, color=colors[3])
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# ax.set_ylim([58, 60.1])
 
# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Critical mass flux vs pressure ratio")
ax.set_xlabel(r"Pressure ratio")
ax.set_ylabel(r"Critical mass flux [kg/s/m2]")
ax.set_xscale("linear")
ax.set_yscale("linear")
ax.plot(PR, mass_flux_crit_1, linewidth=1.25, label="Critical stator 1", marker=marker, color=colors[0])
ax.plot(PR, mass_flux_crit_2, linewidth=1.25, label="Critical rotor 1", marker=marker, color=colors[1])
ax.plot(PR, mass_flux_crit_3, linewidth=1.25, label="Critical stator 2", marker=marker, color=colors[2])
ax.plot(PR, mass_flux_crit_4, linewidth=1.25, label="Critical rotor 2", marker=marker, color=colors[3])
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# ax.set_ylim([58, 60.1])
 
# # Plot results
# fig = plt.figure(figsize=(6.4, 4.8))
# ax = fig.gca()
# ax.set_title("Corrected mass flow vs pressure ratio")
# ax.set_xlabel(r"Pressure ratio")
# ax.set_ylabel(r"Corrected mass flow")
# ax.set_xscale("linear")
# ax.set_yscale("linear")
# ax.plot(PR, mass_flows/mass_flux_crit_1, linewidth=1.25, label="Stator 1", marker=marker, color=colors[0])
# ax.plot(PR, mass_flows/mass_flux_crit_2, linewidth=1.25, label="Rotor 1", marker=marker, color=colors[1])
# ax.plot(PR, mass_flows/mass_flux_crit_3, linewidth=1.25, label="Stator 2", marker=marker, color=colors[2])
# ax.plot(PR, mass_flows/mass_flux_crit_4, linewidth=1.25, label="Rotor 2", marker=marker, color=colors[3])
# leg = ax.legend(loc="best")
# fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# # ax.set_ylim([58, 60.1])
 
 
plt.show()