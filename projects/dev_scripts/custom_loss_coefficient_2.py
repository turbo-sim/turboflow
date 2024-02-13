# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:48:28 2024

@author: lasseba
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp as cp
from scipy import optimize

# Define bounary conditions
p0_in = 13.8e4
p_out = p0_in/2.3
T0_in = 295.6
fluid_name = 'air'

# Define geometry
geometry = {"A_throat" : 0.025,
            "A_out" : 0.075}

# Define loss coefficient at throat and exit 
loss_coefficients = {"throat" : 0.1,
                     "out" : 0.2}

# Define fluid
fluid = cp.AbstractState('HEOS', fluid_name)

# Calculate inlet state
fluid.update(cp.PT_INPUTS, p0_in, T0_in)
h0_in = fluid.hmass()
s_in = fluid.smass()

# Define inlet stagnation state
inlet_stagnation = {"p0_in" : p0_in,
                    "h0_in" : h0_in}

def get_exit_flow_angle_aungier(Ma_exit, Ma_crit_throat, geometry):
    """
    Calculate deviation angle using the method proposed by :cite:`aungier_turbine_2006`.
    """
    # TODO add equations of Aungier model to docstring
    # gauging_angle = geometry["metal_angle_te"]
    gauging_angle = np.arccos(geometry["A_throat"]/geometry["A_out"])*180/np.pi

    # Compute deviation for  Ma<0.5 (low-speed)
    beta_g = 90-abs(gauging_angle)
    delta_0 = np.arcsin(np.cos(gauging_angle*np.pi/180) * (1 + (1 - np.cos(gauging_angle*np.pi/180)) * (beta_g / 90) ** 2))*180/np.pi - beta_g

    # Initialize an empty array to store the results
    delta = np.empty_like(Ma_exit, dtype=float)

    # Compute deviation for Ma_exit < 0.50 (low-speed)
    low_speed_mask = Ma_exit < 0.50
    delta[low_speed_mask] = delta_0

    # Compute deviation for 0.50 <= Ma_exit < 1.00
    medium_speed_mask = (0.50 <= Ma_exit)  & (Ma_exit < Ma_crit_throat)
    X = (2 * Ma_exit[medium_speed_mask] - 1) / (2 * Ma_crit_throat - 1)
    delta[medium_speed_mask] = delta_0 * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)

    # Extrapolate to zero deviation for supersonic flow
    supersonic_mask = Ma_exit >= Ma_crit_throat
    delta[supersonic_mask] = 0.00

    # Compute flow angle from deviation
    beta = abs(gauging_angle) - delta

    return beta

def evaluate_plane(x, h0_in, p0_in, A, fluid):
    
    v = x[0]
    p = x[1]
    beta = x[2]
            
    # Calculate static state
    h = h0_in-0.5*v**2 
    fluid.update(cp.HmassP_INPUTS, h, p)
    s = fluid.smass()
    d = fluid.rhomass()
    a = fluid.speed_sound()
    
    # Calculate total state
    fluid.update(cp.HmassSmass_INPUTS, h0_in, s)
    p0 = fluid.p()
    
    Ma = v/a
    m = d*v*np.cos(beta*np.pi/180)*A
    Y_def = (p0_in-p0)/(p0-p)

    return m, Ma, Y_def


def compute_critical_state(v_out, p_out, beta_out, h0_in, p0_in, v0, geometry, loss_coefficients, fluid):
     
    Y_throat = loss_coefficients["throat"]
    Y_out = loss_coefficients["out"]
    
    f = lambda x : -1*evaluate_plane([x[0]*v0,x[1]*p0_in,0], h0_in, p0_in, geometry["A_throat"], fluid)[0] 
    cons = lambda x : evaluate_plane([x[0]*v0,x[1]*p0_in,0], h0_in, p0_in, geometry["A_throat"], fluid)[-1] - Y_throat
    constraint = {"type" : 'eq', "fun" : cons}
    bounds = [(0.1,1.5), (0.1, 0.8)]
    
    solution = optimize.minimize(f, x0 = [0.8, 0.5], bounds = bounds, constraints = constraint)
    v_throat = solution.x[0]*v0
    p_throat = solution.x[1]*p0_in
    m_crit_throat, Ma_crit_throat, Y_crit_throat = evaluate_plane([v_throat, p_throat, 0], h0_in, p0_in, geometry["A_throat"], fluid)
    critical_state_throat = {"mass_flow" : m_crit_throat,
                             "Ma" : Ma_crit_throat,
                             "Y" : Y_crit_throat}
    
    m_crit_out, Ma_crit_out, Y_crit_out = evaluate_plane([v_out, p_out, beta_out], h0_in, p0_in, geometry["A_out"], fluid) 
    critical_state_out = {"mass_flow" : m_crit_out,
                             "Ma" : Ma_crit_out,
                             "Y" : Y_crit_out}
    
    critical_state = {"throat" : critical_state_throat, "out" : critical_state_out}
    
    return critical_state

def evaluate_cacsade(x, inlet_stagnation, p_out,  loss_coefficients, fluid, geometry, m_ref, v0):
    
    p0_in = inlet_stagnation["p0_in"]
    h0_in = inlet_stagnation["h0_in"]
    
    v_crit_out = x[0]
    p_crit_out = x[1]
    beta_crit_out = x[2]
    v_out = x[3]
    beta_out  = x[4]
        
    Y_throat = loss_coefficients["throat"]
    Y_out = loss_coefficients["out"]
    
    A_throat = geometry["A_throat"]
    A_out = geometry["A_out"]
    
    # Calculate critical state
    critical_state = compute_critical_state(v_crit_out, p_crit_out, beta_crit_out, h0_in, p0_in, v0, geometry, loss_coefficients, fluid)
    
    # Calculate exit state
    input_exit = [v_out, p_out, beta_out]
    mass_flow_out, Ma_out, Y_def_out = evaluate_plane(input_exit, h0_in, p0_in, A_out, fluid)
    d_out = mass_flow_out/(v_out*np.cos(beta_out*np.pi/180)*geometry["A_out"])

    # Evaluate deviation model
    beta_crit = get_exit_flow_angle_aungier(critical_state["out"]["Ma"], critical_state["throat"]["Ma"], geometry)
    
    if Ma_out < critical_state["out"]["Ma"]:
        beta = get_exit_flow_angle_aungier(Ma_out, critical_state["throat"]["Ma"], geometry)
    else:
        beta = np.arccos(critical_state["throat"]["mass_flow"]/(d_out*v_out*geometry["A_out"]))*180/np.pi
    
    # Evaluate residuals
    Y_crit_error = (critical_state["out"]["Y"] - Y_out)/Y_out
    subsonic_solution = max(0, critical_state["out"]["Ma"]-critical_state["throat"]["Ma"])
    m_crit_error = (critical_state["out"]["mass_flow"] - critical_state["throat"]["mass_flow"])/m_ref + subsonic_solution*np.sign(critical_state["out"]["mass_flow"] - critical_state["throat"]["mass_flow"])
    beta_crit_error = np.cos(beta_crit*np.pi/180) - np.cos(beta_crit_out*np.pi/180)
    beta_error = np.cos(beta*np.pi/180) - np.cos(beta_out*np.pi/180)
    Y_error = (Y_def_out - Y_out)/Y_out
    
    residuals = np.array([Y_crit_error, m_crit_error, beta_crit_error, beta_error, Y_error])
    
    results = {"out" : {}, "critical" : critical_state}
    results["out"]["mass_flow"] = mass_flow_out
    results["out"]["beta"] = beta_out
    results["out"]["Ma"] = Ma_out
    
    return [residuals, results]

v_crit_out = 0.7
p_crit_out = 0.6
beta_crit_out = np.arccos(geometry["A_throat"]/geometry["A_out"])
v_out = 0.6
beta_out = 70*np.pi/180
x0 = [v_crit_out, p_crit_out, beta_crit_out, v_out, beta_out]

N = 30
pressure_ratio = np.linspace(2.0, 1.8, N)

beta_vec = np.zeros(N)
m_vec = np.zeros(N)
Ma_vec = np.zeros(N)

for i in range(N):

    p_out = p0_in/pressure_ratio[i]
    
    print('\n')
    print(f'Iteration: {i}')
    print(f"Exit pressure: {p_out}")
    print(f"Pressure ratio: {pressure_ratio[i]}")

    # Calculate v0 and m_ref
    fluid.update(cp.PSmass_INPUTS, p_out, s_in)
    h_is = fluid.hmass()
    d_is = fluid.rhomass()
    v0 = np.sqrt(2*(h0_in-h_is))
    m_ref = d_is*v0*geometry["A_out"]
    print(v0)
    
    if i > 0:
        x0[0] /= v0
        x0[1] /= p0_in
        x0[2] /= (180/np.pi)
        x0[3] /= v0
        x0[4] /= (180/np.pi)
        
    print([x0[0]*v0, x0[1]*p0_in, x0[2]*180/np.pi, x0[3]*v0, x0[4]*180/np.pi])    
    residuals = lambda x: evaluate_cacsade([x[0]*v0, x[1]*p0_in, x[2]*180/np.pi, x[3]*v0, x[4]*180/np.pi], inlet_stagnation, p_out,  loss_coefficients, fluid, geometry, m_ref, v0)[0]
    solution = optimize.root(residuals, x0, method = 'lm', tol = 1e-8)
    x0 = solution.x
    x0[0] *= v0
    x0[1] *= p0_in
    x0[2] *= 180/np.pi
    x0[3] *= v0
    x0[4] *= 180/np.pi
    print(x0)
            
    residuals, results = evaluate_cacsade([x0[0], x0[1], x0[2], x0[3], x0[4]], inlet_stagnation, p_out,  loss_coefficients, fluid, geometry, m_ref, v0)
    print(f"Max residual: {max(residuals)}")
    print(f"Critical mach throat: {results['critical']['throat']['Ma']}")
    print(f"Critical mach out: {results['critical']['out']['Ma']}")
    print(f"Exit mach: {results['out']['Ma']}")

    
    beta_vec[i] = results["out"]["beta"]
    m_vec[i] = results["out"]["mass_flow"]
    Ma_vec[i] = results["out"]["Ma"]
    
fig, ax = plt.subplots()
ax.plot(pressure_ratio, beta_vec)

fig1, ax1 = plt.subplots()
ax1.plot(pressure_ratio, m_vec)

fig2, ax2 = plt.subplots()
ax2.plot(Ma_vec, beta_vec)

    
    