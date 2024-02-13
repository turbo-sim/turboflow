# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:48:12 2024

@author: lasseba
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp as cp
from scipy import optimize


p0_in = 1e5
T0_in = 300
fluid_name = 'air'

geometry = {"A_throat" : 0.025,
            "A_out" : 0.075}

Ma_crit_exit = 0.85
Y_out = 0.1
Y_throat = 0.1

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

def calculate_residual_derivative(v_out, p_out, v0, h0_in, Y):

    # This function should only take one argument (v_out)
    return optimize.approx_fprime(v_out, lambda v: calculate_residual(v, p_out, v0, h0_in, Y))

def calculate_residual(v_out, p_out, v0, h0_in, Y):
        
    # Scale variables
    v_out = v_out*v0
    v_out = v_out if np.isscalar(v_out) else v_out[0]
    
    # Calculate exit enthalpy
    h_out = h0_in - 0.5*v_out**2
    
    # Calculate exit total pressure
    fluid.update(cp.HmassP_INPUTS, h_out, p_out)
    s_out = fluid.smass()
    fluid.update(cp.HmassSmass_INPUTS, h0_in, s_out)
    p0_out = fluid.p()
    
    Y_def = (p0_in - p0_out)/(p0_out - p_out)
    
    residual = Y_def - Y
        
    return residual

def calculate_velocity(p_out, v0, h0_in, Y):
    
    solution = optimize.root_scalar(calculate_residual, args = (p_out, v0, h0_in, Y), x0 = 0.5, fprime=calculate_residual_derivative)
    
    return solution.root

def calculate_critical_throat(p_throat, h0_in, Y):
    
    # Scale p_throat
    p_throat *= p0_in
    p_throat = p_throat if np.isscalar(p_throat) else p_throat[0]
    
    # Calculate spouting velocity
    fluid.update(cp.PSmass_INPUTS, p_throat, s_in)
    h_isentropic = fluid.hmass()
    v0_throat = np.sqrt(2*(h0_in-h_isentropic))
    
    v_throat = calculate_velocity(p_throat, v0_throat, h0_in, Y)    
    v_throat *= v0_throat
    v_throat = v_throat if np.isscalar(v_throat) else v_throat[0]

    h_throat = h0_in - 0.5*v_throat**2
    fluid.update(cp.HmassP_INPUTS, h_throat, p_throat)
    d_throat = fluid.rhomass()
    a_throat = fluid.speed_sound()
    
    Ma_throat = v_throat/a_throat
    m_throat = d_throat*v_throat*geometry["A_throat"]

    return m_throat, Ma_throat

def calculate_mass_flow(p_throat, h0_in, Y):
    mass_flow, Ma = calculate_critical_throat(p_throat, h0_in, Y)
    return -mass_flow

def calculate_critical_mass_flow(h0_in, Y_throat, Y_out):
    
    solution = optimize.minimize(calculate_mass_flow, 0.5, args = (h0_in, Y_throat), bounds = [(0.1, 1.0)])
    mass_flow_throat, Ma_throat = calculate_critical_throat(solution.x, h0_in, Y_throat)
    
    
    
    return mass_flow_throat, Ma_throat

N = 100
Ma_exit = np.linspace(0, 1.2, N)
beta = np.zeros(N)
mass_flow = np.zeros(N)

pressure_ratio = np.linspace(1.2, 4, N)
p_vec = p0_in/pressure_ratio
fluid = cp.AbstractState('HEOS', fluid_name)

# Calculate inlet state
fluid.update(cp.PT_INPUTS, p0_in, T0_in)
h0_in = fluid.hmass()
s_in = fluid.smass()

# Calculate critical mass flow rate
m_crit, Ma_crit_throat = calculate_critical_mass_flow(h0_in, Y_throat)

for i in range(N):
    
    print(f"Iteration: {i}")
    
    # Update exit pressure
    p_out = p_vec[i]
    
    # Calculate isentropic exit state
    fluid.update(cp.PSmass_INPUTS, p_out, s_in)
    h_isentropic = fluid.hmass()
    
    # Calculate spouting velocity
    v0 = np.sqrt(2*(h0_in-h_isentropic))
    
    # calculate exit velocity
    v_out = calculate_velocity(p_out, v0, h0_in, Y_out)
    v_out *= v0
    v_out = v_out if np.isscalar(v_out) else v_out[0]
    
    # Calculate exit state
    h_out = h0_in - 0.5*v_out**2
    fluid.update(cp.HmassP_INPUTS, h_out, p_out)
    a_out = fluid.speed_sound()
    d_out = fluid.rhomass()
    Ma_exit = v_out/a_out
       
    if Ma_exit < Ma_crit_exit:
        beta[i] = get_exit_flow_angle_aungier(Ma_exit, Ma_crit_throat, geometry)
    else:
        beta[i] = np.arccos(m_crit/(d_out*v_out*geometry["A_out"]))*180/np.pi
        
    mass_flow[i] = d_out*v_out*np.cos(beta[i]*np.pi/180)*geometry["A_out"]
    
def plane(x, h0_in, p0_in, A):
    
    v =x[0]
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

    
fig, ax = plt.subplots()
ax.plot(pressure_ratio, beta)

fig1, ax1 = plt.subplots()
ax1.plot(pressure_ratio, mass_flow)
