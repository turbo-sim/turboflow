# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:31:16 2023

@author: laboan
"""

import numpy as np
import pandas as pd
from scipy import optimize
import CoolProp as CP
from ..properties import FluidCoolProp_2Phase
from ..math import smooth_max
from . import loss_model as lm
from . import deviation_model as dm
from scipy.optimize._numdiff import approx_derivative

def evaluate_cascade_series(independant_variables, fluid, angular_speed, geometry, model_options, references):
    
    n_cascades = geometry["n_cascades"]
    
    # Initialize cascades data   
    
    # initialize residual arrays
    residuals_values = np.array([])
    residuals_keys = np.array([])
    
    for i in range(n_cascades):
        
        # Update angular speed
        angular_speed_cascade = angular_speed*(i % 2)
        
        # Update cascade geometry
        
        # Evaluate current cascade
        cascade_inlet_input = {"h0" : enthalpy_stag_in, "s" : entropy_in, "alpha" : flow_angle_abs_in, "v" : velocity_abs_in}
        cascade_throat_input = {"w" : velocity_rel_throat, "beta" : flow_angle_rel_throat, "entropy" : entropy_throat}
        cascade_exit_input = {"w" : velocity_rel_out, "beta" : flow_angle_rel_out, "entropy" : entropy_out}
        critical_cascade_input = {"v*_in" : velocity_abs_crit_in, "w*_out" : velocity_rel_crit_out, "s*_out" : entropy_crit_out}
        cascade_residual_values, cascade_residual_keys = evaluate_cascade(cascade_inlet_input, cascade_throat_input, cascade_exit_input, fluid, cascade_geometry, angular_speed_cascade, cascades_data, model_options, references)
        
        # Add cascade residuals to residual arrays
        residuals_values = np.concatenate((residuals_values, cascade_residual_values))
        residuals_keys = np.concatenate((residuals_keys, cascade_residual_keys))
        
        # Calculate input of next cascade (Assume no change in density)
        density_exit = cascade_data["d"][-1]
        velocity_abs_m_exit = cascade_data["v_m"][-1]
        velocity_abs_theta_exit = cascade_data["v_theta"][-1]
        enthalpy_stag_exit = cascade_data["h0"][-1]
        
        velocity_abs_theta_in = velocity_abs_theta_exit*radius_exit/radius_in
        velocity_abs_m_in = velocity_abs_m_exit*area_exit/area_in 
        velocity_abs_in = np.sqrt(velocity_abs_theta_in**2 + velocity_abs_m_in**2)
        flow_angle_abs_in = np.tan(velocity_abs_theta_in/velocity_abs_m_in)
        enthalpy_stag_in = enthalpy_stag_exit
        enthalpy_in = enthalpy_stag_in-0.5*velocity_abs_in**2
        density_in = density_exit 
        stagnation_properties = fluid.compute_properties_meanline(CP.DmassHmass_INPUTS, density_in, enthalpy_in)
        entropy_in = stagnation_properties["s"] 
        
        
    
    

def evaluate_cascade(cascade_inlet_input, cascade_throat_input, cascade_exit_input, critical_cascade_input, fluid, geometry, angular_speed, cascades_data, model_options, references):
    
    # Define model options
    displacement_thickness = model_options.get("displacement_thickness", 0)
    loss_model = model_options.get("loss_model", 'benner')
    choking_condition = model_options.get("choking_condition", 'deviation')
    deviation_model = model_options.get("deviation_model", 'aungier')
    
    # Load reference values
    m_ref = references["m_ref"]
    delta_ref = references["delta_ref"]
    
    # Define residual array and residual keys array
    residuals_cascade = np.array([])
    keys_cascade = np.array([])
    
    # Evaluate inlet plane
    inlet_plane = evaluate_inlet(cascade_inlet_input, fluid, geometry, angular_speed, delta_ref)
    cascades_data["plane"].loc[len(cascades_data["plane"])] = inlet_plane
    
    # Evaluate throat plane
    cascade_throat_input["rothalpy"] = inlet_plane["rothalpy"]
    throat_plane = evaluate_exit(cascade_throat_input, fluid, geometry, inlet_plane, angular_speed, displacement_thickness, loss_model)
    cascades_data["plane"].loc[len(cascades_data["plane"])] = throat_plane

    # Evaluate exit plane
    cacsade_exit_input["rothalpy"] = inlet_plane["rothalpy"]
    exit_plane = evaluate_exit(cascade_exit_input, fluid, geometry, inlet_plane, angular_speed, displacement_thickness, loss_model)
    cascades_data["plane"].loc[len(cascades_data["plane"])] = exit_plane

    # Add loss coefficient error to residual array
    residuals_cascade = np.append(residuals_cascade, [throat_plane["Y_err"], exit_plane["Y_err"]])
    keys_cascade = np.append(keys_cascade, ["Y_err_throat", "Y_err_exit"])
    
    # Add mass flow rate error
    residuals_cascade = np.append(residuals_cascade, [inlet_plane["m"]-throat_plane["m"], inlet_plane["m"]-exit_plane["m"]]/m_ref)
    keys_cascade = np.append(keys_cascade, ["m_err_throat", "m_err_exit"])
    
    # Calculate critical state
    critical_cascade_input["h0_in"] = cascade_inlet_input["h0"]
    critical_cascade_input["s_in"] = cascade_inlet_input["s"]
    critical_cascade_input["alpha_in"] = cascade_inlet_input["alpha"]
    x_crit = np.array([critical_cascade_input["v*_in"], critical_cascade_input["w*_out"], critical_cascade_input["s*_out"]])
    res_critical, critical_state = evaluate_lagrange_gradient(x_crit, critical_cascade_input, fluid, geometry, angular_speed, model_options, references)
    
    # Add final closing equation (deviaton model or mach error)
    if choking_condition == 'deviation':
        
        density_correction = throat_plane["d"]/critical_state["d"] # density correction due to nested finite differences
        res = calculate_deviation_equation(geometry, critical_state, exit_plane, density_correction, deviation_model)
       
    elif choking_condition == 'mach_critical' or choking_condition == 'mach_unity':
        if choking_condition == 'mach_unity':
            critical_state["Marel"] = 1
        res = calculate_mach_equation(critical_state["Ma"], exit_plane["Marel"], throat_plane["Marel"])
        density_correction = np.nan
    else:
        raise Exception("choking_condition must be 'deviation', 'mach_critical' or 'mach_unity'")    
    residuals_cascade = np.append(residuals_cascade, res)
    keys_cascade = np.append(keys_cascade, choking_condition)
    
    return residuals_cascade, keys_cascade
        
def calculate_mach_equation(Ma_crit, Ma_exit, Ma_throat, alpha = -100):
    
    xi = np.array([Ma_crit, Ma_exit])
    
    actual_mach = smooth_max(xi, method="boltzmann", alpha = alpha)
    
    res = Ma_throat - actual_mach
    
    return res
    
def calculate_deviation_equation(geometry, critical_state, exit_plane, density_correction, deviation_model):
    
    # Load cascade geometry
    metal_angle_out = geometry["theta_out"]
    A_out = geometry["A_out"]
    opening = geometry["o"]
    pitch = geometry["s"]
    
    # Load calculated critical condition
    m_crit = critical_state["m"]
    Ma_crit = critical_state["Marel"]
    
    # Load exit plane
    Ma = exit_plane["Marel"]
    density = exit_plane["d"]
    velocity_rel = exit_plane["w"]
    flow_angle_rel = exit_plane["beta"]
    blockage = exit_plane["blockage"]
    
    if Ma < Ma_crit:
        flow_angle_model = dm.deviation(deviation_model, metal_angle_out, opening, pitch, Ma, Ma_crit)
    else:
        flow_angle_model = np.arccos(m_crit/density/velocity_rel/A_out/blockage*density_correction)*180/np.pi
    # Compute error of guessed beta and deviation model
    res = np.cos(flow_angle_model*np.pi/180)-np.cos(flow_angle_rel)
    
    return res

def evaluate_inlet(cascade_inlet_input, fluid, geometry, angular_speed, delta_ref):
    
    # Load cascade inlet input
    enthalpy_stag = cascade_inlet_input["h0"]
    entropy = cascade_inlet_input["s"]
    velocity_abs = cascade_inlet_input["v"]
    flow_angle_abs = cascade_inlet_input["alpha"]
    
    # Load geometry
    radius = geometry["r_in"]
    chord = geometry["chord"]
    area = geometry["A_in"]
    
    # Calculate velocity triangles
    blade_speed = radius*angular_speed
    velocity_triangle = evaluate_velocity_triangle_in(blade_speed, velocity_abs, flow_angle_abs)
    velocity_rel = velocity_triangle["w"]
    velocity_rel_m = velocity_triangle["w_m"]

    # Calculate static properties
    enthalpy = enthalpy_stag-0.5*velocity_abs**2
    static_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, enthalpy, entropy)
    density = static_properties["d"]
    viscosity = static_properties["mu"]
    speed_sound = static_properties["a"]
    
    # Calculate stagnation properties
    stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, enthalpy_stag, entropy)
    stagnation_properties = add_string_to_keys(stagnation_properties, '0')
    
    # Calculate relatove stagnation properties
    enthalpy_stag_rel = enthalpy + 0.5*velocity_rel**2
    relative_stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, enthalpy_stag_rel, entropy)
    relative_stagnation_properties = add_string_to_keys(relative_stagnation_properties, '0rel')
    
    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = velocity_abs / speed_sound
    Ma_rel = velocity_rel / speed_sound
    Re = density * velocity_rel * chord / viscosity
    m = density * velocity_rel_m * area
    rothalpy = enthalpy_stag_rel-0.5*blade_speed**2
    
    # Calculate inlet displacement thickness to blade height ratio based on reference vale
    delta = delta_ref * Re**(-1/7)
    
    # Store result
    plane = {**velocity_triangle, **static_properties, **stagnation_properties, **relative_stagnation_properties}
    plane["Ma"] = Ma
    plane["Marel"] = Ma_rel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = delta
    plane["rothalpy"] = rothalpy
    plane["Y_err"] = np.nan
    plane["Y"] = np.nan
    plane["Y_p"] = np.nan
    plane["Y_cl"] = np.nan
    plane["Y_s"] = np.nan
    plane["Y_te"] = np.nan
    plane["Y_inc"] = np.nan
    plane["blockage"] = np.nan
    
    return plane

def evaluate_exit(cascade_exit_input, fluid, geometry, inlet_plane, angular_speed, displacement_thickness, loss_model):
    
    # Load cascade exit variables
    velocity_rel = cascade_exit_input["w"]
    flow_angle_rel = cascade_exit_input["beta"]
    entropy = cascade_exit_input["entropy"]
    rothalpy = cascade_exit_input["rothalpy"]
    
    # Load geometry
    radius = geometry["r_out"]
    area = geometry["A_out"]
    chord = geometry["chord"]
    opening = geometry["o"]
    
    # Calculate velocity triangles
    blade_speed = angular_speed*radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, velocity_rel, flow_angle_rel)
    velocity_abs = velocity_triangle["v"]
    velocity_rel_m = velocity_triangle["w_m"]
    
    # Calculate static properties
    enthalpy = rothalpy + 0.5*blade_speed**2-0.5*velocity_rel**2
    static_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, enthalpy, entropy)
    density = static_properties["d"]
    viscosity = static_properties["mu"]
    speed_sound = static_properties["a"]
    
    # Calculate stagnation properties
    enthalpy_stag = enthalpy + 0.5*velocity_abs**2
    stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, enthalpy_stag, entropy)
    stagnation_properties = add_string_to_keys(stagnation_properties, '0')
    
    # Calculate relatove stagnation properties
    enthalpy_stag_rel = enthalpy + 0.5*velocity_rel**2
    relative_stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, enthalpy_stag_rel, entropy)
    relative_stagnation_properties = add_string_to_keys(relative_stagnation_properties, '0rel')
    
    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = velocity_abs / speed_sound
    Ma_rel = velocity_rel / speed_sound
    Re = density * velocity_rel * chord / viscosity
    m = density * velocity_rel_m * area
    rothalpy = enthalpy_stag_rel-0.5*blade_speed**2

    # Account for blockage effect due to boundary layer displacement thickness    
    if displacement_thickness == None:
        displacement_thickness = 0.048/Re**(1/5)*0.9*chord
    correction = 1-2*displacement_thickness/opening
    m *= correction
    
    # Evaluate loss coefficient
    loss_model_input = {"geometry" : geometry,
                    "flow" : {},
                    "loss_model" : loss_model,
                    "type" : "stator"*(blade_speed == 0)+"rotor"*(blade_speed != 0)}
    
    loss_model_input["flow"]["p0_rel_in"] = inlet_plane["p0rel"]
    loss_model_input["flow"]["p0_rel_out"] = relative_stagnation_properties["p0rel"]
    loss_model_input["flow"]["p_in"] = inlet_plane["p"]
    loss_model_input["flow"]["p_out"] = static_properties["p"]
    loss_model_input["flow"]["beta_out"] = flow_angle_rel
    loss_model_input["flow"]["beta_in"] = inlet_plane["beta"]
    loss_model_input["flow"]["Ma_rel_in"] = inlet_plane["Marel"]
    loss_model_input["flow"]["Ma_rel_out"] = Ma_rel
    loss_model_input["flow"]["Re_in"] = inlet_plane["Re"]
    loss_model_input["flow"]["Re_out"] = Re
    loss_model_input["flow"]["delta"] = inlet_plane["delta"]
    loss_model_input["flow"]["gamma_out"] = static_properties["gamma"]
    
    Y, Y_err, Y_info = evaluate_loss_model(loss_model_input)
    
    # Store result
    plane = {**velocity_triangle, **static_properties, **stagnation_properties, **relative_stagnation_properties}
    
    plane["Ma"] = Ma
    plane["Marel"] = Ma_rel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = np.nan # Not relevant for exit/throat plane
    plane["rothalpy"] = rothalpy
    plane["Y_err"] = Y_err
    plane["Y"] = Y
    plane["Y_p"] = Y_info["Profile"]
    plane["Y_cl"] = Y_info["Clearance"]
    plane["Y_s"] = Y_info["Secondary"]
    plane["Y_te"] = Y_info["Trailing"]
    plane["Y_inc"] = Y_info["Incidence"] 
    plane["blockage"] = correction

    return plane, Y_info

def evaluate_lagrange_gradient(x_crit, critical_cascade_input, fluid, geometry, angular_speed, model_options, references):
    """
    Evaluate the gradient of the Lagrange function of the critical mass flow rate function.

    Args:
        x (numpy.ndarray): Array containing [v_in*, V_throat*, s_throat*].
        critical_cascade_data (dict): Dictionary containing critical cascade data.

    Returns:
        numpy.ndarray: Residuals of the Lagrange function.

    This function evaluates the gradient of the Lagrange function of the critical mass flow rate function.
    It calculates the Lagrange multipliers explicitly and returns the residuals of the Lagrange gradient.

    Note:
        The function assumes specific structures in the critical_cascade_data dictionary.
        Please ensure the required keys are present for accurate evaluation.

    Example:
        # Define critical_cascade_data with required parameters
        critical_cascade_data = {
            "fixed_params": {"m_ref": 1.0},
            ...
        }

        # Define initial degrees of freedom in x
        x = np.array([...])

        # Evaluate the Lagrange gradient
        lagrange_grad = evaluate_lagrange_gradient(x, critical_cascade_data)
    """
    
    # Load reference values
    m_ref = references["m_ref"]
    
    # Define critical state dictionary to store information 
    critical_state = {}

    # Evaluate the current cascade at critical conditions
    f0 = evaluate_critical_cascade(x_crit, critical_cascade_input, fluid, geometry, angular_speed, critical_state,  model_options, references)

    # Evaluate the Jacobian of the evaluate_critical_cascade function
    J = compute_critical_cascade_jacobian(x_crit, critical_cascade_input, fluid, geometry, angular_speed, critical_state, model_options, references, f0) 
    
    # Rename gradients
    a11, a12, a21, a22, b1, b2 = J[1, 0], J[2, 0], J[1, 1+1], J[2, 1+1], -1 * J[0, 0], -1 * J[0, 1+1] # For isentropic

    # Calculate the Lagrange multipliers explicitly
    l1 = (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21)
    l2 = (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)

    # Evaluate the last equation
    df, dg1, dg2 = J[0, 2-1], J[1, 2-1], J[2, 2-1] # for isentropic
    grad = (df + l1 * dg1 + l2 * dg2) / m_ref

    # Return last 3 equations of the Lagrangian gradient (df/dx2+l1*dg1/dx2+l2*dg2/dx2 and g1, g2)
    g = f0[1:]  # The two constraints
    res = np.insert(g, 0, grad)

    return res, critical_state

def evaluate_critical_cascade(x_crit, critical_cascade_input, fluid, geometry, angular_speed, critical_state, model_options, references):
   
    # Load reference values
    m_ref = references["m_ref"]
    v0 = references["v0"]
    s_ref = references["s_ref"]
    delta_ref = references["delta_ref"]
    
    # Load geometry
    theta_out = geometry["theta_out"]
    
    # Load cinput for critical cascade
    entropy_in = critical_cascade_input["s_in"]
    enthalpy_stag = critical_cascade_input["h0_in"]
    flow_angle_abs = critical_cascade_input["alpha_in"]
   
    velocity_abs_in, velocity_rel_throat, entropy_throat = x_crit[0] * v0, x_crit[1] * v0, x_crit[2] * s_ref
    
    # Define model options
    displacement_thickness = model_options.get("displacement_thickness", 0)
    loss_model = model_options.get("loss_model", 'benner')
    
    # Evaluate inlet plane
    critical_inlet_input = {"v" : velocity_abs_in, "s" : entropy_in, "h0_in" : enthalpy_stag, "alpha" : flow_angle_abs}
    inlet_plane = evaluate_inlet(critical_inlet_input, fluid, geometry, angular_speed, delta_ref)
    
    # Evaluate throat plane
    critical_exit_input = {"w" : velocity_rel_throat, "s" : entropy_throat, "beta" : theta_out, "rothalpy" : inlet_plane["rothalpy"]}
    throat_plane = evaluate_exit(critical_exit_input, fluid, geometry, inlet_plane, angular_speed, displacement_thickness, loss_model)
    
    # Add residuals
    residuals = np.array([(inlet_plane["m"] - throat_plane["m"]) / m_ref, throat_plane["Y_err"]])
    
    critical_state["m"] = throat_plane["m"]
    critical_state["Marel"] = throat_plane["Marel"]
    critical_state["d"] = throat_plane["d"]

    output = np.insert(residuals, 0, throat_plane["m"])
    
    return output


def compute_critical_cascade_jacobian(x, critical_cascade_input, fluid, geometry, angular_speed, critical_state, model_options, references, f0):

    eps = 1e-3 * x
    J = approx_derivative(evaluate_critical_cascade, x, method='2-point', f0 = f0, abs_step=eps,
                         args=(critical_cascade_input, fluid, geometry, angular_speed, critical_state, model_options, references))

    return J


    
def evaluate_velocity_triangle_in(u, v, alpha):
    """
    Compute the velocity triangle at the inlet of the cascade.

    Args:
        u (float): Blade speed.
        v (float): Absolute velocity.
        alpha (float): Absolute flow angle (in radians).

    Returns:
        dict: Dictionary containing the following properties:
            v (float): Absolute velocity.
            v_m (float): Meridional component of absolute velocity.
            v_t (float): Tangential component of absolute velocity.
            alpha (float): Absolute flow angle.
            w (float): Relative velocity magnitude.
            w_m (float): Meridional component of relative velocity.
            w_t (float): Tangential component of relative velocity.
            beta (float): Relative flow angle (in radians).
    """

    # Absolute velocities
    v_t = v * np.sin(alpha)
    v_m = v * np.cos(alpha)

    # Relative velocities
    w_t = v_t - u
    w_m = v_m
    w = np.sqrt(w_t ** 2 + w_m ** 2)

    # Relative flow angle
    beta = np.arctan(w_t / w_m)
    
    # Store in dict
    vel_in = {"v" : v, "v_m" : v_m, "v_t" : v_t, "alpha" : alpha,
              "w" : w, "w_m" : w_m, "w_t" : w_t, "beta" : beta}

    return vel_in

def evaluate_velocity_triangle_out(u, w, beta):
    """
    Compute the velocity triangle at the outlet of the cascade.

    Args:
        u (float): Blade speed.
        w (float): Relative velocity.
        beta (float): Relative flow angle (in radians).

    Returns:
        dict: Dictionary containing the following properties:
            v (float): Absolute velocity.
            v_m (float): Meridional component of absolute velocity.
            v_t (float): Tangential component of absolute velocity.
            alpha (float): Absolute flow angle (in radians).
            w (float): Relative velocity magnitude.
            w_m (float): Meridional component of relative velocity.
            w_t (float): Tangential component of relative velocity.
            beta (float): Relative flow angle.
    """

    # Relative velocities
    w_t = w * np.sin(beta)
    w_m = w * np.cos(beta)

    # Absolute velocities
    v_t = w_t + u
    v_m = w_m
    v = np.sqrt(v_t ** 2 + v_m ** 2)

    # Absolute flow angle
    alpha = np.arctan(v_t / v_m)
    
    # Store in dict
    vel_out = {"v" : v, "v_m" : v_m, "v_t" : v_t, "alpha" : alpha,
              "w" : w, "w_m" : w_m, "w_t" : w_t, "beta" : beta}

    return vel_out

def evaluate_loss_model(loss_model_input, is_throat=False):
    """
    Evaluate the loss according to both loss correlation and definition.
    Return the loss coefficient, error, and breakdown of losses.

    Args:
        cascade_data (dict): Data for the cascade.

    Returns:
        tuple: A tuple containing the loss coefficient, error, and additional information.
    """
    # Load necessary parameters
    lossmodel = loss_model_input["loss_model"]
    p0rel_in = loss_model_input["flow"]["p0_rel_in"]
    p0rel_out = loss_model_input["flow"]["p0_rel_out"]
    p_out = loss_model_input["flow"]["p_out"]

    # Compute the loss coefficient from its definition
    Y_def = (p0rel_in - p0rel_out) / (p0rel_out - p_out)

    # Compute the loss coefficient from the correlations
    Y, Y_info = lm.loss(loss_model_input, lossmodel, is_throat)

    # Compute loss coefficient error
    Y_err = Y_def - Y

    return Y, Y_err, Y_info

def add_string_to_keys(input_dict, suffix):
    """
    Add a suffix to each key in the input dictionary.

    Args:
        input_dict (dict): The input dictionary.
        suffix (str): The string to add to each key.

    Returns:
        dict: A new dictionary with modified keys.

    Example:
        >>> input_dict = {'a': 1, 'b': 2, 'c': 3}
        >>> add_string_to_keys(input_dict, '_new')
        {'a_new': 1, 'b_new': 2, 'c_new': 3}
    """
    return {f"{key}{suffix}": value for key, value in input_dict.items()}