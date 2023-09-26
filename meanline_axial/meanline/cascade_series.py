# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:41:43 2023

@author: laboan
"""

import numpy as np
import pandas as pd
from scipy import optimize
import CoolProp as CP
from scipy.optimize._numdiff import approx_derivative
from ..properties import FluidCoolProp_2Phase
from . import loss_model as lm

# Keys of the information that should be stored in cascades_data
keys_plane = ['v', 'v_m', 'v_t', 'alpha', 'w', 'w_m', 'w_t', 'beta', 'p', 'T', 'h',
       's', 'd', 'Z', 'a', 'mu', 'k', 'cp', 'cv', 'gamma', 'p0', 'T0', 'h0',
       's0', 'd0', 'Z0', 'a0', 'mu0', 'k0', 'cp0', 'cv0', 'gamma0', 'p0rel',
       'T0rel', 'h0rel', 's0rel', 'd0rel', 'Z0rel', 'a0rel', 'mu0rel', 'k0rel',
       'cp0rel', 'cv0rel', 'gamma0rel', 'Ma', 'Marel', 'Re', 'm', 'delta',
       'Y_err', 'Y']

keys_cascade = ["Y_tot", "Y_p", "Y_s", "Y_cl", "dh_s"]

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

def evaluate_cascade_series(x, cascades_data):
    """
    Evaluate the performance of a series of cascades.

    Args:
        x (numpy.ndarray): Array containing degrees of freedom for each cascade.
        cascades_data (dict): Dictionary containing parameters and configurations.

    Returns:
        numpy.ndarray: Array of residuals representing the performance of each cascade.

    This function iterates through a series of cascades, evaluates their performance,
    and calculates residuals. It uses the provided cascades_data to extract necessary
    parameters for the evaluation.

    Parameters in cascades_data:
        - BC (dict): Boundary conditions including p0_in, T0_in, p_out, fluid, alpha_in.
        - geometry (dict): Geometric parameters including n_cascades.
        - fixed_params (dict): Fixed parameters like m_ref, s_ref, v0, delta_ref.
        - loss_model (obj): Loss model object.
        - overall (dict): Overall parameters including blade_speed, m, eta_ts.

    Returns a numpy array of residuals, which indicate the performance of each cascade,
    including critical point and cascade evaluations.

    Note:
        The function assumes specific structures in the cascades_data dictionary.
        Please ensure the required keys are present for accurate evaluation.

    Raises:
        (SpecificException): Description of the specific error scenario.

    Example:
        # Define cascades_data with required parameters
        cascades_data = {
            "BC": {"p0_in": 1.0, "T0_in": 300.0, "p_out": 0.9, ...},
            "geometry": {"n_cascades": 3, ...},
            ...
        }

        # Define initial degrees of freedom in x
        x = np.array([...])

        # Evaluate cascade series
        residuals = evaluate_cascade_series(x, cascades_data)
    """
    # Load parameters from cascades_data
    p0_in = cascades_data["BC"]["p0_in"]
    T0_in = cascades_data["BC"]["T0_in"]
    p_out = cascades_data["BC"]["p_out"]
    fluid = cascades_data["BC"]["fluid"]
    alpha_in = cascades_data["BC"]["alpha_in"]
    n_cascades = cascades_data["geometry"]["n_cascades"]
    n_stages = cascades_data["fixed_params"]["n_stages"]
    h_out_s = cascades_data["fixed_params"]["h_out_s"]
    s_ref = cascades_data["fixed_params"]["s_ref"]
    v0 = cascades_data["fixed_params"]["v0"]
    m_ref = cascades_data["fixed_params"]["m_ref"]
    delta_ref = cascades_data["fixed_params"]["delta_ref"]
    lossmodel = cascades_data["loss_model"]
    u = cascades_data["overall"]["blade_speed"]
    n_dof = 5  # Number of degrees of freedom for each cascade except for v_in (used for keeping track on indices)
    n_dof_crit = 3  # Number of degrees of freedom for each critical cascade

    # Define vector for storing residuals
    residuals = np.array([])

    # Define degrees of freedom
    x_series = x[0:n_dof * n_cascades + 1]
    x_crit = x[n_dof * n_cascades + 1:]
    v_in = x_series[0] * v0
    x_series = np.delete(x_series, 0)

    # Ensure cascades_data elements are is empty
    df = pd.DataFrame(columns=keys_plane)
    cascades_data["plane"] = df
    df = pd.DataFrame(columns=keys_cascade)
    cascades_data["cascade"] = df

    # Construct data structure for critical point
    critical_cascade_data = {
        "BC": {"p0_in": p0_in, "T0_in": T0_in, "fluid": fluid, "alpha_in": alpha_in},
        "fixed_params": {"m_ref": m_ref, "s_ref": s_ref, "v0": v0, "delta_ref": delta_ref},
        "loss_model": lossmodel
    }

    for i in range(n_cascades):

        # Make sure blade speed is zero for stator
        u_cascade = u * (i % 2)
        
        # Fetch geometry for current cascade
        geometry = {key: values[i] for key, values in cascades_data["geometry"].items() if key != "n_cascades"}

        # Calculate critical mach number for current cascade with the given inlet conditions
        critical_cascade_data["u"] = u_cascade

        x_crit_cascade = x_crit[i * n_dof_crit:(i + 1) * n_dof_crit]

        crit_res = evaluate_lagrange_gradient(x_crit_cascade, critical_cascade_data, geometry)
        residuals = np.concatenate((residuals, crit_res))
        Ma_crit = critical_cascade_data["Ma_crit"]

        # Retrieve the DOF for the current cascade
        x_cascade = x_series[i * n_dof:(i + 1) * n_dof]

        # Scale DOF
        x_cascade[0] = x_cascade[0] * v0
        x_cascade[1] = x_cascade[1] * v0
        x_cascade[2] = x_cascade[2] * s_ref
        x_cascade[3] = x_cascade[3] * s_ref

        # Evaluate the current cascade at the given conditions
        cascade_res = evaluate_cascade(i, v_in, x_cascade, u_cascade, cascades_data, geometry, Ma_crit)

        # Change boundary conditions for next cascade (for critical mach calculation)
        critical_cascade_data["BC"]["p0_in"] = cascades_data["plane"]["p0rel"].values[-1]
        critical_cascade_data["BC"]["T0_in"] = cascades_data["plane"]["T0rel"].values[-1]
        critical_cascade_data["BC"]["alpha_in"] = cascades_data["plane"]["alpha"].values[-1]

        # Store cascade residuals
        residuals = np.concatenate((residuals, cascade_res))

    # Add exit pressure error to residuals
    p_calc = cascades_data["plane"]["p"].values[-1]
    p_error = (p_calc - p_out) / p0_in
    residuals = np.append(residuals, p_error)

    if n_stages != 0:
        # Calculate stage variables
        calculate_stage_parameters(cascades_data)

    # Store global variables
    cascades_data["overall"]["m"] = cascades_data["plane"]["m"].values[-1]

    stag_h = cascades_data["plane"]["h0"].values
    cascades_data["overall"]["eta_ts"] = (stag_h[0] - stag_h[-1]) / (stag_h[0] - h_out_s)

    loss_fracs = calculate_eta_drop_fractions(cascades_data)
    cascades_data["loss_fracs"] = loss_fracs
    cascades_data["cascade"] = pd.concat([cascades_data["cascade"], loss_fracs], axis=1)
    
    return residuals

def evaluate_cascade(i, v_in, x_cascade, u, cascades_data, geometry, Macrit):
    """
    Evaluate the performance of a cascade.

    Args:
        i (int): Index of the cascade in the series.
        v_in (float): Inlet velocity.
        x_cascade (numpy.ndarray): Array of degrees of freedom for the current cascade.
        u (float): Blade speed.
        cascades_data (dict): Dictionary containing parameters and configurations.
        Macrit (float): Critical Mach number.

    Returns:
        numpy.ndarray: Array of residuals representing the performance of the cascade.

    This function evaluates the performance of a single cascade. It calculates various properties
    and residuals, and stores the plane data in the cascades_data.

    Note:
        The function assumes specific structures in the cascades_data dictionary.
        Please ensure the required keys are present for accurate evaluation.

    Example:
        # Define cascades_data with required parameters
        cascades_data = {
            "BC": {"fluid": "Air", ...},
            "geometry": {"A_out": 1.0, "theta_out": 30.0, ...},
            ...
        }

        # Define initial degrees of freedom in x_cascade
        x_cascade = np.array([...])

        # Define other necessary parameters (v_in, u, Macrit)

        # Evaluate the cascade
        cascade_res = evaluate_cascade(1, 100, x_cascade, 500, cascades_data, 0.8)
    """
    # Load necessary parameters
    fluid = cascades_data["BC"]["fluid"]
    # geometry = {key: values[i] for key, values in cascades_data["geometry"].items() if key != "n_cascades"}
    A_out = geometry["A_out"]
    theta_out = geometry["theta_out"]
    m_ref = cascades_data["fixed_params"]["m_ref"]

    # Structure degrees of freedom
    vel_throat, vel_out, s_throat, s_out, beta_out = x_cascade[:5]

    x_throat = np.array([vel_throat, theta_out, s_throat])
    x_out = np.array([vel_out, beta_out, s_out])

    # Define residual array
    cascade_res = np.array([])

    # Define index for state dataframe
    inlet, throat, outlet = 3 * i, 3 * i + 1, 3 * i + 2

    # Evaluate inlet plane
    if i == 0:  # For first stator inlet plane
        inlet_plane = evaluate_first_inlet(v_in, cascades_data, u, geometry)
    else:  # For all other inlet planes
        exit_plane = cascades_data["plane"].loc[inlet - 1]
        inlet_plane = evaluate_inlet(cascades_data, u, geometry, exit_plane)

    cascades_data["plane"].loc[len(cascades_data["plane"])] = inlet_plane

    # Evaluate throat
    throat_plane, Y_info_throat = evaluate_outlet(x_throat, geometry, cascades_data, u, A_out, inlet_plane)
    cascades_data["plane"].loc[len(cascades_data["plane"])] = throat_plane

    Y_err_throat = throat_plane["Y_err"]
    cascade_res = np.append(cascade_res, Y_err_throat)

    # Evaluate exit
    exit_plane, Y_info_exit = evaluate_outlet(x_out, geometry, cascades_data, u, A_out, inlet_plane)
    cascades_data["plane"].loc[len(cascades_data["plane"])] = exit_plane
    Y_err_out = exit_plane["Y_err"]
    cascade_res = np.append(cascade_res, Y_err_out)

    # Compute mach number error
    # actual_mach = min(cascades_data["plane"]["Marel"].values[outlet], Macrit)
    alpha = -16
    actual_mach = (cascades_data["plane"]["Marel"].values[outlet] ** alpha + Macrit ** alpha) ** (1 / alpha)

    mach_err = cascades_data["plane"]["Marel"].values[throat] - actual_mach
    cascade_res = np.append(cascade_res, mach_err)

    # Add mass flow deviations to residuals
    cascade_res = np.concatenate((cascade_res, (cascades_data["plane"]["m"][inlet:outlet].values -
                                               cascades_data["plane"]["m"][throat:outlet + 1].values))) / m_ref

    # Add cascade information to cascade dataframe
    static_properties_isentropic_expansion = fluid.compute_properties_meanline(CP.PSmass_INPUTS, exit_plane["p"], inlet_plane["s"])
    dhs = exit_plane["h"] - static_properties_isentropic_expansion["h"]
    cascade_data = {
        "Y_tot": Y_info_exit["Total"],
        "Y_p": Y_info_exit["Profile"],
        "Y_s": Y_info_exit["Secondary"],
        "Y_cl": Y_info_exit["Clearance"],
        "dh_s": dhs
    }

    cascades_data["cascade"].loc[len(cascades_data["cascade"])] = cascade_data

    return cascade_res

def evaluate_first_inlet(v, cascades_data, u, geometry):
    """
    Evaluate first stator inlet plane.

    Args:
        v (float): Absolute velocity.
        cascades_data (dict): Dictionary containing data for cascades.
        u (float): Blade speed.
        geometry (dict): Dictionary containing geometry information.

    Returns:
        dict: Dictionary containing evaluated properties of the inlet plane.
    """
    # Load boundary conditions
    T0_in = cascades_data["BC"]["T0_in"]
    p0_in = cascades_data["BC"]["p0_in"]
    fluid = cascades_data["BC"]["fluid"]
    alpha_in = cascades_data["BC"]["alpha_in"]
    delta_ref = cascades_data["fixed_params"]["delta_ref"]  # Reference inlet displacement thickness

    # Load geometry
    A = geometry["A_in"]
    c = geometry["c"]

    # Evaluate velocity triangle
    vel_in = evaluate_velocity_triangle_in(u, v, alpha_in)
    w = vel_in["w"]
    w_m = vel_in["w_m"]

    # Stagnation properties
    stagnation_properties = fluid.compute_properties_meanline(CP.PT_INPUTS, p0_in, T0_in)
    stagnation_properties = add_string_to_keys(stagnation_properties, '0')
    h0 = stagnation_properties["h0"]

    # Static properties
    s = stagnation_properties["s0"]
    h = h0 - v**2 / 2
    static_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h, s)
    a = static_properties["a"]
    d = static_properties["d"]
    mu = static_properties["mu"]

    # Relative properties
    h0rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0rel, s)
    relative_stagnation_properties = add_string_to_keys(relative_stagnation_properties, '0rel')

    # Calculate mach, Reynolds and mass flow rate for cascade inlet
    Ma = v / a
    Marel = w / a
    Re = d * w * c / mu
    m = d * w_m * A

    # Calculate inlet displacement thickness to blade height ratio
    delta = delta_ref * Re**(-1/7)

    # Store result
    plane = {**vel_in, **static_properties, **stagnation_properties, **relative_stagnation_properties}

    plane["Ma"] = Ma
    plane["Marel"] = Marel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = delta
    plane["Y_err"] = np.nan
    plane["Y"] = np.nan

    return plane


def evaluate_inlet(cascades_data, u, geometry, exit_plane):
    
    """
    Evaluates the inlet plane of current cascade (except for first stator).

    Args:
        cascades_data (dict): Dictionary containing data for cascades.
        u (float): Blade speed.
        geometry (dict): Dictionary containing geometry information.
        exit_plane (dict): Dictionary containing properties of the exit plane.

    Returns:
        dict: Dictionary containing evaluated properties of the inlet plane.
    """
        
    # Load thermodynamic boundary conditions
    fluid     = cascades_data["BC"]["fluid"]
    delta_ref = cascades_data["fixed_params"]["delta_ref"] # Reference inlet displacement thickness (for loss calculations)

    # Load geometrical parameters
    A = geometry["A_in"]
    c = geometry["c"]

    # Evaluate velocity triangle
    # Absolute velocity and flow angle is equal the previous cascade
    v = exit_plane["v"]
    alpha = exit_plane["alpha"]
    vel_in = \
        evaluate_velocity_triangle_in(u, v, alpha)
    w = vel_in["w"]
    w_m = vel_in["w_m"]
    
    # Get thermodynamic state (equal exit of last cascade)
    # Static properties are equal the outlet of the previous cascade
    h = exit_plane["h"]
    s = exit_plane["s"]
    a = exit_plane["a"]
    d = exit_plane["d"]
    mu = exit_plane["mu"]
    
    # Relative stagnation properties change from the inlet of cascade i to the outlet of cascade i-1
    h0rel = h + w**2 / 2
    relative_stagnation_properites = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0rel, s)
    relative_stagnation_properites = add_string_to_keys(relative_stagnation_properites, '0rel')
    
    # Calcluate mach, Reonlds and mass flow rate for cascade inlet
    Ma = v / a
    Marel = w / a
    Re = d*w*c/mu;
    m = d*w_m*A
    
    # Calculate inlet displacement thickness to blade height ratio
    delta = delta_ref*Re**(-1/7)
    
    # Store result
    plane = exit_plane.copy()
    for key, value in {**vel_in,**relative_stagnation_properites}.items():
        plane[key] = value

    plane["Ma"] = Ma
    plane["Marel"] = Marel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = delta
    plane["Y_err"] = np.nan # Not relevant for inblet plane
    plane["Y"] = np.nan

    return plane    

def evaluate_outlet(x, geometry, cascades_data, u,  A, inlet_plane):
    
    """
     Evaluates the throat or exit plane of a cascade.
    
     Args:
         x (array-like): Degrees of freedom for the cascade.
         geometry (dict): Dictionary containing geometry information.
         cascades_data (dict): Dictionary containing data for cascades.
         u (float): Blade speed.
         A (float): Inlet area of the cascade.
         inlet_plane (dict): Properties of the inlet plane.
    
     Returns:
         tuple: A tuple containing:
             dict: Properties of the evaluated outlet plane.
             float: Efficiency loss coefficient (Y).
             dict: Additional information about the efficiency loss.
     """
    
    # Load thermodynamic boundaries
    fluid     = cascades_data["BC"]["fluid"]
    lossmodel = cascades_data["loss_model"]
    
    # Load geomerrical parameters 
    c = geometry["c"]
    
    # Load inlet state variables from inlet of current cascade
    p0rel_in = inlet_plane["p0rel"]
    h0rel    = inlet_plane["h0rel"]
    p_in     = inlet_plane["p"]
    beta_in  = inlet_plane["beta"]
    Marel_in = inlet_plane["Marel"]
    Re_in    = inlet_plane["Re"]
    delta    = inlet_plane["delta"]
    
    # Load degrees of freedom
    w    = x[0] # Relative velocity
    beta = x[1] # Relative flow angle
    s    = x[2] # Entropy
    
    # Calculate velocity triangle
    vel_out = \
        evaluate_velocity_triangle_out(u, w, beta)
    w = vel_out["w"]
    w_m = vel_out["w_m"]
    v = vel_out["v"]
    
    # Calculate thermodynamic state
    h = h0rel - w**2 / 2
    h0 = h + v**2 / 2
    
    static_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h, s)
    a = static_properties["a"]
    d = static_properties["d"]
    mu = static_properties["mu"]
    
    stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0, s)
    stagnation_properties = add_string_to_keys(stagnation_properties, '0')
    
    relative_stagnation_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0rel, s)
    relative_stagnation_properties = add_string_to_keys(relative_stagnation_properties, '0rel')

    # Calcluate mach, Reynolds and mass flow rate for cascade exit
    Ma = v / a
    Marel = w / a
    Re = d*w*c/mu;
    m = d*w_m*A
    
    # Evaluate loss coefficient
    cascade_data = {"geometry" : geometry,
                    "flow" : {},
                    "loss_model" : lossmodel,
                    "type" : "stator"*(u == 0)+"rotor"*(u != 0)}
    
    
    cascade_data["flow"]["p0_rel_in"] = p0rel_in
    cascade_data["flow"]["p0_rel_out"] = relative_stagnation_properties["p0rel"]
    cascade_data["flow"]["p_in"] = p_in
    cascade_data["flow"]["p_out"] = static_properties["p"]
    cascade_data["flow"]["beta_out"] = beta
    cascade_data["flow"]["beta_in"] = beta_in
    cascade_data["flow"]["Ma_rel_in"] = Marel_in
    cascade_data["flow"]["Ma_rel_out"] = Marel
    cascade_data["flow"]["Re_in"] = Re_in
    cascade_data["flow"]["Re_out"] = Re
    cascade_data["flow"]["delta"] = delta
    cascade_data["flow"]["gamma_out"] = static_properties["gamma"]
    
    Y, Y_err, Y_info = evaluate_loss_model(cascade_data)
    
    # Store result
    plane = {**vel_out, **static_properties, **stagnation_properties, **relative_stagnation_properties}
    
    plane["Ma"] = Ma
    plane["Marel"] = Marel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = np.nan # Not relevant for exit/throat plane
    plane["Y_err"] = Y_err
    plane["Y"] = Y
        
    return plane, Y_info
    
def evaluate_loss_model(cascade_data):
    """
    Evaluate the loss according to both loss correlation and definition.
    Return the loss coefficient, error, and breakdown of losses.

    Args:
        cascade_data (dict): Data for the cascade.

    Returns:
        tuple: A tuple containing the loss coefficient, error, and additional information.
    """
    # Load necessary parameters
    lossmodel = cascade_data["loss_model"]
    p0rel_in = cascade_data["flow"]["p0_rel_in"]
    p0rel_out = cascade_data["flow"]["p0_rel_out"]
    p_out = cascade_data["flow"]["p_out"]

    # Compute the loss coefficient from its definition
    Y_def = (p0rel_in - p0rel_out) / (p0rel_out - p_out)

    # Compute the loss coefficient from the correlations
    Y, Y_info = lm.loss(cascade_data, lossmodel)

    # Compute loss coefficient error
    Y_err = Y_def - Y

    return Y, Y_err, Y_info
 

    
def evaluate_lagrange_gradient(x, critical_cascade_data, geometry):
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
    
    # Load scale
    m_ref = critical_cascade_data["fixed_params"]["m_ref"]

    # Evaluate the current cascade at critical conditions
    f0 = evaluate_critical_cascade(x, critical_cascade_data, geometry)

    # Evaluate the Jacobian of the evaluate_critical_cascade function
    J = compute_critical_cascade_jacobian(x, critical_cascade_data, geometry, f0)

    # Rename gradients
    a11, a12, a21, a22, b1, b2 = J[1, 0], J[2, 0], J[1, 1], J[2, 1], -1 * J[0, 0], -1 * J[0, 1]

    # Calculate the Lagrange multipliers explicitly
    l1 = (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21)
    l2 = (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)

    # Evaluate the last equation
    df, dg1, dg2 = J[0, 2], J[1, 2], J[2, 2]
    grad = (df + l1 * dg1 + l2 * dg2) / m_ref

    # Return last 3 equations of the Lagrangian gradient (df/dx2+l1*dg1/dx2+l2*dg2/dx2 and g1, g2)
    g = f0[1:]  # The two constraints
    res = np.insert(g, 0, grad)

    return res

def compute_critical_cascade_jacobian(x, critical_cascade_data, geometry, f0):
    """
    Compute the Jacobian of the evaluate_critical_cascade function.

    Args:
        x (np.ndarray): The input vector.
        critical_cascade_data (dict): Data for the critical cascade.
        geometry (dict): The geometry of the cascade.
        f0 (callable): The function to evaluate at x.

    Returns:
        np.ndarray: The Jacobian matrix.
    """
    eps = 1e-3 * x
    # eps = 1e-3*np.maximum(1, abs(x))*np.sign(x)
    J = approx_derivative(evaluate_critical_cascade, x, method='2-point', f0=f0, abs_step=eps,
                         args=(critical_cascade_data, geometry))

    return J


def evaluate_critical_cascade(x_crit, critical_cascade_data, geometry):
    """
    Evaluate the critical cascade.

    Args:
        x_crit (numpy.ndarray): Array containing critical cascade degrees of freedom.
        critical_cascade_data (dict): Dictionary containing critical cascade data.
        geometry (dict): Dictionary containing geometric parameters.

    Returns:
        numpy.ndarray: Array containing mass flow rate and residuals of mass conservation and loss coeffiicient error

    This function evaluates the critical cascade. It loads necessary parameters, evaluates the inlet plane,
    evaluates the throat, calculates error of mass balance and loss coefficient, and stores the critical Mach number.

    Note:
        The function assumes specific structures in the critical_cascade_data and geometry dictionaries.
        Please ensure the required keys are present for accurate evaluation.

    Example:
        # Define critical_cascade_data and geometry with required parameters
        critical_cascade_data = {
            "u": 500,
            "fixed_params": {"v0": 100, "s_ref": 10, "m_ref": 2},
            ...
        }
        geometry = {
            "A_out": 1.0,
            "theta_out": 30.0,
            ...
        }

        # Define initial degrees of freedom in x_crit

        # Evaluate the critical cascade
        critical_cascade_result = evaluate_critical_cascade(x_crit, critical_cascade_data, geometry)
    """
    u = critical_cascade_data["u"]
    A_out = geometry["A_out"]
    theta_out = geometry["theta_out"]

    v0 = critical_cascade_data["fixed_params"]["v0"]
    s_ref = critical_cascade_data["fixed_params"]["s_ref"]
    m_ref = critical_cascade_data["fixed_params"]["m_ref"]

    v_in, w_throat, s_throat = x_crit[0] * v0, x_crit[1] * v0, x_crit[2] * s_ref

    inlet_plane = evaluate_first_inlet(v_in, critical_cascade_data, u, geometry)

    beta_throat = theta_out
    x = np.array([w_throat, beta_throat, s_throat])
    throat_plane, Y_info = evaluate_outlet(x, geometry, critical_cascade_data, u, A_out, inlet_plane)

    m_in, m_throat = inlet_plane["m"], throat_plane["m"]
    residuals = np.array([(m_in - m_throat) / m_ref, throat_plane["Y_err"]])

    critical_cascade_data["Ma_crit"] = throat_plane["Marel"]

    output = np.insert(residuals, 0, m_throat)

    return output

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


def generate_initial_guess(cascades_data, ig):
    """
    Generate an initial guess for the root-finder and design optimization.

    Args:
        cascades_data (dict): Data structure containing boundary conditions, geometry, etc.
        eta (float): Efficiency guess (default is 0.9).
        R (float): Degree of reaction guess (default is 0.4).
        Macrit (float): Critical Mach number guess (default is 0.92).

    Returns:
        numpy.ndarray: Initial guess for the root-finder.
    """
    # FIXME: fix docstring
    
    # Load initial giuess from ig
    R = ig["R"]
    eta = ig["eta"]
    Macrit = ig["Ma_crit"]
    
    # Load necessary parameters
    p0_in      = cascades_data["BC"]["p0_in"]
    T0_in      = cascades_data["BC"]["T0_in"]
    p_out      = cascades_data["BC"]["p_out"]
    fluid      = cascades_data["BC"]["fluid"]
    n_cascades = cascades_data["geometry"]["n_cascades"]    
    n_stages   = cascades_data["fixed_params"]["n_stages"]
    h_out_s    = cascades_data["fixed_params"]["h_out_s"]
    u          = cascades_data["overall"]["blade_speed"]
   
    # Inlet stagnation state
    stagnation_properties_in = fluid.compute_properties_meanline(CP.PT_INPUTS, p0_in, T0_in) 
    h0_in      = stagnation_properties_in["h"]
    d0_in      = stagnation_properties_in["d"]
    s_in       = stagnation_properties_in["s"]
    h_out = h0_in - eta*(h0_in-h_out_s) 
    
    # Exit static state for expansion with guessed efficiency
    static_properties_exit = fluid.compute_properties_meanline(CP.HmassP_INPUTS, h_out, p_out)
    s_out = static_properties_exit["s"]
    d_out = static_properties_exit["d"]
        
    # Same entropy production in each cascade
    s_cascades = np.linspace(s_in,s_out,n_cascades+1)
    
    # Initialize x0
    x0 = np.array([])
    x0_crit = np.array([])
    
    # Assume d1 = d01 for first inlet
    d1 = d0_in
    
    if n_stages != 0: 
        h_stages = np.linspace(h0_in, h_out, n_stages+1) # Divide enthalpy loss equally between stages
        h_in = h_stages[0:-1] # Enthalpy at inlet of each stage
        h_out = h_stages[1:] # Enthalpy at exit of each stage
        h_mid = h_out + R*(h_in-h_out) # Enthalpy at stator exit for each stage (computed by degree of reaction)
        h01 = h0_in
                
        for i in range(n_stages):
            
            # Iterate through each stage to calculate guess for velocity
            
            index_stator = i*2
            index_rotor  = i*2 + 1
            
            # 3 stations: 1: stator inlet 2: stator exit/rotor inlet, 3: rotor exit
            # Rename parameters
            A1 = cascades_data["geometry"]["A_in"][index_stator]
            A2 = cascades_data["geometry"]["A_out"][index_stator]
            A3 = cascades_data["geometry"]["A_out"][index_rotor]
            
            alpha1 = cascades_data["geometry"]["theta_in"][index_stator]
            alpha2 = cascades_data["geometry"]["theta_out"][index_stator]
            beta2  = cascades_data["geometry"]["theta_in"][index_rotor]
            beta3  = cascades_data["geometry"]["theta_out"][index_rotor]
            
            h1 = h_in[i]
            h2 = h_mid[i]
            h3 = h_out[i]
            
            s1 = s_cascades[i*2]
            s2 = s_cascades[i*2+1]
            s3 = s_cascades[i*2+2]
            
            # Condition at stator exit
            h02 = h01
            v2 = np.sqrt(2*(h02-h2))
            vel2_data = evaluate_velocity_triangle_in(u, v2, alpha2)
            w2 = vel2_data["w"]
            h0rel2 = h2 + 0.5*w2**2
            x0 = np.append(x0, np.array([v2, v2, s2, s2, alpha2]))
            
            # Critical condition at stator exit
            static_properties_2 = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h2, s2)
            a2 = static_properties_2["a"]
            d2 = static_properties_2["d"]
            h2_crit = h01-0.5*a2**2
            static_properties_2_crit = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h2_crit, s2)
            d2_crit = static_properties_2_crit["d"]
            v2_crit = Macrit*a2
            m2_crit = d2_crit*v2_crit*np.cos(alpha2)*A2
            vm1_crit = m2_crit/(d1*A1)
            v1_crit = vm1_crit/np.cos(alpha1)
            x0_crit = np.append(x0_crit, np.array([v1_crit, v2_crit, s2]))
        
            # Condition at rotor exit
            w3 = np.sqrt(2*(h0rel2-h3))
            vel3_data = evaluate_velocity_triangle_out(u, w3, beta3)
            v3 = vel3_data["v"]
            vm3 = vel3_data["v_m"]
            h03 = h3 + 0.5*v3**2
            x0 = np.append(x0, np.array([w3, w3, s3, s3, beta3]))

            # Critical condition at rotor exit
            static_properties_3 = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h3, s3)
            a3 = static_properties_3["a"]
            h3_crit = h0rel2-0.5*a3**2
            static_properties_3_crit = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h3_crit, s3)
            d3_crit = static_properties_3_crit["d"]
            v3_crit = Macrit*a3
            m3_crit = d3_crit*v3_crit*np.cos(beta3)*A3
            vm2_crit = m3_crit/(d2*A2)
            v2_crit = vm2_crit/np.cos(alpha2) 
            x0_crit = np.append(x0_crit, np.array([v2_crit, v3_crit, s3]))
            
            # Inlet stagnation state for next cascade equal stagnation state for current cascade
            h01 = h03
            d1 = static_properties_3["d"]
        
        # Inlet guess from mass convervation
        A_in = cascades_data["geometry"]["A_in"][0]
        A_out = cascades_data["geometry"]["A_out"][n_cascades-1]
        m_out = d_out*vm3*A_out
        v_in = m_out/(A_in*d0_in)
        x0 = np.insert(x0, 0, v_in)
        
                    
    else:
        
        # For only the case of only one cascade we split into two station: 1: cascade inlet, 3: cascade exit
        h01 = h0_in
        h3 = h_out
        s3 = s_out
        d3 = d_out
                
        alpha1 = cascades_data["geometry"]["theta_in"][0]
        alpha3 = cascades_data["geometry"]["theta_out"][0]
        A1 = cascades_data["geometry"]["A_in"][0]
        A_in = A1
        A3 = cascades_data["geometry"]["A_out"][0]
        
        v3 = np.sqrt(2*(h01-h3))
        vm3 = v3*np.cos(alpha3)
        x0 = np.append(x0, np.array([v3, v3, s3, s3, alpha3]))
        
        # Critical conditions
        static_properties_3 = fluid.compute_properties_meanline(CP.PSmass_INPUTS, h3, s3)
        a3 = static_properties_3["a"]
        h3_crit = h01-0.5*a3**2
        static_properties_3_crit = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h3_crit, s3)
        d3_crit = static_properties_3_crit["d"]
        v3_crit = a3*Macrit
        m3_crit = d3_crit*v3_crit*np.cos(alpha3)*A3
        vm1_crit = m3_crit/(d1*A1)
        v1_crit = vm1_crit/np.cos(alpha1)
        x0_crit = np.array([v1_crit, v3_crit, s3])
        
        
        # Inlet guess from mass convervation
        m3 = d3*vm3*A3
        v1 = m3/(A1*d1)
        x0 = np.insert(x0, 0, v1)
        
    # Merge initial guess for critical state and actual point
    x0 = np.concatenate((x0, x0_crit))
            
    return x0
    
def calculate_number_of_stages(cascades_data):
    """
    Calculate the number of stages based on the number of cascades.

    Args:
        cascades_data (dict): The data for the cascades.

    Raises:
        Exception: If the number of cascades is not valid.

    Returns:
        None
    """
    n_cascades = cascades_data["geometry"]["n_cascades"]    

    if n_cascades == 1:
        # If only one cascade, there are no stages
        n_stages = 0
        
    elif (n_cascades % 2) == 0:
        # If n_cascades is divisible by two, n_stages = n_cascades/2
        n_stages = int(n_cascades / 2)
        
    else:
        # If n_cascades is not 1 or divisible by two, it's an invalid configuration
        raise Exception("Invalid number of cascades")
   
    # Store the result in fixed_params
    cascades_data["fixed_params"]["n_stages"] = n_stages
    
def calculate_eta_drop_fractions(cascades_data):
    """
    Calculate efficiency penalty for each loss component.
    
    Args:
        cascades_data (dict): The data for the cascades.
    
    Returns:
        pd.DataFrame: A DataFrame containing efficiency drop fractions.
    """
    # Load parameters
    h_out_s = cascades_data["fixed_params"]["h_out_s"]
    n_cascades = cascades_data["geometry"]["n_cascades"]
    h0_in = cascades_data["plane"]["h0"].values[0]
    h_out = cascades_data["plane"]["h"].values[-1]
    v_out = cascades_data["plane"]["v"].values[-1]
    cascade = cascades_data["cascade"]
    
    dhs_sum = cascade["dh_s"].sum()
    dhss = h_out - h_out_s
    a = dhss / dhs_sum
    
    # Initialize DataFrame
    loss_fractions = pd.DataFrame(columns=["eta_drop_p", "eta_drop_s", "eta_drop_cl"])
    
    for i in range(n_cascades):
        fractions_temp = cascade.loc[i, "Y_p":"Y_cl"] / cascade.loc[i, "Y_tot"]
    
        dhs = cascade["dh_s"][i]
        eta_drop = a * dhs / (h0_in - h_out_s)
        loss_fractions_temp = fractions_temp * eta_drop
        loss_fractions.loc[len(loss_fractions)] = loss_fractions_temp.values
    
    eta_drop_kinetic = 0.5 * v_out**2 / (h0_in - h_out_s)
    cascades_data["overall"]["eta_drop_kinetic"] = eta_drop_kinetic
    
    return loss_fractions


def calculate_stage_parameters(cascades_data):
    """
    Calculate parameters relevant for the whole stage, e.g. reaction.

    Args:
        cascades_data (dict): The data for the cascades.

    Returns:
        None
    """
    n_stages = cascades_data["fixed_params"]["n_stages"]
    planes = cascades_data["plane"]
    h_vec = planes["h"].values
    
    # Degree of reaction
    R = np.zeros(n_stages)
    
    for i in range(n_stages):
        h1 = h_vec[i * 6]
        h2 = h_vec[i * 6 + 2]
        h3 = h_vec[i * 6 + 5]
        R[i] = (h2 - h3) / (h1 - h3)
        
    stages = {"R": R}
    cascades_data["stage"] = pd.DataFrame(stages)

    
def check_geometry(cascades_data):
    """
    Check if given geometry satisfies the model's assumptions.
    
    Args:
    cascades_data (dict): Data containing geometry information.
    
    Raises:
    Exception: If flaring between cascades or varying mean radius is detected.
    """
    geometry = cascades_data["geometry"]
    check_flaring(geometry)
    check_constant_radius(geometry)


def check_flaring(geometry):
    """
    Check if the area at exit of a cascade is equal to inlet of the next cascade.

    Args:
        geometry (dict): Geometry data.

    Raises:
        Exception: If flaring between cascades is detected.
    """
    A_out = geometry["A_out"]
    A_in = geometry["A_in"]
    if any(A_out[:-1] != A_in[1:]):
        raise Exception("Flaring between cascades detected")


def check_constant_radius(geometry):
    """
    Check if the given radius is constant for each cascade.

    Args:
        geometry (dict): Geometry data.

    Raises:
        Exception: If mean radius is not constant for each cascade.
    """
    radius = geometry["radius"]
    if any(radius != radius[0]):
        raise Exception("Mean radius is not constant for each cascade")


def check_n_cascades(geometry, n_cascades):
    """
    Check if the given number of cascades equals the number of geometries given.

    Args:
        geometry (dict): Geometry data.
        n_cascades (int): Number of cascades.

    Returns:
        int: Adjusted number of cascades.

    Raises:
        ValueError: If the number of geometries doesn't match the number of cascades.
    """
    n = len(geometry["r_ht_in"])
    if n != n_cascades:
        raise ValueError("Number of geometries doesn't match the number of cascades")
    return n

def convert_scaled_x0(x, cascades_data):
    
    """
    Convert scaled solution to real values using reference entropy and spouting velocity.
    """    
    
    v0         = cascades_data["fixed_params"]["v0"]
    s_ref      = cascades_data["fixed_params"]["s_ref"]
    n_cascades = cascades_data["geometry"]["n_cascades"]
    
    # Slice x into actual and critical values
    x_real = x.copy()[0:5*n_cascades+1]
    xcrit_real = x.copy()[5*n_cascades+1:]

    # Convert x0 to real values
    x_real[0] *= v0
    xcrit_real *=v0
    for i in range(n_cascades):
        x_real[5*i+1] *= v0
        x_real[5*i+2] *= v0
        x_real[5*i+3] *= s_ref
        x_real[5*i+4] *= s_ref
        
        xcrit_real[3*i+2] *= (s_ref/v0)
        
    x_real = np.concatenate((x_real, xcrit_real))
    
    return x_real

def scale_x0(x, cascades_data):
    """
    Scale initial guess with reference entropy and spouting velocity.
    """

    # Load parameters
    v0 = cascades_data["fixed_params"]["v0"]
    s_ref = cascades_data["fixed_params"]["s_ref"]
    n_cascades = cascades_data["geometry"]["n_cascades"]    

    # Slice x into actual and critical values
    x_scaled = x.copy()[:5*n_cascades+1]
    xcrit_scaled = x.copy()[5*n_cascades+1:]
    
    # Scale x0
    x_scaled[0] /= v0
    xcrit_scaled /= v0
    
    for i in range(n_cascades):
        x_scaled[5*i+1] /= v0
        x_scaled[5*i+2] /= v0
        x_scaled[5*i+3] /= s_ref
        x_scaled[5*i+4] /= s_ref
        
        xcrit_scaled[3*i+2] *= (v0/s_ref)

    x_scaled = np.concatenate((x_scaled, xcrit_scaled))

    return x_scaled


def update_fixed_params(cascades_data):
    
    """
    Update parameters used throughout the code, including v0, s_ref, m_ref, u, and h_out_s.
    Also initialize dataframes for cascade_data.

    Args:
        cascades_data (dict): Dictionary containing data related to cascades.

    Returns:
        None
    """
    
    # Load input paramaters
    p0_in      = cascades_data["BC"]["p0_in"]
    T0_in      = cascades_data["BC"]["T0_in"]
    p_out      = cascades_data["BC"]["p_out"]
    fluid      = cascades_data["BC"]["fluid"]
    A_in       = cascades_data["geometry"]["A_in"][0]
    radius     = cascades_data["geometry"]["radius"][0]
    omega      = cascades_data["BC"]["omega"]

    # Calculate stagnation properties at inlet
    stagnation_properties_in = fluid.compute_properties_meanline(CP.PT_INPUTS, p0_in, T0_in)
    s_in  = stagnation_properties_in["s"] # Entropy
    h0_in = stagnation_properties_in["h"] # Enthalpy
    d0_in = stagnation_properties_in["d"] # Density
    
    # Calculate properties for isentropic expansion
    static_properites_isentropic_expansion = fluid.compute_properties_meanline(CP.PSmass_INPUTS, p_out, s_in)
    h_out_s = static_properites_isentropic_expansion["h"] # Enthalpy
    
    # Calculate paramaters
    v0 = np.sqrt(2*(h0_in-h_out_s)) # spouting velocity
    u = omega*radius # Blade speed
    m_ref = A_in*v0*d0_in # Reference mass flow rate
    
    # Store data in cascades_data
    cascades_data["fixed_params"]["v0"] = v0
    cascades_data["fixed_params"]["h_out_s"] = h_out_s
    cascades_data["fixed_params"]["s_ref"] = s_in
    cascades_data["fixed_params"]["delta_ref"] = 0.011/3e5**(-1/7)
    cascades_data["fixed_params"]["m_ref"] = m_ref
    cascades_data["overall"]["blade_speed"] = u
    
    # Initialize dataframe for storing plane and cascade variables
    cascades_data["plane"] = pd.DataFrame(columns = keys_plane)
    cascades_data["cascade"] = pd.DataFrame(columns = keys_cascade)

def get_dof_bounds(n_cascades):
    
    """
    Return bounds on degrees of freedom.
    """    
    # Define bounds for exite relative flow angle
    lb_beta_out = np.array([40*np.pi/180 if i % 2 == 0 else -80*np.pi/180 for i in range(n_cascades)])
    ub_beta_out = np.array([80*np.pi/180 if i % 2 == 0 else -40*np.pi/180 for i in range(n_cascades)])
    
    # Generate bounds for actual cascades
    lb = np.array([0.01])
    ub = np.array([0.5])
    for i in range(n_cascades):
        lb = np.append(lb, [0.1, 0.1, 1, 1, lb_beta_out[i]])
        ub = np.append(ub, [0.99, 0.99, 1.2, 1.2, ub_beta_out[i]])

    # Generate bounds for critical cascades
    lb = np.append(lb, 0.01)
    ub = np.append(ub, 0.7)
    for i in range(n_cascades):
        if i != 0:
            lb = np.append(lb, 0.1)
            ub = np.append(ub, 0.99)
        lb = np.append(lb, [0.1, 1])
        ub = np.append(ub, [0.99, 1.2])

    return (lb, ub)
    
    
