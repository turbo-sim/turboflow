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

from ..solver import solve_nonlinear_system


from . import loss_model as lm
from .df_keys import *



def evaluate_velocity_triangle_in(u, v, alpha):
    # Compute the velocity triangle at the inlet of the cascade
    # Input:
    # 1) Blade speed
    # 2) Absolute velocity
    # 3) Absolute flow angle

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
    # Compute the velocity triangle at the outlet of the cascade
    # Input:
    # 1) Blade speed
    # 2) Relative velocity
    # 3) Relative flow angle

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

def evaluate_cascade_series(x, data_structure):
     
    p0_in      = data_structure["BC"]["p0_in"]
    T0_in      = data_structure["BC"]["T0_in"]
    p_out      = data_structure["BC"]["p_out"]
    fluid      = data_structure["BC"]["fluid"]
    alpha_in   = data_structure["BC"]["alpha_in"]
    n_cascades = data_structure["geometry"]["n_cascades"]
    n_stages   = data_structure["fixed_params"]["n_stages"]
    h_out_s    = data_structure["fixed_params"]["h_out_s"]
    s_ref      = data_structure["fixed_params"]["s_ref"]
    v0         = data_structure["fixed_params"]["v0"]
    m_ref      = data_structure["fixed_params"]["m_ref"]
    delta_ref  = data_structure["fixed_params"]["delta_ref"]
    lossmodel  = data_structure["loss_model"]
    u          = data_structure["overall"]["blade_speed"]
    n_dof      = 5 # Number of degrees of freedom for each cascade except for v_in (used for keeping track on indices)
    n_dof_crit = 3 # Number of degrees of freedom for each critical cascade 
    
    # Define vector for storing residuals
    residuals = np.array([])
    
    # Define degrees of freedom
    x_series = x[0:n_dof*n_cascades+1]
    x_crit   = x[n_dof*n_cascades+1:]
    v_in     = x_series[0]*v0
    x_series = np.delete(x_series, 0)
    
    # Ensure data_structure elements are is empty
    df = pd.DataFrame(columns=keys_plane)
    data_structure["plane"] = df
    df = pd.DataFrame(columns=keys_cascade)
    data_structure["cascade"] = df
        
    # Construct 
    data_structure_crit = {"BC" : {"p0_in" : p0_in, "T0_in" : T0_in, "fluid" : fluid, "alpha_in" : alpha_in},
                           "fixed_params" : {"m_ref" : m_ref, "s_ref" : s_ref, "v0" : v0,
                                             "delta_ref" : delta_ref},
                           "loss_model" : lossmodel}

    for i in range(n_cascades):
        
        # Make sure blade speed is zero for stator 
        u_cascade = u*(i % 2)
        
        # Calculate critical mach number for current cascade with the given inlet conditions 
        data_structure_crit["geometry"] = {key : values[i] for key, values in data_structure["geometry"].items() if key != "n_cascades"}
        data_structure_crit["u"]        = u_cascade
        
        x_crit_cascade = x_crit[i*n_dof_crit:(i+1)*n_dof_crit]
        
        crit_res = critical_residuals(x_crit_cascade, data_structure_crit)
        residuals = np.concatenate((residuals, crit_res))
        Ma_crit = data_structure_crit["Ma_crit"]
                
        # Retrieve the DOF for the current cascade
        x_cascade = x_series[i*n_dof:(i+1)*n_dof]
        
        # Scale DOF
        x_cascade[0] = x_cascade[0]*v0
        x_cascade[1] = x_cascade[1]*v0
        x_cascade[2] = x_cascade[2]*s_ref
        x_cascade[3] = x_cascade[3]*s_ref
        
        # Evaluate the current cascade at the given conditions
        cascade_res = evaluate_cascade(i, v_in, x_cascade, u_cascade, data_structure, Ma_crit)
        
        # Change boundary conditins for next cascade (for critical mach calculation)
        data_structure_crit["BC"]["p0_in"]    = data_structure["plane"]["p0rel"].values[-1]
        data_structure_crit["BC"]["T0_in"]    = data_structure["plane"]["T0rel"].values[-1]
        data_structure_crit["BC"]["alpha_in"] = data_structure["plane"]["alpha"].values[-1] 
                
        # Store cascade residuals
        residuals = np.concatenate((residuals, cascade_res))
    
    # Add exit pressure error to residuals 
    p_calc = data_structure["plane"]["p"].values[-1]
    p_error = (p_calc-p_out)/p0_in
    residuals = np.append(residuals, p_error)    
    
    if n_stages != 0:
        # Calculate stage variables
        stage_variables(data_structure)
    
    # Store global variables
    data_structure["overall"]["m"] = data_structure["plane"]["m"].values[-1]
    
    stag_h = data_structure["plane"]["h0"].values
    data_structure["overall"]["eta_ts"] = (stag_h[0]-stag_h[-1])/(stag_h[0]-h_out_s)
    
    loss_fracs = loss_fractions(data_structure)
    data_structure["loss_fracs"] = loss_fracs
    data_structure["cascade"] = pd.concat([data_structure["cascade"], loss_fracs], axis = 1)

    return residuals

def evaluate_cascade(i, v_in, x_cascade, u, data_structure, Macrit):
    
    # Evaluate current cascade
    # 1: Load parameters and degrees of freedom
    # 2: Evaluate each plane sequentially
    # 3: Returns array of residuals and dataframe of the states at each plane
    
    # Load necessary parameters
    fluid     = data_structure["BC"]["fluid"]
    geometry  = {key : values[i] for key, values in data_structure["geometry"].items() if key != "n_cascades"}
    A_out     = geometry["A_out"]
    theta_out = geometry["theta_out"]
    m_ref     = data_structure["fixed_params"]["m_ref"]
        
    # Structure degrees of freedom
    vel_throat = x_cascade[0]
    vel_out    = x_cascade[1]
    s_throat   = x_cascade[2]
    s_out      = x_cascade[3] 
    beta_out   = x_cascade[4]
    
    x_throat = np.array([vel_throat, theta_out, s_throat])
    x_out    = np.array([vel_out, beta_out, s_out])
    
    # Define residual array
    cascade_res = np.array([])
    
    # Define index for state dataframe
    inlet  = 3*i
    throat = 3*i+1
    outlet = 3*i+2
    
    #Evaluate inlet plane
    if i == 0: # For first stator inlet plane
        inlet_plane = evaluate_first_inlet(v_in, data_structure, u, geometry)
        data_structure["plane"].loc[len(data_structure["plane"])] = inlet_plane # Add plane state to dataframe
    else: # For all other inlet planes
        exit_plane = data_structure["plane"].loc[inlet-1] # State for previous plane
        inlet_plane = evaluate_inlet(data_structure, u, geometry, exit_plane)
        data_structure["plane"].loc[len(data_structure["plane"])] = inlet_plane # Add plane state to dataframe
        
    # Evaluate throat
    throat_plane, Y_info_throat = evaluate_outlet(x_throat, geometry, data_structure, u, A_out, inlet_plane) # Using A_out to compensate for using axial velocity in mass flow rate calculation
    data_structure["plane"].loc[len(data_structure["plane"])] = throat_plane # Add plane state to dataframe


    Y_err_throat = throat_plane["Y_err"]
    cascade_res = np.append(cascade_res, Y_err_throat)
    
    # Evaluate exit
    exit_plane, Y_info_exit = evaluate_outlet(x_out, geometry, data_structure, u, A_out, inlet_plane)
    data_structure["plane"].loc[len(data_structure["plane"])] = exit_plane # Add plane state to dataframe
    Y_err_out = exit_plane["Y_err"]
    cascade_res = np.append(cascade_res, Y_err_out)
    
    # Compute mach number error
    # actual_mach_0 = min(data_structure["plane"]["Marel"].values[outlet], Macrit)
    
    alpha = -16
    actual_mach = (data_structure["plane"]["Marel"].values[outlet]**alpha + Macrit **alpha)**(1/alpha)
        
    mach_err = data_structure["plane"]["Marel"].values[throat] - actual_mach
    cascade_res = np.append(cascade_res, mach_err)
    
    # Add mass flow deviations to residuals
    cascade_res = np.concatenate((cascade_res, (data_structure["plane"]["m"][inlet:outlet].values-data_structure["plane"]["m"][throat:outlet+1].values)))/m_ref

    # Add cascade information to cascade dataframe
    props_isen = fluid.compute_properties_meanline(CP.PSmass_INPUTS, exit_plane["p"], inlet_plane["s"])
    dhs = exit_plane["h"]-props_isen["h"]
    cascade_data = {"Y_tot" : Y_info_exit["Total"],
                    "Y_p" : Y_info_exit["Profile"],
                    "Y_s" : Y_info_exit["Secondary"],
                    "Y_cl" : Y_info_exit["Clearance"],
                    "dh_s" : dhs}
    
    data_structure["cascade"].loc[len(data_structure["cascade"])] = cascade_data

    return cascade_res
        
def evaluate_first_inlet(v, data_structure, u, geometry): 
    
        # Evaluate first stator inlet plane 
    
        # Load boundary conditions
        T0_in     = data_structure["BC"]["T0_in"]
        p0_in     = data_structure["BC"]["p0_in"]
        fluid     = data_structure["BC"]["fluid"]
        alpha_in  = data_structure["BC"]["alpha_in"]
        delta_ref = data_structure["fixed_params"]["delta_ref"] # Reference inlet displacement thickness (for loss calculations)
                
        # Load geometry
        A = geometry["A_in"]
        c = geometry["c"]

        # Evaluate velocity triangle
        vel_in = \
            evaluate_velocity_triangle_in(u, v, alpha_in)
        w   = vel_in["w"]
        w_m = vel_in["w_m"]
        
        # Stagnation properties
        stag_props = fluid.compute_properties_meanline(CP.PT_INPUTS, p0_in, T0_in)
        stag_props = change_dict(stag_props, '0')
        h0 = stag_props["h0"]
        
        # Static propertes
        s = stag_props["s0"]
        h  = h0 - v**2 / 2
        static_props = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h, s)
        a  = static_props["a"]
        d  = static_props["d"]
        mu = static_props["mu"]
        
        # Relative properties
        h0rel = h+0.5*w**2
        rel_props = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0rel, s)
        rel_props = change_dict(rel_props, '0rel')

        # Calcluate mach, Reonlds and mass flow rate for cascade inlet
        Ma = v / a
        Marel = w / a
        Re = d*w*c/mu;
        m = d*w_m*A
        
        # Calculate inlet displacement thickness to blade height ratio
        delta = delta_ref*Re**(-1/7)
        
        # Store result
        plane = {**vel_in, **static_props, **stag_props, **rel_props}

        plane["Ma"] = Ma
        plane["Marel"] = Marel
        plane["Re"] = Re
        plane["m"] = m
        plane["delta"] = delta
        plane["Y_err"] = np.nan
        plane["Y"] = np.nan
        plane["dh_s"] = 0
                
        return plane

def evaluate_inlet(data_structure, u, geometry, exit_plane):
    
    # Evaluates the inlet plane of current cascade (except for first stator)
        
    # Load thermodynamic boundary conditions
    fluid     = data_structure["BC"]["fluid"]
    delta_ref = data_structure["fixed_params"]["delta_ref"] # Reference inlet displacement thickness (for loss calculations)

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
    rel_props = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0rel, s)
    rel_props = change_dict(rel_props, '0rel')
    
    # Calcluate mach, Reonlds and mass flow rate for cascade inlet
    Ma = v / a
    Marel = w / a
    Re = d*w*c/mu;
    m = d*w_m*A
    
    # Calculate inlet displacement thickness to blade height ratio
    delta = delta_ref*Re**(-1/7)
    
    # Store result
    plane = exit_plane.copy()
    for key, value in {**vel_in,**rel_props}.items():
        plane[key] = value

    plane["Ma"] = Ma
    plane["Marel"] = Marel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = delta
    plane["Y_err"] = np.nan # Not relevant for inblet plane
    plane["Y"] = np.nan
    plane["dh_s"] = 0

    return plane    

def evaluate_outlet(x, geometry, data_structure, u,  A, inlet_plane):
    
    # Evaluates the throat or exit plane
    
    # Load thermodynamic boundaries
    fluid     = data_structure["BC"]["fluid"]
    lossmodel = data_structure["loss_model"]
    
    # Load geomerrical parameters 
    c = geometry["c"]
    
    # Load inlet state variables from inlet of current cascade
    s_in     = inlet_plane["s"]
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
    
    static_props = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h, s)
    a = static_props["a"]
    d = static_props["d"]
    mu = static_props["mu"]
    p = static_props["p"]
    
    stag_props = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0, s)
    stag_props = change_dict(stag_props, '0')
    
    rel_props = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h0rel, s)
    rel_props = change_dict(rel_props, '0rel')

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
    cascade_data["flow"]["p0_rel_out"] = rel_props["p0rel"]
    cascade_data["flow"]["p_in"] = p_in
    cascade_data["flow"]["p_out"] = static_props["p"]
    cascade_data["flow"]["beta_out"] = beta
    cascade_data["flow"]["beta_in"] = beta_in
    cascade_data["flow"]["Ma_rel_in"] = Marel_in
    cascade_data["flow"]["Ma_rel_out"] = Marel
    cascade_data["flow"]["Re_in"] = Re_in
    cascade_data["flow"]["Re_out"] = Re
    cascade_data["flow"]["delta"] = delta
    cascade_data["flow"]["gamma_out"] = static_props["gamma"]
    
    Y, Y_err, Y_info = evaluate_loss_model(cascade_data)
    
    # Store result
    plane = {**vel_out, **static_props, **stag_props, **rel_props}
    
    plane["Ma"] = Ma
    plane["Marel"] = Marel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = np.nan # Not relevant for exit/throat plane
    plane["Y_err"] = Y_err
    plane["Y"] = Y
        
    return plane, Y_info
    
def evaluate_loss_model(cascade_data):
    
    # Load necessary parameters
     lossmodel = cascade_data["loss_model"]
     p0rel_in  = cascade_data["flow"]["p0_rel_in"]
     p0rel_out = cascade_data["flow"]["p0_rel_out"]
     p_out     = cascade_data["flow"]["p_out"] 
     
     # Compute the loss coefficient from its definition
     Y_def = (p0rel_in-p0rel_out)/(p0rel_out-p_out)
     
     # Compute the loss coefficient from the correlations
     Y, Y_info = lm.loss(cascade_data, lossmodel)
     
     # Compute loss coefficient error
     Y_err = (Y_def-Y)
    
     return Y, Y_err, Y_info    

def root_function(x, data_structure):
    
    # Function evaluating the cascade series with x degrees of freedom
    # Returns the residuals
    
    residuals = evaluate_cascade_series(x, data_structure)
        
    return residuals
    
def cascade_series_analysis(data_structure, x0, scaled = True):
    
    # Check if geometry is valid
    check_geometry(data_structure["geometry"], data_structure["geometry"]["n_cascades"])
            
    # Solve nonlinear system of equations
    sol, conv_hist = solve_nonlinear_system(root_function, x0, args = (data_structure), method = 'hybr')
        
    return sol, conv_hist

def critical_residuals(x, data_structure_crit):
    
    # x consost of v_in*, v_throat* and s_throat*
    
    m_ref = data_structure_crit["fixed_params"]["m_ref"] 
    
    f0 = evaluate_critical_cascade(x, data_structure_crit) # Evaluate first to give both f0 to estimate Jac and evaluate constraints
    J = critical_cascade_jacobian(x, data_structure_crit, f0) # Calculate Jacobian of a function returning f, g1 and g2
    
    a11 = J[1,0] 
    a12 = J[2,0]
    a21 = J[1,1]
    a22 = J[2,1]
    
    b1 = -1*J[0,0]
    b2 = -1*J[0,1]
    
    # Solve A*l = b
    l1 = (a22*b1-a12*b2)/(a11*a22-a12*a21)
    l2 = (a11*b2-a21*b1)/(a11*a22-a12*a21)
    
    # Define the last element of the gradient of the Lagrange function  
    df = J[0,2]
    dg1 = J[1,2]
    dg2 = J[2,2]
    grad = (df+l1*dg1+l2*dg2)/m_ref
            
    # Return last 3 elements of the Lagrangian gradient (df/dx2+l1*dg1/dx2+l2*dg2/dx2 and g1, g2)
    g = f0[1:]
    res = np.insert(g,0,grad)

    return res

def critical_cascade_jacobian(x, data, f0):
    
    eps = 1e-3*x # FIXME: lower eps not working
    # eps = np.sqrt(np.finfo(float).eps)*np.maximum(1, abs(x))*np.sign(x)
    J = approx_derivative(evaluate_critical_cascade, x, method = '2-point', f0 = f0, abs_step = eps, args = (data,))
    
    return J

def critical_cascade_constraints(x, data_structure_crit):
    
    # Evaluates critical casacde and return the residuals (constraints)
    
    output = evaluate_critical_cascade(x, data_structure_crit)
    residuals = output[1:]

    return residuals

def evaluate_critical_cascade(x_crit, data_structure_crit):
    
    # Evaluates critical cascade
    # 1: Loads relevant parameters
    # 2: Evaluate inlet plane
    # 3: Evaluate exit plane
    # 4: Stores critical mach number
    # 5: Returns mass flow rate and residuals (obj function and residuals)
    
    # Load necessary parameters
    u         = data_structure_crit["u"]
    geometry  = data_structure_crit["geometry"]
    A_out     = geometry["A_out"]
    theta_out = geometry["theta_out"]
    
    # Variables for scaling
    v0    = data_structure_crit["fixed_params"]["v0"]
    s_ref = data_structure_crit["fixed_params"]["s_ref"]
    m_ref = data_structure_crit["fixed_params"]["m_ref"]

    
    # Rename degrees of freedom
    v_in     = x_crit[0]*v0
    w_throat = x_crit[1]*v0
    s_throat = x_crit[2]*s_ref
    
    # Evlauate inlet plane (for critical cascade every inlet is "first" inlet)
    inlet_plane = evaluate_first_inlet(v_in, data_structure_crit, u, geometry)
    
    # Evaluate throat
    beta_throat = theta_out
    x = np.array([w_throat, beta_throat, s_throat])
    throat_plane, Y_info = evaluate_outlet(x, geometry, data_structure_crit, u,  A_out, inlet_plane)
        
    # Calculate error of mass balance and loss coeffienct
    m_in = inlet_plane["m"]
    m_throat = throat_plane["m"]
    residuals = np.array([(m_in-m_throat)/m_ref, throat_plane["Y_err"]])
    
    # Store critical mach number
    data_structure_crit["Ma_crit"] = throat_plane["Marel"]
    
    output = np.insert(residuals, 0, m_throat)
    
    return output


def change_dict(input_dict, string):
    for key in list(input_dict.keys()):
        new_key = str(key)+string
        input_dict[new_key] = input_dict.pop(key)
    return input_dict

def generate_initial_guess(data_structure, eta = 0.9, R = 0.4, Macrit = 0.92):
    
    #FIXME: Could still need some updates. Still slower than the fixed one
    
    p0_in      = data_structure["BC"]["p0_in"]
    T0_in      = data_structure["BC"]["T0_in"]
    p_out      = data_structure["BC"]["p_out"]
    fluid      = data_structure["BC"]["fluid"]
    n_cascades = data_structure["geometry"]["n_cascades"]    
    n_stages   = data_structure["fixed_params"]["n_stages"]
    h_out_s    = data_structure["fixed_params"]["h_out_s"]
    omega      = data_structure["BC"]["omega"]
    radius     = data_structure["geometry"]["radius"][0] #XXX: Constant radius
    
    u        = radius*omega
    
    # Inlet stagnation state
    stag_props = fluid.compute_properties_meanline(CP.PT_INPUTS, p0_in, T0_in) 
    h0_in      = stag_props["h"]
    d0_in      = stag_props["d"]
    s_in       = stag_props["s"]
    h_out = h0_in - eta*(h0_in-h_out_s) 
    
    # Exit static state for expansion with given efficiency
    static_exit = fluid.compute_properties_meanline(CP.HmassP_INPUTS, h_out, p_out)
    s_out = static_exit["s"]
    d_out = static_exit["d"]
        
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
            A1 = data_structure["geometry"]["A_in"][index_stator]
            A2 = data_structure["geometry"]["A_out"][index_stator]
            A3 = data_structure["geometry"]["A_out"][index_rotor]
            
            alpha1 = data_structure["geometry"]["theta_in"][index_stator]
            alpha2 = data_structure["geometry"]["theta_out"][index_stator]
            beta2  = data_structure["geometry"]["theta_in"][index_rotor]
            beta3  = data_structure["geometry"]["theta_out"][index_rotor]
            
            h1 = h_in[i]
            h2 = h_mid[i]
            h3 = h_out[i]
            
            s1 = s_cascades[i*2]
            s2 = s_cascades[i*2+1]
            s3 = s_cascades[i*2+2]
            
            # Conditions stator exit
            h02 = h01
            v2 = np.sqrt(2*(h02-h2))
            vel2_data = evaluate_velocity_triangle_in(u, v2, alpha2)
            w2 = vel2_data["w"]
            h0rel2 = h2 + 0.5*w2**2
            x0 = np.append(x0, np.array([v2, v2, s2, s2, alpha2]))
            
            # Critical conditions stator exit
            static_props2 = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h2, s2)
            a2 = static_props2["a"]
            d2 = static_props2["d"]
            h2_crit = h01-0.5*a2**2
            static_props2_crit = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h2_crit, s2)
            d2_crit = static_props2_crit["d"]
            v2_crit = Macrit*a2
            m2_crit = d2_crit*v2_crit*np.cos(alpha2)*A2
            vm1_crit = m2_crit/(d1*A1)
            v1_crit = vm1_crit/np.cos(alpha1)
            x0_crit = np.append(x0_crit, np.array([v1_crit, v2_crit, s2]))
        
            # Conditions rotor exit
            w3 = np.sqrt(2*(h0rel2-h3))
            vel3_data = evaluate_velocity_triangle_out(u, w3, beta3)
            v3 = vel3_data["v"]
            vm3 = vel3_data["v_m"]
            h03 = h3 + 0.5*v3**2
            x0 = np.append(x0, np.array([w3, w3, s3, s3, beta3]))

            # Critical conditions rotor exit
            static_props3 = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h3, s3)
            a3 = static_props3["a"]
            h3_crit = h0rel2-0.5*a3**2
            static_props3_crit = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h3_crit, s3)
            d3_crit = static_props3_crit["d"]
            v3_crit = Macrit*a3
            m3_crit = d3_crit*v3_crit*np.cos(beta3)*A3
            vm2_crit = m3_crit/(d2*A2)
            v2_crit = vm2_crit/np.cos(alpha2) 
            x0_crit = np.append(x0_crit, np.array([v2_crit, v3_crit, s3]))
            
            # Inlet stagnation state for next cascade equal stagnation state for current cascade
            h01 = h03
            d1 = static_props3["d"]
        
        # Inlet guess from mass convervation
        A_in = data_structure["geometry"]["A_in"][0]
        A_out = data_structure["geometry"]["A_out"][n_cascades-1]
        m_out = d_out*vm3*A_out
        v_in = m_out/(A_in*d0_in)
        x0 = np.insert(x0, 0, v_in)
        
                    
    else:
        
        # For only the case of only one cascade we split into two station: 1: cascade inlet, 3: cascade exit
        h01 = h0_in
        h3 = h_out
        s3 = s_out
        d3 = d_out
                
        alpha1 = data_structure["geometry"]["theta_in"][0]
        alpha3 = data_structure["geometry"]["theta_out"][0]
        A1 = data_structure["geometry"]["A_in"][0]
        A_in = A1
        A3 = data_structure["geometry"]["A_out"][0]
        
        v3 = np.sqrt(2*(h01-h3))
        vm3 = v3*np.cos(alpha3)
        x0 = np.append(x0, np.array([v3, v3, s3, s3, alpha3]))
        
        # Critical conditions
        static_props3 = fluid.compute_properties_meanline(CP.PSmass_INPUTS, h3, s3)
        a3 = static_props3["a"]
        h3_crit = h01-0.5*a3**2
        static_props3_crit = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h3_crit, s3)
        d3_crit = static_props3["d"]
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
    
def number_stages(data_structure):
    
    n_cascades = data_structure["geometry"]["n_cascades"]    

    if n_cascades == 1:
        n_stages = 0
        
    elif (n_cascades % 2) == 0:
        n_stages = int(n_cascades/2)
        
    else:
        raise Exception("Invalid number of cascades")
    
    data_structure["fixed_params"]["n_stages"] = n_stages

def loss_fractions(data_structure):
    
    # Calculate efficiency penalty for each loss component
        
    h_out_s    = data_structure["fixed_params"]["h_out_s"]
    n_cascades = data_structure["geometry"]["n_cascades"]    
    h0_in      = data_structure["plane"]["h0"].values[0]
    h_out      = data_structure["plane"]["h"].values[-1]
    v_out      = data_structure["plane"]["v"].values[-1]
    cascade    = data_structure["cascade"]
    
    dhs_sum = sum(cascade["dh_s"])
                
    dhss = h_out-h_out_s
    a = dhss/dhs_sum
    
    loss_fractions = pd.DataFrame(columns = ["eta_drop_p", "eta_drop_s", "eta_drop_cl"]) 
    
    for i in range(n_cascades):
        
        fractions_temp = cascade.loc[i]["Y_p":"Y_cl"]/cascade.loc[i]["Y_tot"]
        
        dhs = cascade["dh_s"][i]
        eta_drop = a*dhs/(h0_in-h_out_s)
        loss_fractions_temp = fractions_temp*eta_drop
        loss_fractions.loc[len(loss_fractions)] = loss_fractions_temp.values
        
    eta_drop_kinetic = 0.5*v_out**2/(h0_in-h_out_s)
    data_structure["overall"]["eta_drop_kinetic"] = eta_drop_kinetic
        
    return loss_fractions

def stage_variables(data_structure):
    
    n_stages = data_structure["fixed_params"]["n_stages"]
    planes   = data_structure["plane"]
    h_vec = planes["h"].values
    
    # Degree of reaction
    R = np.zeros(n_stages)
    
    for i in range(n_stages):
        h1 = h_vec[i*6]
        h2 = h_vec[i*6+2]
        h3 = h_vec[i*6+5]
        R[i] = (h2-h3)/(h1-h3)
        
    stages = {"R" : R}
    data_structure["stage"] = pd.DataFrame(stages)

def check_geometry(geometry, n_cascades):
    
    # Check if given geometry satisifies the models assumptions
    
    if n_cascades > 1:
        check_flaring(geometry) # no flaring between cascades
        check_radius(geometry) # constant mean radius

def check_flaring(geometry):
    # Check if the area at exit of a cascade is equal to inlet of next cascade
    A_in = geometry["A_in"]
    A_out = geometry["A_out"]
    A_err = A_out[0:-1]-A_in[1:]

    if len(np.nonzero(A_err)[0] != 0):
        raise Exception("Flaring between cascades detected")

def check_radius(geometry):
    # Check if given radius is constant for each cascades
    nonzero_indices = np.nonzero(geometry["radius"]-geometry["radius"][0])[0]
    if len(nonzero_indices != 0):
        raise Exception("Mean radius is not constant for each cascade")

def check_n_cascades(geometry, n_cascades):
    # Check if the given number of cascades equal the number of geometries given
    n = len(geometry["r_ht_in"])
    if n != n_cascades:
        n_cascades = n
        
    return n_cascades

def rescale_x0(x, data_structure):
    
    # Convert the array of degrees of freedom from scaled to real values
    
    v0         = data_structure["fixed_params"]["v0"]
    s_ref      = data_structure["fixed_params"]["s_ref"]
    n_cascades = data_structure["geometry"]["n_cascades"]
    x_real = x.copy()[0:5*n_cascades+1]
    xcrit_real = x.copy()[5*n_cascades+1:]
    
    vel_indices = np.array([])
    s_indices   = np.array([])
    
    scrit_indices   = np.array([])

    for i in range(n_cascades):
        vel_indices = np.append(vel_indices, 5*i+1)
        vel_indices = np.append(vel_indices, 5*i+2)
        s_indices   = np.append(s_indices,   5*i+3)
        s_indices   = np.append(s_indices,   5*i+4)
        
        scrit_indices = np.append(scrit_indices, 3*i+2)
    
    x_real[0] *= v0
    for vel_i, s_i in zip(vel_indices, s_indices):
        x_real[int(vel_i)] *= v0 
        x_real[int(s_i)]   *= s_ref
        
    xcrit_real *=v0
    for s_i in scrit_indices:
        xcrit_real[int(s_i)] *= (s_ref/v0)
        
    x_real = np.concatenate((x_real, xcrit_real))
    return x_real

def scale_x0(x, data_structure):
    
    # Convert the array of degrees of freedom from real to scaled values
    v0         = data_structure["fixed_params"]["v0"]
    s_ref      = data_structure["fixed_params"]["s_ref"]
    n_cascades = data_structure["geometry"]["n_cascades"]    
    x_scaled = x.copy()[0:5*n_cascades+1]
    xcrit_scaled = x.copy()[5*n_cascades+1:]
    
    vel_indices = np.array([])
    s_indices   = np.array([])
    
    scrit_indices   = np.array([])

    for i in range(n_cascades):
        vel_indices = np.append(vel_indices, 5*i+1)
        vel_indices = np.append(vel_indices, 5*i+2)
        s_indices   = np.append(s_indices,   5*i+3)
        s_indices   = np.append(s_indices,   5*i+4)
        
        scrit_indices = np.append(scrit_indices, 3*i+2)
        
    x_scaled[0] /= v0
    for vel_i, s_i in zip(vel_indices, s_indices):
        x_scaled[int(vel_i)] /= v0 
        x_scaled[int(s_i)]   /= s_ref
    
    xcrit_scaled /= v0
   
    for s_i in scrit_indices:
        xcrit_scaled[int(s_i)] *= (v0/s_ref)
        
    x_scaled = np.concatenate((x_scaled, xcrit_scaled))
    return x_scaled

def update_fixed_params(data_structure):
    
    p0_in      = data_structure["BC"]["p0_in"]
    T0_in      = data_structure["BC"]["T0_in"]
    p_out      = data_structure["BC"]["p_out"]
    fluid      = data_structure["BC"]["fluid"]
    A_in       = data_structure["geometry"]["A_in"][0]
    radius     = data_structure["geometry"]["radius"][0]
    omega      = data_structure["BC"]["omega"]

    stag_props = fluid.compute_properties_meanline(CP.PT_INPUTS, p0_in, T0_in)
    s_in  = stag_props["s"]
    h0_in = stag_props["h"]
    d0_in = stag_props["d"]
    static_isentropic_props = fluid.compute_properties_meanline(CP.PSmass_INPUTS, p_out, s_in)
    h_out_s = static_isentropic_props["h"]
    v0 = np.sqrt(2*(h0_in-h_out_s))
    u = omega*radius
    
    data_structure["fixed_params"]["v0"] = v0
    data_structure["fixed_params"]["h_out_s"] = h_out_s
    data_structure["fixed_params"]["s_ref"] = s_in
    data_structure["fixed_params"]["delta_ref"] = 0.011/3e5**(-1/7)
    m_ref = A_in*v0*d0_in
    data_structure["fixed_params"]["m_ref"] = m_ref
    data_structure["overall"]["blade_speed"] = u
    
