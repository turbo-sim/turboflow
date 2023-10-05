# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:25:28 2023

@author: lasseba
"""

import numpy as np


def calculate_loss_coefficient(cascade_data):
    
    # Compute the loss coefficient for a cascade row by using the Kacker-Okapuu loss model :cite:`kacker_prediction_1982`
    # The loss coefficient is a sum of the profile, secondary, trailing edge and tip clearance loss coefficent
    # The loss models adopts the total pressure loss coefficient
    # This model does not account for incidence losses (off-design losses)

    
    flow_parameters = cascade_data["flow"]
    geometry = cascade_data["geometry"]
    cascade_type=cascade_data["type"]
    
    # Profile loss coefficient
    Y_p = get_profile_loss(flow_parameters, geometry, cascade_type)

    # Secondary loss coefficient
    Y_s = get_secondary_loss(flow_parameters, geometry)

    # Tip clearance loss coefficient
    Y_cl = get_tip_clearance_loss(flow_parameters, geometry, cascade_type)
    
    # Trailing edge loss coefficienct
    Y_te=get_trailing_edge_loss(flow_parameters, geometry)
    
    Y=Y_p+Y_s+Y_cl+Y_te
    
    loss_dict = {"Profile" : Y_p,
                 "Incidence": 0,
                 "Trailing": Y_te,
                 "Secondary" : Y_s,
                 "Clearance" : Y_cl,
                 "Total" : Y}
    
    return [Y, loss_dict]

def get_profile_loss(flow_parameters, geometry, cascade_type):
    
    # Computes the profile loss coefficient for the current cascade according to the Kacker and Okapuu loss model
    # flow_parameters is a dictionary containing kinetic and thermodynamic variables describing the flow through the current cascade
    # geometry is a dictionary with the geometrical parameters describing the current cascade
    # cascade_type is a string that specify whether the current cascade is a stator or a rotor
    
    # Load data
    Re = flow_parameters["Re_out"]
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    Ma_rel_in = flow_parameters["Ma_rel_in"]
    p0rel_in = flow_parameters["p0_rel_in"]
    p_in = flow_parameters["p_in"]
    p0rel_out = flow_parameters["p0_rel_out"]
    p_out = flow_parameters["p_out"]
    beta_out = flow_parameters["beta_out"]

    r_ht_in = geometry["r_ht_in"]
    s = geometry["s"]
    c = geometry["c"]
    theta_in = geometry["theta_in"]
    t_max = geometry["t_max"]
    
    # Reynolds number correction factor
    f_Re=(Re/2e5)**(-0.4)*(Re<2e5)+1*(Re >= 2e5 and Re <= 1e6) + (Re/1e6)**(-0.2)*(Re>1e6)
    
    # Mach number correction factor
    f_Ma=1+60*(Ma_rel_out-1)**2*(Ma_rel_out > 1)    
        
    # Compute losses related to shock effects at the inlet of the cascade
    f_hub = get_hub_to_mean_mach_ratio(r_ht_in,cascade_type)
    a = max(0,f_hub*Ma_rel_in-0.4)
    Y_shock = 0.75*a**1.75*r_ht_in*(p0rel_in-p_in)/(p0rel_out-p_out)
    Y_shock = max(0,Y_shock)
    
    # Compute compressible flow correction factors
    Kp, K2, K1 = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)
    
    # Yp_reaction and Yp_impulse according to Aungier correlation
    # These formulas are valid for 40<abs(angle_out)<80
    # Extrapolating outside of this limits might give completely wrong results
    # If the optimization algorithm has upper and lower bounds for the outlet 
    # angle there is no need to worry about this problem
    # angle_out_bis keeps the 40deg-losses for outlet angles lower than 40deg
    angle_out_bis = max(abs(beta_out),40*np.pi/180)
    Yp_reaction = nozzle_blades(s/c,angle_out_bis)
    Yp_impulse = impulse_blades(s/c,angle_out_bis)
    
    # Formula according to Kacker-Okapuu
    Y_p = Yp_reaction-abs(theta_in/beta_out)*(theta_in/beta_out)*(Yp_impulse-Yp_reaction)

    # Limit the extrapolation of the profile loss to avoid negative values for
    # blade profiles with little deflection
    # Low limit to 80% of the axial entry nozzle profile loss
    # This value is completely arbitrary
    Y_p = max(Y_p,0.8*Yp_reaction)
    
    # Avoid unphysical effect on the thickness by defining the variable aa
    aa=max(0,-theta_in/beta_out)
    Y_p = Y_p*((t_max/c)/0.2)**aa
    Y_p = 0.914*(2/3*Y_p*Kp+Y_shock)
    
    # Corrected profile loss coefficient
    Y_p = f_Re*f_Ma*Y_p
    
    return Y_p

def get_secondary_loss(flow_parameters, geometry):
    
    # Computes the secondary loss coefficient for the current cascade according to the Kacker and Okapuu loss model
    # flow_parameters is a dictionary containing kinetic and thermodynamic variables describing the flow through the current cascade
    # geometry is a dictionary with the geometrical parameters describing the current cascade
    
    Ma_rel_out=flow_parameters["Ma_rel_out"]
    Ma_rel_in=flow_parameters["Ma_rel_in"]
    beta_out=flow_parameters["beta_out"]
    beta_in=flow_parameters["beta_in"]
    
    b=geometry["b"]
    H=geometry["H"]
    c=geometry["c"]
    theta_in=geometry["theta_in"]

    
    # Compute compressible flow correction factors
    Kp, K2, K1 = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)
    
    # Secondary loss coefficient  
    K3 = (b/H)**2
    Ks = 1-K3*(1-Kp)
    Ks = max(0.1,Ks)
    angle_m = np.arctan((np.tan(beta_in)+np.tan(beta_out))/2)
    Z = 4*(np.tan(beta_in)-np.tan(beta_out))**2*np.cos(beta_out)**2/np.cos(angle_m)
    far = (1-0.25*np.sqrt(abs(2-H/c)))/(H/c)*(H/c < 2)+1/(H/c)*(H/c >= 2)
    Y_s = 1.2*Ks*0.0334*far*Z*np.cos(beta_out)/np.cos(theta_in)
    
    return Y_s

def nozzle_blades(r_sc,angle_out):
    
    # Use Aungier correlation to compute the pressure loss coefficient
    # This correlation is a formula that reproduces the figures from the Ainley
    # and Mathieson original figures
    
    phi = 90-angle_out*180/np.pi
    r_sc_min = (0.46+phi/77)*(phi < 30) + (0.614+phi/130)*(phi >= 30)
    X = r_sc-r_sc_min
    A = (0.025+(27-phi)/530)*(phi < 27) + (0.025+(27-phi)/3085)*(phi >= 27)
    B = 0.1583-phi/1640
    C = 0.08*((phi/30)**2-1)
    n = 1+phi/30
    Yp_reaction = (A+B*X**2+C*X**3)*(phi < 30) + (A+B*abs(X)**n)*(phi >= 30)
    
    return Yp_reaction

def impulse_blades(r_sc,angle_out):
    
    # Use Aungier correlation to compute the pressure loss coefficient
    # This correlation is a formula that reproduces the figures from the Ainley
    # and Mathieson original figures
    
    phi = 90-angle_out*180/np.pi
    r_sc_min = 0.224+1.575*(phi/90)-(phi/90)**2
    X = r_sc-r_sc_min
    A = 0.242-phi/151+(phi/127)**2
    B = (0.3+(30-phi)/50)*(phi < 30) + (0.3+(30-phi)/275)*(phi >=30)
    C = 0.88-phi/42.4+(phi/72.8)**2
    Yp_impulse = A+B*X**2-C*X**3
    
    return Yp_impulse

def get_compressible_correction_factors(Ma_rel_in,Ma_rel_out):
    
    # Compute compressible flow correction factor according to Kacker and Okapuu loss model
    # The loss correlation proposed by ainley ant Mathieson overpredicts losses at high mach numbers 
    # These correction factors reduces losses accordingly at higher mach numbers
    
    K1=1*(Ma_rel_out < 0.2) + (1-1.25*(Ma_rel_out-0.2))*(Ma_rel_out > 0.2 and Ma_rel_out < 1.00)
    K2=(Ma_rel_in/Ma_rel_out)**2
    Kp=1-K2*(1-K1)
    Kp=max(0.1,Kp)
    return [Kp,K2,K1]

def get_hub_to_mean_mach_ratio(r_ht, cascade_type):
    
    # Compute the ratio between mach at hub and mean span at the inlet of the current cascade
    # Due to radial variation in gas conditions, mach at the hub will alway be higher than at mean
    # Thus, shock losses at the hub could occur even when the mach is subsonic at the mean balde span
    
    # r_ht is the hub to tip ratio at the inlet of the current cascade
    # cascade_type is a string that specify whether the current cascade is a stator or a rotor
    
    if r_ht < 0.5:
        r_ht = 0.5     # Numerical trick to prevent extrapolation
        
    r_ht_data = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    #Stator curve
    f_data_S = [1.4, 1.18, 1.05, 1.0, 1.0, 1.0]
    
    #Rotor curve
    f_data_R = [2.15, 1.7, 1.35, 1.12, 1.0, 1.0]
    
    if cascade_type == 'stator':
        f = np.interp(r_ht,r_ht_data,f_data_S)
    elif cascade_type == 'rotor':
        f = np.interp(r_ht,r_ht_data,f_data_R)
    else:
        print("Specify the type of cascade")
        
    return f    

def get_trailing_edge_loss(flow_parameters, geometry):
    
    # Computes the trailing edge coefficient for the current cascade according to the Kacker and Okapuu loss model
    # flow_parameters is a dictionary containing kinetic and thermodynamic variables describing the flow through the current cascade
    # geometry is a dictionary with the geometrical parameters describing the current cascade
    
    t_te = geometry["te"]
    o = geometry["o"]
    angle_in = geometry["theta_in"]
    angle_out = flow_parameters["beta_out"]
    
    # Range of trailing edge to throat opening ratio
    r_to_data = [0, 0.2, 0.4]
    
    # Reacting blading
    phi_data_reaction = [0, 0.045, 0.15]
    
    # Impulse blading
    phi_data_impulse = [0, 0.025, 0.075]
    
    # Numerical trick to avoid too big r_to's
    r_to = min(0.4,t_te/o)
    
    # Interpolate data
    d_phi2_reaction = np.interp(r_to,r_to_data,phi_data_reaction)
    d_phi2_impulse = np.interp(r_to,r_to_data,phi_data_impulse)
    
    #Compute kinetic energy loss coefficient
    d_phi2 = d_phi2_reaction-abs(angle_in/angle_out)*(angle_in/angle_out)*(d_phi2_impulse-d_phi2_reaction)

    # Limit the extrapolation of the trailing edge loss
    d_phi2 = max(d_phi2,d_phi2_impulse/2)
    Y_te = 1/(1-d_phi2)-1
    
    return Y_te

def get_tip_clearance_loss(flow_parameters, geometry, cascade_type):
    
    # Computes the profile loss coefficient for the current cascade according to the Kacker and Okapuu loss model
    # flow_parameters is a dictionary containing kinetic and thermodynamic variables describing the flow through the current cascade
    # geometry is a dictionary with the geometrical parameters describing the current cascade
    # cascade_type is a string that specify whether the current cascade is a stator or a rotor

    
    beta_out = flow_parameters["beta_out"]
    beta_in = flow_parameters["beta_in"]
    
    H = geometry["H"]
    c = geometry["c"]
    t_cl = geometry["t_cl"]
    
    angle_m = np.arctan((np.tan(beta_in)+np.tan(beta_out))/2)
    Z = 4*(np.tan(beta_in)-np.tan(beta_out))**2*np.cos(beta_out)**2/np.cos(angle_m)
    
    #Empirical parameter (0 for stator, 0.37 for shrouded rotor)
    if cascade_type == 'stator':
        B = 0
    elif cascade_type == 'rotor':
        B = 0.37
    else:
        print("Specify the type of cascade")
       
    # Tip clearance loss coefficient
    Y_cl = B*Z*c/H*(t_cl/H)**0.78
    
    return Y_cl

def get_incidence_loss(flow_parameters, geometry, beta_des):
    
    # Computes the incidence coefficient for the current cascade according to the :cite'moustapha_incidence_1990'
    # flow_parameters is a dictionary containing kinetic and thermodynamic variables describing the flow through the current cascade
    # geometry is a dictionary with the geometrical parameters describing the current cascade
    # cascade_type is a string that states if the current cascade row is a stator or a rotor row 

    
    Ma_rel_out=flow_parameters["Ma_rel_out"]
    beta_in=flow_parameters["beta_in"]
    gamma = flow_parameters["gamma"]

    s=geometry["s"]
    le=geometry["le"]
    theta_in=geometry["theta_in"]
    theta_out = geometry["theta_out"]
    
    chi = get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des)
    
    if abs(chi)>800:
        raise Warning("Incidence paameter out of range: chi = {chi}")
    
    if chi >= 0:
        dPhi = 0.778e-5*chi+0.56e-7*chi**2+0.4e-10*chi**3+2.054e-19*chi**6 
    elif chi < 0:
        dPhi = -5.1734e-6*chi+7.6902e-9*chi**2 
        
    Y_inc = convert_kinetic_energy_coefficient(dPhi, gamma, Ma_rel_out)
        
    return Y_inc

def get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des):
    
    # TODO: add docstring explaning the equations and the original paper
    # TODO: possibly include the equation number (or figure number) of the original paper
    # TODO: explain smoothing/blending tricks

    chi = (le/s)**(-1.6)*(np.cos(theta_in)/np.cos(theta_out))**(-2)*(beta_in-beta_des)*180/np.pi
    return chi
    
def convert_kinetic_energy_coefficient(dPhi,gamma,Ma_rel_out):

    # TODO: this conversion assumes that the fluid is a perfect gas
    # TODO: we can create a more general function to convert between different loss coefficient deffinitions
    #   TODO: stagation pressure loss coefficient
    #   TODO: enthalpy loss coefficient (two-variants)
    #   TODO: kinetic energy loss coefficient
    #   TODO: entropy loss coefficient
    # Given the fluid state, I believe it is possible to convert between all of them

    denom = 1-(1+(gamma-1)/2*Ma_rel_out**2)**(-gamma/(gamma-1))
    numer =  (1-(gamma-1)/2*Ma_rel_out**2*(1/(1-dPhi)-1))**(-gamma/(gamma-1))-1
    
    Y = numer/denom
     
    return Y
