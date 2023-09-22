# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:25:28 2023

@author: lasseba
"""

import numpy as np


def calculate_loss_coefficient(cascade_data):
    
    kinetics = cascade_data["flow"]
    geometry = cascade_data["geometry"]
    
    # Compute the loss coefficient using the Kacker-Okapuu loss model
    
    # Load kinetic variables from turbine
    Re=kinetics["Re_out"]
    Ma_rel_out=kinetics["Ma_rel_out"]
    Ma_rel_in=kinetics["Ma_rel_in"]
    beta_out=kinetics["beta_out"]
    beta_in=kinetics["beta_in"]
    p0rel_in=kinetics["p0_rel_in"]
    p_in=kinetics["p_in"]
    p0rel_out=kinetics["p0_rel_out"]
    p_out=kinetics["p_out"]
    
    # Load geometrical variables from turbine
    s=geometry["s"]
    c=geometry["c"]
    t_max=geometry["t_max"]
    # theta_out=geometry["theta_out"]
    theta_in=geometry["theta_in"]
    r_ht_in=geometry["r_ht_in"]
    b=geometry["b"]
    H=geometry["H"]
    t_cl=geometry["t_cl"]
    t_te=geometry["te"]
    o=geometry["o"]
    # A_in=geometry["A_in"]
    # A_out=geometry["A_out"]
    # r_m=geometry["r_m"]
    cascade_type=cascade_data["type"]
    
    
    # Reynolds number correction factor
    f_Re=(Re/2e5)**(-0.4)*(Re<2e5)+1*(Re >= 2e5 and Re <= 1e6) + (Re/1e6)**(-0.2)*(Re>1e6)
    
    # Mach number correction factor
    f_Ma=1+60*(Ma_rel_out-1)**2*(Ma_rel_out > 1)    
        
    # Profile loss coefficient
    # Inlet chock loss
    f_hub=fhub(r_ht_in,cascade_type)
    a=max(0,f_hub*Ma_rel_in-0.4)
    Y_shock=0.75*a**1.75*r_ht_in*(p0rel_in-p_in)/(p0rel_out-p_out)
    Y_shock = max(0,Y_shock)
    
    # Compressible flow correction factor
    Kp,K2,K1=K_p(Ma_rel_in, Ma_rel_out)
    
    # Yp_reaction and Yp_impulse according to Aungier correlation
    # These formulas are valid for 40<abs(angle_out)<80
    # Extrapolating outside of this limits might give completely wrong results
    # If the optimization algorithm has upper and lower bounds for the outlet 
    # angle there is no need to worry about this problem
    # angle_out_bis keeps the 40deg-losses for outlet angles lower than 40deg
    angle_out_bis= max(abs(beta_out),40*np.pi/180)
    Yp_reaction=nozzle_blades(s/c,angle_out_bis)
    Yp_impulse=impulse_blades(s/c,angle_out_bis)
    
    #Formula according to Kacker-Okapuu
    Yp=Yp_reaction-abs(theta_in/beta_out)*(theta_in/beta_out)*(Yp_impulse-Yp_reaction)

    # Limit the extrapolation of the profile loss to avoid negative values for
    # blade profiles with little deflection
    # Low limit to 80% of the axial entry nozzle profile loss
    # This value is completely arbitrary
    Yp=max(Yp,0.8*Yp_reaction)
    
    # Avoid unphysical effect on the thickness by defining the variable aa
    aa=max(0,-theta_in/beta_out)
    Yp=Yp*((t_max/c)/0.2)**aa
    Y_p=0.914*(2/3*Yp*Kp+Y_shock)
    
    # Corrected profile loss coefficient
    Yp=f_Re*f_Ma*Y_p
    
    # Secondary loss coefficient  
    K3=(b/H)**2
    Ks=1-K3*(1-Kp)
    Ks=max(0.1,Ks)
    angle_m=np.arctan((np.tan(beta_in)+np.tan(beta_out))/2)
    Z=4*(np.tan(beta_in)-np.tan(beta_out))**2*np.cos(beta_out)**2/np.cos(angle_m)
    far=(1-0.25*np.sqrt(abs(2-H/c)))/(H/c)*(H/c < 2)+1/(H/c)*(H/c >= 2)
    Ys=1.2*Ks*0.0334*far*Z*np.cos(beta_out)/np.cos(theta_in)
    
    # Tip clearance loss coefficient
    Ycl = Y_cl(beta_in,beta_out,cascade_type,t_cl,c,H)
    
    # Trailing edge loss coefficienct
    Yte=Y_te(theta_in,beta_out,t_te,o)
    
    Y=Yp+Ys+Ycl+Yte
    
    loss_dict = {"Profile" : Yp+Yte,
                 "Secondary" : Ys,
                 "Clearance" : Ycl,
                 "Total" : Y}
    
    return [Y, loss_dict]

def nozzle_blades(r_sc,angle_out):
    
    # Use Aungier correlation to compute the pressure loss coefficient
    # This correlation is a formula that reproduces the figures from the Ainley
    # and Mathieson original figures
    
    phi=90-angle_out*180/np.pi
    r_sc_min=(0.46+phi/77)*(phi < 30) + (0.614+phi/130)*(phi >= 30)
    X=r_sc-r_sc_min
    A=(0.025+(27-phi)/530)*(phi < 27) + (0.025+(27-phi)/3085)*(phi >= 27)
    B=0.1583-phi/1640
    C=0.08*((phi/30)**2-1)
    n=1+phi/30
    Yp_reaction= (A+B*X**2+C*X**3)*(phi < 30) + (A+B*abs(X)**n)*(phi >= 30)
    return Yp_reaction

def impulse_blades(r_sc,angle_out):
    
    # Use Aungier correlation to compute the pressure loss coefficient
    # This correlation is a formula that reproduces the figures from the Ainley
    # and Mathieson original figures
    
    phi=90-angle_out*180/np.pi
    r_sc_min=0.224+1.575*(phi/90)-(phi/90)**2
    X=r_sc-r_sc_min
    A=0.242-phi/151+(phi/127)**2
    B=(0.3+(30-phi)/50)*(phi < 30) + (0.3+(30-phi)/275)*(phi >=30)
    C=0.88-phi/42.4+(phi/72.8)**2
    Yp_impulse=A+B*X**2-C*X**3
    return Yp_impulse

def K_p(Ma_rel_in,Ma_rel_out):
    K1=1*(Ma_rel_out < 0.2) + (1-1.25*(Ma_rel_out-0.2))*(Ma_rel_out > 0.2 and Ma_rel_out < 1.00)
    K2=(Ma_rel_in/Ma_rel_out)**2
    Kp=1-K2*(1-K1)
    Kp=max(0.1,Kp)
    return [Kp,K2,K1]

def fhub(r_ht,cascade_type):
    if r_ht < 0.5:
        r_ht=0.5     # Numerical trick to prevent extrapolation
        
    r_ht_data=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    #Stator curve
    f_data_S = [1.4, 1.18, 1.05, 1.0, 1.0, 1.0]
    
    #Rotor curve
    f_data_R= [2.15, 1.7, 1.35, 1.12, 1.0, 1.0]
    
    if cascade_type == 'stator':
        f = np.interp(r_ht,r_ht_data,f_data_S)
    elif cascade_type == 'rotor':
        f = np.interp(r_ht,r_ht_data,f_data_R)
    else:
        print("Specify the type of cascade")
        
    return f

def Y_te(angle_in,angle_out,t_te,o):
    
    # Range of trailing edge to throat opening ratio
    r_to_data=[0, 0.2, 0.4]
    
    # Reacting blading
    phi_data_reaction=[0, 0.045, 0.15]
    
    # Impulse blading
    phi_data_impulse=[0, 0.025, 0.075]
    
    # Numerical trick to avoid too big r_to's
    r_to=min(0.4,t_te/o)
    
    # Interpolate data
    d_phi2_reaction = np.interp(r_to,r_to_data,phi_data_reaction)
    d_phi2_impulse = np.interp(r_to,r_to_data,phi_data_impulse)
    
    #Compute kinetic energy loss coefficient
    d_phi2=d_phi2_reaction-abs(angle_in/angle_out)*(angle_in/angle_out)*(d_phi2_impulse-d_phi2_reaction)

    # Limit the extrapolation of the trailing edge loss
    d_phi2=max(d_phi2,d_phi2_impulse/2)
    Y_te=1/(1-d_phi2)-1
    
    return Y_te

def Y_cl(beta_in,beta_out,cascade_type,t_cl,c,H):
    angle_m=np.arctan((np.tan(beta_in)+np.tan(beta_out))/2)
    Z=4*(np.tan(beta_in)-np.tan(beta_out))**2*np.cos(beta_out)**2/np.cos(angle_m)
    
    #Empirical parameter (0 for stator, 0.37 for shrouded rotor)
    if cascade_type == 'stator':
        B = 0
    elif cascade_type == 'rotor':
        B = 0.37
    else:
        print("Specify the type of cascade")
       
    # Tip clearance loss coefficient
    Ycl=B*Z*c/H*(t_cl/H)**0.78
    
    return Ycl
