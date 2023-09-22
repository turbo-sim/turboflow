# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:37:42 2023

@author: lasseba
"""


import numpy as np

def calculate_loss_coefficient(cascade_data):
    
    # Load kinetic variables from turbine
    beta_out=cascade_data["flow"]["beta_out"]
    beta_in=cascade_data["flow"]["beta_in"]
    delta = cascade_data["flow"]["delta"]
    Ma_rel_out=cascade_data["flow"]["Ma_rel_out"]
    gamma = cascade_data["flow"]["gamma_out"]
    
    # Load geometrical variables from turbine
    s=cascade_data["geometry"]["s"] # Pitch
    c=cascade_data["geometry"]["c"] # Chord
    theta_in=cascade_data["geometry"]["theta_in"] # Inlet metal angle
    theta_out = cascade_data["geometry"]["theta_out"]
    b=cascade_data["geometry"]["b"] # Axial chord
    h=cascade_data["geometry"]["H"] # Mean bade height
    xi = cascade_data["geometry"]["xi"] # Stagger
    d = cascade_data["geometry"]["le"] # Leading edge thickness
    We = cascade_data["geometry"]["We"] # Wedge angle
    cascade_type=cascade_data["type"]
    t_cl = cascade_data["geometry"]["t_cl"]
    
    YpKO = Yp_KO(cascade_data)
    
    beta_des = theta_in # Assume zero incidence at design 
    
    chi = Chi(d,s,We,theta_in,theta_out,beta_in,beta_des)
        
    dPhip = dPhi_p(chi)
    
    Y_inc = coefficient_conversion(dPhip,gamma,Ma_rel_out)
        
    Ymid = YpKO+Y_inc
    
    CR = np.cos(beta_in)/np.cos(beta_out) # Convergence ratio from Benner et al.[2006]
    
    BSx = b/s # Axial blade solidity
    BL_rel = delta # Boundary layer displacement thickness relative to blade height
    AR = h/c # Aspect ratio
    
    ZTE = Z_TE(CR,AR,BSx,beta_in,beta_out,BL_rel) # ZTE/h
    ZTE = min(ZTE, 0.99)
    
    Yp = Ymid*(1-ZTE)
    
    Ysec = Y_sec(CR,AR,beta_out,xi,BL_rel)
    
    Ycl = Y_cl(beta_in, beta_out, cascade_type, t_cl, c, h)
    
    Y = Yp+Ysec+Ycl
    
    # Comments
    # What is beta_des: Assume zero incidence at design 
    # How to handle design profile loss and off-design profile loss: sum of KO and Benner
    # How to get boundary layer displacement thickness
    # Get in leading edge diameter and wedge angle as design parameters
    
    
    loss_dict = {"Profile" : Yp,
                 "Secondary" : Ysec,
                 "Clearance" : Ycl,
                 "Total" : Y}
    
    
    return [Y, loss_dict]
    

def Yp_KO(cascade_data):
    
    # Load kinetic variables from turbine
    Re=cascade_data["flow"]["Re_out"]
    Ma_rel_out=cascade_data["flow"]["Ma_rel_out"]
    Ma_rel_in=cascade_data["flow"]["Ma_rel_in"]
    beta_out=cascade_data["flow"]["beta_out"]
    p0rel_in=cascade_data["flow"]["p0_rel_in"]
    p_in=cascade_data["flow"]["p_in"]
    p0rel_out=cascade_data["flow"]["p0_rel_out"]
    p_out=cascade_data["flow"]["p_out"]
    
    # Load geometrical variables from turbine
    s=cascade_data["geometry"]["s"]
    c=cascade_data["geometry"]["c"]
    t_max=cascade_data["geometry"]["t_max"]
    theta_in=cascade_data["geometry"]["theta_in"]
    r_ht_in=cascade_data["geometry"]["r_ht_in"]
    o=cascade_data["geometry"]["o"]
    t_te=cascade_data["geometry"]["te"]

    cascade_type=cascade_data["type"]
    
    
    f_Re=(Re/2e5)**(-0.4)*(Re<2e5)+1*(Re >= 2e5 and Re <= 1e6) + (Re/1e6)**(-0.2)*(Re>1e6)
    f_Ma=1+60*(Ma_rel_out-1)**2*(Ma_rel_out > 1)  
    
    fhub=f_hub(r_ht_in,cascade_type)
    
    a=max(0,fhub*Ma_rel_in-0.4)
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
    
    Yte = Y_te(theta_in,beta_out,t_te,o)    
    
    return Yp+Yte
    
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

def f_hub(r_ht,cascade_type):
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

def K_p(Ma_rel_in,Ma_rel_out):
    K1=1*(Ma_rel_out < 0.2) + (1-1.25*(Ma_rel_out-0.2))*(Ma_rel_out > 0.2 and Ma_rel_out < 1.00)
    K2=(Ma_rel_in/Ma_rel_out)**2
    Kp=1-K2*(1-K1)
    Kp=max(0.1,Kp)
    return [Kp,K2,K1]

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

def coefficient_conversion(dPhi,gamma,Ma_rel_out):
    denom = 1-(1+(gamma-1)/2*Ma_rel_out**2)**(-gamma/(gamma-1))
    numer =  (1-(gamma-1)/2*Ma_rel_out**2*(1/(1-dPhi)-1))**(-gamma/(gamma-1))-1
    
    Y = numer/denom
     
    return Y

def dPhi_p(Chi):
    
    if Chi >=0:
        a_vec = np.array([3.711e-7,-5.318e-6,1.106e-5,9.017e-5,-1.542e-4,-2.506e-4,1.327e-3,-6.149e-5])
    
        Chi_vec = np.array([Chi**8,Chi**7,Chi**6,Chi**5,Chi**4,Chi**3,Chi**2,Chi])
    else:
        a_vec = np.array([1.358e-4,-8.72e-4])
        Chi_vec = np.array([Chi**2,Chi])
    
    dPhi_p = np.sum(a_vec*Chi_vec)
    
    return dPhi_p

def Chi(d,s,We,theta_in,theta_out,beta_in,beta_des):
    Chi = (d/s)**(-0.05)*(We*180/np.pi)**(-0.2)*(np.cos(theta_in)/np.cos(theta_out))**(-1.4)*(beta_in-beta_des)*180/np.pi
    return Chi

def Z_TE(CR,AR,BSx,beta_in,beta_out,BL_rel):
        
    Ft = F_t(BSx,beta_in,beta_out)
                
    Z_TE = 0.10*Ft**0.79/np.sqrt(CR)/(AR)**0.55+32.70*(BL_rel)**2
    
    return Z_TE
    
def F_t(BSx,beta_in,beta_out):
    
    a_m = np.arctan(0.5*(np.tan(beta_in)+np.tan(beta_out)))
        
    F_t = 2*1/BSx*np.cos(a_m)**2*(abs(np.tan(beta_in))+abs(np.tan(beta_out)))
    
    return F_t

def Y_sec(CR,AR,beta_out,xi,BL_rel):
        
    
    if AR <= 2:
        denom = np.sqrt(np.cos(xi))*CR*AR**0.55*(np.cos(beta_out)/(np.cos(xi)))**0.55
        Y_sec = (0.038+0.41*np.tanh(1.2*BL_rel))/denom
    else:
        denom = np.sqrt(np.cos(xi))*CR*AR*(np.cos(beta_out)/(np.cos(xi)))**0.55
        Y_sec = (0.052+0.56*np.tanh(1.2*BL_rel))/denom
        
    return Y_sec