
import numpy as np
from .. import math

def compute_losses(input_parameters, component):

    # Map loss model functions
    component_function = {
        "impeller" : compute_impeller_losses,
        "vaneless_diffuser" : compute_vaneless_diffuser_losses,
    }   

    loss_dict = component_function[component](input_parameters)

    return loss_dict


def compute_vaneless_diffuser_losses(input_parameters):

    """
    Oh's vanless diffuser losses
    """
    
    # Load parameters
    T0_in = input_parameters["inlet_plane"]["T0"]
    p0_in = input_parameters["inlet_plane"]["p0"]
    v_in = input_parameters["inlet_plane"]["v"]
    cp = input_parameters["inlet_plane"]["cp"]
    cv = input_parameters["inlet_plane"]["cv"]
    p_out = input_parameters["exit_plane"]["p"]
    p0_out = input_parameters["exit_plane"]["p0"]

    # Calculate vaneless diffuser losses (Stanitz)
    gamma = cp/cv
    alpha = (gamma - 1)/gamma
    Y_tot = cp*T0_in*((p_out/p0_out)**alpha - (p_out/p0_in)**alpha)

    # Store losses in dict
    scale = 0.5*v_in**2
    loss_dict = {
        "loss_total" : Y_tot/scale
    }

    return loss_dict

def compute_impeller_losses(input_parameters):

    """
    Oh. Impeller losses
    """
    
    # Load inlet plane
    v_in = input_parameters["inlet_plane"]["v"]
    v_m_in = input_parameters["inlet_plane"]["v_m"]
    w_in = input_parameters["inlet_plane"]["w"]
    w_t_in = input_parameters["inlet_plane"]["w_t"]
    beta_in = input_parameters["inlet_plane"]["beta"]
    d_in = input_parameters["inlet_plane"]["d"]
    h0_in = input_parameters["inlet_plane"]["h0"]
    a_in = input_parameters["inlet_plane"]["a"]
    mu_in = input_parameters["inlet_plane"]["mu"]

    # Load exit plane
    v_out = input_parameters["exit_plane"]["v"]
    v_t_out = input_parameters["exit_plane"]["v_t"]
    alpha_out = input_parameters["exit_plane"]["alpha"]
    beta_out = input_parameters["exit_plane"]["beta"]
    w_out = input_parameters["exit_plane"]["w"]
    u_out = input_parameters["exit_plane"]["blade_speed"]
    d_out = input_parameters["exit_plane"]["d"]
    h0_out = input_parameters["exit_plane"]["h0"]
    a_out = input_parameters["exit_plane"]["a"]

    # Load geometry
    t_cl = input_parameters["geometry"]["tip_clearance"]
    b_out = input_parameters["geometry"]["width_out"]
    L_ax = input_parameters["geometry"]["length_axial"]
    z = input_parameters["geometry"]["number_of_blades"]
    r_in_hub = input_parameters["geometry"]["radius_hub_in"]
    r_in_tip = input_parameters["geometry"]["radius_tip_in"]
    r_out = input_parameters["geometry"]["radius_out"]

    # Load model factors
    wake_width = input_parameters["factors"]["wake_width"]

    # Incidence loss (Conrad)
    f_inc = 0.5
    Y_inc = f_inc*0.5*w_t_in**2

    # Blade loading loss (Coppage)
    Df = 1 - w_out/w_in + 0.75*(h0_out - h0_in)*w_out/((z/np.pi*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out**2)
    Y_bld = 0.05*Df**2*u_out**2

    # Skin friction loss (Jansen)
    L_b = np.pi/8*(r_out*2-(r_in_tip + r_in_hub)-b_out+2*L_ax)*(2/(math.cosd(beta_in)+math.cosd(beta_out)))
    D_h = 2*r_out*math.cosd(beta_out)/(z/np.pi + 2*r_out*math.cosd(beta_out)/b_out) + 0.5*(r_in_tip+r_in_hub)/r_out*math.cosd(beta_in)/(z/np.pi + (r_in_tip + r_in_hub)/(r_in_tip-r_in_hub)*math.cosd(beta_in))
    w_avg = (v_in + v_out + 3*w_in + 3*w_out)/8
    Re = u_out*D_h*d_in/mu_in
    Cf = 0.0412*Re**(-0.1925)
    Y_sf = 2*Cf*L_b/D_h*w_avg**2

    # Clearance loss (Jansen)
    Y_cl = 0.6*t_cl/b_out*v_t_out*(4*np.pi/(b_out*z)*(r_in_tip**2 - r_in_hub**2)/((r_out - r_in_tip)*(1+d_out/d_in))*v_t_out*v_m_in)**(0.5)

    # Mixing loss (Johnston and Dean)
    width_diffuser = b_out # Assume no change in width from impeller exit to diffuser inlet
    b_star = width_diffuser/b_out
    Y_mix = 1/(1+math.tand(alpha_out)**2)*((1-wake_width-b_star)/(1-wake_width))**2*0.5*v_out**2

    # Parasitic losses
    parasitic_losses = compute_parasitic_losses(input_parameters)
    Y_lk = parasitic_losses["leakage"]
    Y_df = parasitic_losses["disk_friction"]
    Y_rc = parasitic_losses["recirculation"]

    # Total losses
    scale = 0.5*v_in**2
    Y_inc = Y_inc/scale
    Y_bld = Y_bld/scale
    Y_sf = Y_sf/scale
    Y_cl = Y_cl/scale
    Y_mix = Y_mix/scale

    Y_lk = Y_lk/scale
    Y_df = Y_df/scale
    Y_rc = Y_rc/scale

    Y_parasitic = Y_lk + Y_df + Y_rc
    Y_tot = Y_inc + Y_bld + Y_sf + Y_cl + Y_mix + Y_parasitic

    delta_W = 2*np.pi*2*r_out*v_t_out/(z*L_b)
    W_max = (w_in + w_out+delta_W)/2
    D_eq = W_max/w_out

    # Scale loss coefficent
    loss_dict = {"incidence" : Y_inc,
                 "blade_loading" : Y_bld,
                 "skin_friction" : Y_sf,
                 "tip_clearance" : Y_cl,
                 "wake_mixing" : Y_mix,
                 "leakage" : Y_lk,
                 "recirculation" : Y_rc,
                 "disk_friction" : Y_df,
                 "D_eq" : D_eq,
                 "loss_total" : Y_tot,
                 }
    
    return loss_dict

def compute_parasitic_losses(input_parameters):

    """
    Oh's parasitic losses
    """

    # Load boundary conditions 
    mass_flow_rate = input_parameters["boundary_conditions"]["mass_flow_rate"]

    # Load inlet plane
    v_t_in = input_parameters["inlet_plane"]["v_t"]
    w_in = input_parameters["inlet_plane"]["w"]
    d_in = input_parameters["inlet_plane"]["d"]
    h0_in = input_parameters["inlet_plane"]["h0"]
    u_in = input_parameters["inlet_plane"]["blade_speed"]

    # Load exit plane
    v_t_out = input_parameters["exit_plane"]["v_t"]
    alpha_out = input_parameters["exit_plane"]["alpha"]
    w_out = input_parameters["exit_plane"]["w"]
    u_out = input_parameters["exit_plane"]["blade_speed"]
    d_out = input_parameters["exit_plane"]["d"]
    h0_out = input_parameters["exit_plane"]["h0"]
    mu_out = input_parameters["exit_plane"]["mu"]

    # Load geometry
    t_cl = input_parameters["geometry"]["tip_clearance"]
    b_out = input_parameters["geometry"]["width_out"]
    b_in = input_parameters["geometry"]["width_in"]
    L_ax = input_parameters["geometry"]["length_axial"]
    L_m = input_parameters["geometry"]["length_meridional"]
    z = input_parameters["geometry"]["number_of_blades"]
    r_in_tip = input_parameters["geometry"]["radius_tip_in"]
    r_out = input_parameters["geometry"]["radius_out"]
    r_mean_in = input_parameters["geometry"]["radius_mean_in"]

    # Recirculation losses (Oh)
    # Df = 1 - w_out/w_in + 0.75*(h0_out - h0_in)*w_out/((z/np.pi*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out**2)
    # Y_rc = 8e-5*np.sinh(3.5*(alpha_out*np.pi/180)**3)*Df**2*u_out**2

    # # Recirculation losses (Coppage)
    Df = 1 - w_out/w_in + 0.75*(abs(u_out*v_t_out) - abs(u_in*v_t_in))*w_out/((z/np.pi*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out**2)
    Y_rc = 0.02*np.sqrt(math.tand(alpha_out))*Df**2*u_out**2 # Error in exponent in paper?


    # Disk friction losses (Dailey and Nece)
    Re = d_out*u_out*r_out/mu_out
    f_df = 2.67/Re**0.5*(Re<3e5) + 0.0622/Re**0.2*(Re>=3e5)
    d_avg = (d_in + d_out)/2
    Y_df = f_df*d_avg*r_out**2*u_out**3/(4*mass_flow_rate)

    # Leakage loss(Aungier)
    r_avg = (r_mean_in + r_out)/2
    b_avg = (b_in + b_out)/2 
    delta_p = mass_flow_rate*(r_out*v_t_out - r_mean_in*v_t_in)/(z*r_avg*b_avg*L_m)
    u_cl = 0.816*np.sqrt(2*delta_p/d_out)
    m_cl = d_out*z*t_cl*L_m*u_cl
    Y_lk = m_cl*u_cl*u_out/(2*mass_flow_rate)

    loss_dict = {"recirculation" : Y_rc,
                 "disk_friction" : Y_df,
                 "leakage" : Y_lk,
                 "loss_total" : Y_rc + Y_df + Y_lk,
                 }
    
    return loss_dict