import numpy as np
from .. import math

def compute_losses(input_parameters, component):

    # Map loss model functions
    component_function = {
        "impeller" : compute_impeller_losses,
        "vaneless_diffuser" : compute_vaneless_diffuser_losses,
        "vaned_diffuser" : compute_vaned_diffuser_losses,
        "volute" : compute_volute_losses,
    }   

    loss_dict = component_function[component](input_parameters)

    return loss_dict

def compute_vaned_diffuser_losses(input_parameters):
    
    """
    Conrad (See Meroni paper)
    """

    # Load parameters
    r_in = input_parameters["geometry"]["radius_in"]
    r_out = input_parameters["geometry"]["radius_out"]
    theta_in = input_parameters["geometry"]["leading_edge_angle"]
    theta_out = input_parameters["geometry"]["trailing_edge_angle"]
    b_out = input_parameters["geometry"]["width_out"]
    b_throat = input_parameters["geometry"]["width_throat"]
    opening = input_parameters["geometry"]["opening"]
    pitch = input_parameters["geometry"]["pitch"]
    v_in = input_parameters["inlet_plane"]["v"]
    v_out = input_parameters["exit_plane"]["v"]
    v_m_out = input_parameters["exit_plane"]["v_m"]
    alpha_out = input_parameters["exit_plane"]["alpha"]
    Cf = input_parameters["factors"]["skin_friction"]

    # Skin friction
    L_b = (r_out-r_in)/math.cosd((theta_in+theta_out)/2)
    D_h = opening*b_throat/(opening+b_throat) + pitch*b_out/(pitch+b_out) 
    Y_sf = 2*Cf*v_m_out**2*L_b/D_h    

    # Incidence
    alpha_des = theta_out
    Y_inc = 0.6*math.sind(abs(alpha_out-alpha_des))*0.5*v_out**2

    # Total losses
    scale = 0.5*v_in**2
    Y_inc = Y_inc/scale
    Y_sf = Y_sf/scale

    # Store losses in dict
    loss_dict = {"incidence" : Y_inc,
                 "skin_friction" : Y_sf,
                 "loss_total" : Y_inc + Y_sf
                 }

    return loss_dict

def compute_volute_losses(input_parameters):

    """
    Roberto's volute losses
    """
    
    # Load parameters
    alpha_in = input_parameters["inlet_plane"]["alpha"]
    A_in = input_parameters["geometry"]["area_in"]
    A_scroll = input_parameters["geometry"]["area_scroll"]
    A_out = input_parameters["geometry"]["area_out"]

    # Calculate dissipation of radial kinetic energy
    zeta_rad = math.cosd(alpha_in)**2

    # Calculate dissipation of tangential kinetic energy due to expansion loss in scroll
    zeta_scroll = math.sind(alpha_in)**2*(1-A_in*math.sind(alpha_in)/A_scroll)**2

    # Calculate dissipation of tangential kinetic energy due to expansion loss in cone
    zeta_cone = math.cosd(alpha_in)**2*(A_in/A_scroll)**2*(1-A_scroll/A_out)**2

    # Store losses in dict
    loss_dict = {"loss_radial" : zeta_rad,
                 "loss_scroll" : zeta_scroll,
                 "loss_cone" : zeta_cone,
                 "loss_total" : zeta_rad + zeta_scroll + zeta_cone}

    return loss_dict


def compute_vaneless_diffuser_losses(input_parameters):

    """
    Roberto's vanless diffuser losses
    """
    
    # Load parameters
    Cf = input_parameters["factors"]["skin_friction"]
    r_out = input_parameters["geometry"]["radius_out"]
    r_in = input_parameters["geometry"]["radius_in"]
    b_in = input_parameters["geometry"]["width_in"]

    # Store losses in dict
    loss_dict = {
        "loss_total" : 2*Cf*(r_out - r_in)/b_in
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
    w_m_in = input_parameters["inlet_plane"]["w_m"]
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
    L_ax = input_parameters["geometry"]["impeller_length"]
    z = input_parameters["geometry"]["number_of_blades"]
    r_in_hub = input_parameters["geometry"]["radius_hub_in"]
    r_in_tip = input_parameters["geometry"]["radius_tip_in"]
    r_out = input_parameters["geometry"]["radius_out"]
    A_in = input_parameters["geometry"]["area_in"]
    A_throat = input_parameters["geometry"]["area_throat"]

    # Load model factors
    # Cf = input_parameters["factors"]["skin_friction"] 
    wake_width = input_parameters["factors"]["wake_width"]

    # Incidence loss
    f_inc = 0.6
    Y_inc = f_inc*w_t_in**2/2 

    # Blade loading loss
    Df = 1 - w_out/w_in + 0.75*(h0_out - h0_in)*w_out/((z/np.pi*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out**2)
    Y_bld = 0.05*Df**2*u_out**2 

    # Skin friction loss
    L_b = np.pi/8*(r_out*2-(r_in_tip + r_in_hub)-b_out+2*L_ax)*(2/(math.cosd(beta_in)+math.cosd(beta_out)))
    D_h = 2*r_out*math.cosd(beta_out)/(z/np.pi + 2*r_out*math.cosd(beta_out)/b_out) + 0.5*(r_in_tip+r_in_hub)/r_out*math.cosd(beta_in)/(z/np.pi + (r_in_tip + r_in_hub)/(r_in_tip-r_in_hub)*math.cosd(beta_in))
    w_avg = (v_in + v_out + 3*w_in + 3*w_out)/8
    Re = u_out*D_h*d_in/mu_in
    Cf = 0.0412*Re**(-0.1925)
    Y_sf = 2*Cf*L_b/D_h*w_avg**2

    # Clearance loss
    Y_cl = 0.6*t_cl/b_out*v_t_out*(4*np.pi/(b_out*z)*(r_in_tip**2 - r_in_hub**2)/((r_out - r_in_tip)*(1+d_out/d_in))*v_t_out*v_m_in)**(0.5)

    # Mixing loss
    width_diffuser = b_out # Assume no change in width from impeller exit to diffuser inlet
    b_star = width_diffuser/b_out
    Y_mix = 1/(1+math.tand(alpha_out)**2)*((1-wake_width-b_star)/(1-wake_width))**2*v_out**2/2

    # Shock loss models
    w_th = w_m_in*A_in/A_throat
    Y_shock = 0.56*max(0,((w_th/a_in)**2-1)**3)

    # Total losses
    scale = 0.5*v_in**2
    Y_inc = Y_inc/scale
    Y_bld = Y_bld/scale
    Y_sf = Y_sf/scale
    Y_cl = Y_cl/scale
    Y_mix = Y_mix/scale
    Y_tot = Y_inc + Y_bld + Y_sf + Y_cl + Y_mix + Y_shock

    # Scale loss coefficent
    loss_dict = {"incidence" : Y_inc,
                 "blade_loading" : Y_bld,
                 "skin_friction" : Y_sf,
                 "tip_clearance" : Y_cl,
                 "wake_mixing" : Y_mix,
                 "shock" : Y_shock,
                 "loss_total" : Y_tot}
    
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
    L_ax = input_parameters["geometry"]["impeller_length"]
    z = input_parameters["geometry"]["number_of_blades"]
    r_in_tip = input_parameters["geometry"]["radius_tip_in"]
    r_out = input_parameters["geometry"]["radius_out"]
    r_mean_in = input_parameters["geometry"]["radius_mean_in"]

    # Recirculation losses
    Df = 1 - w_out/w_in + 0.75*(h0_out - h0_in)*w_out/((z/np.pi*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out**2)
    Y_rc = 8e-5*np.sinh(3.5*(alpha_out*np.pi/180)**3)*Df**2*u_out**2

    # Disk friction losses
    Re = d_out*u_out*r_out/mu_out
    f_df = 2.67/Re**0.5*(Re<3e5) + 0.0622/Re**0.2*(Re>=3e5)
    d_avg = (d_in + d_out)/2
    Y_df = f_df*d_avg*r_out**2*u_out**3/(f*mass_flow_rate)

    # Leakage loss
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