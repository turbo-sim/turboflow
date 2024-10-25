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
    pass

def compute_volute_losses(input_parameters):
    
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
    
    # Load parameters
    Cf = input_parameters["Cf"]
    r_out = input_parameters["geometry"]["radius_out"]
    r_in = input_parameters["geometry"]["radius_in"]
    b_in = input_parameters["geometry"]["width_in"]

    # Store losses in dict
    loss_dict = {
        "loss_total" : 2*Cf*(r_out - r_in)/b_in
    }

    return loss_dict

def compute_impeller_losses(input_parameters):
    
    # Load inlet plane
    v_in = input_parameters["inlet_plane"]["v"]
    v_m_in = input_parameters["inlet_plane"]["v_m"]
    w_in = input_parameters["inlet_plane"]["w"]
    w_t_in = input_parameters["inlet_plane"]["w_t"]
    d_in = input_parameters["inlet_plane"]["d"]
    h0_out = input_parameters["inlet_plane"]["h0"]

    # Load exit plane
    v_out = input_parameters["exit_plane"]["v"]
    v_t_out = input_parameters["exit_plane"]["v_t"]
    w_out = input_parameters["exit_plane"]["w"]
    u_out = input_parameters["exit_plane"]["blade_speed"]
    d_out = input_parameters["exit_plane"]["d"]
    h0_out = input_parameters["exit_plane"]["h0"]

    # Load geometry
    t_cl = input_parameters["geometry"]["tip_clearance"]
    b_out = input_parameters["geometry"]["width_out"]
    L_b = input_parameters["geometry"]["impeller_length"]
    D_h = input_parameters["geometry"]["diameter_hydraulic"]
    z = input_parameters["geometry"]["number_of_blades"]
    r_in_hub = input_parameters["geometry"]["radius_hub_in"]
    r_in_tip = input_parameters["geometry"]["radius_tip_in"]
    r_out = input_parameters["geometry"]["radius_out"]

    # Load coefficients
    Cf = input_parameters["Cf"] 

    # Incidence loss
    f_inc = 0.6
    Y_inc = f_inc*w_t_in**2/2

    # Blade loading loss
    Df = 1 - w_out/w_in + 0.75*(h0_out - h0_in)*w_out/((z/np.pi*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out**2)
    Y_bld = 0.05*Df**2*u_out**2

    # Skin friction loss
    w_avg = (v_in + v_out + 3*w_in + 3*w_out)/8
    Y_sf = 2*Cf*L_b/D_h*w_avg**2

    # Clearance loss
    Y_cl = 0.6*t_cl/b_out*v_t_out*(4*np.pi/(b_out*z)*(r_in_tip**2 - r_in_hub**2)/((r_out - r_in_tip)(1+d_out/d_in))*v_t_out*v_m_in)**(0.5)

