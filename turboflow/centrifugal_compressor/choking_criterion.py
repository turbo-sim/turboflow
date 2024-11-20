
import numpy as np
import CoolProp as cp
from scipy.optimize._numdiff import approx_derivative
from .. import utilities as utils
from .. import math

CHOKING_CRITERIONS = [
    "no_throat",
    "evaluate_critical_throat",
    "evaluate_throat"
]


def evaluate_choking(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

    """
    Evaluate throat residuals
    """

    choking_criterion = model_options ["choking_criterion"]

    throat_functions = {
        "no_throat" : lambda *args: ({}, {}),
        "evaluate_critical_throat" : get_critical_throat,
        "evaluate_throat" : get_throat,
    }

    residuals, results = throat_functions[choking_criterion](independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat)

    return residuals, results

def evaluate_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat):

    """
    Evaluate throat based on w_throat. (See get_res_throat, but more detailed)
    """

    w_throat = w_throat[0]

    # Evaluate velocity triangle
    beta_throat = theta_throat
    w_m_throat = w_throat*math.cosd(beta_throat)
    w_t_throat = w_throat*math.sind(beta_throat)
    v_m_throat = w_m_throat
    v_t_throat = w_t_throat + u_throat
    v_throat = np.sqrt(v_t_throat**2 + v_m_throat**2)
    alpha_throat = math.arctand(v_t_throat/v_m_throat)

    velocity_triangle_throat = {"v_t" : v_t_throat,
                   "v_m" : v_m_throat,
                   "v" : v_throat,
                   "alpha" : alpha_throat,
                   "w_t" : w_t_throat,
                   "w_m" : w_m_throat,
                   "w" : w_throat,
                   "beta" : beta_throat,
                   "blade_speed" : u_throat,
                   }

    # Evaluate thermodynamic state
    h0_rel_throat = rothalpy + 0.5*u_throat**2 # Assume same radius at throat and inlet
    h_throat = h0_rel_throat - 0.5*w_throat**2
    h0_throat = h_throat + 0.5*v_throat**2 
    s_throat = s_in # Assume isentropic flow
    static_properties_throat = fluid.get_props(cp.HmassSmass_INPUTS, h_throat, s_throat)
    stagnation_properties_throat = fluid.get_props(cp.HmassSmass_INPUTS, h0_throat, s_throat)
    relative_properties_throat = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel_throat, s_throat)  
    d_throat = static_properties_throat["d"]
    a_throat = static_properties_throat["a"]

    # Evaluate maass flow rate and mach
    m_throat = d_throat*w_throat*A_throat
    Ma_throat = v_throat/a_throat
    Ma_rel_out = w_throat/a_throat

    # Evaluate residuals
    

    # Store throat results
    throat_plane = {
        **velocity_triangle_throat,
        **static_properties_throat,
        **utils.add_string_to_keys(stagnation_properties_throat, "0"),
        **utils.add_string_to_keys(relative_properties_throat, "0_rel"),
        "mass_flow_rate" : m_throat, 
        "Ma" : Ma_throat,
        "Ma_rel" : Ma_rel_out,
    }

    return throat_plane

def evaluate_throat_mass_flow(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat):
     
    """
    Wrapper of evaluate_throat, returning only the residuals. Used like get_res_throat
    """

    throat_results = evaluate_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat)

    return throat_results["mass_flow_rate"]

def evaluate_throat_residual(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat):
     
    """
    Wrapper of evaluate_throat, returning only the residuals. Used like get_res_throat
    """

    throat_results = evaluate_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat)

    return 1-throat_results["mass_flow_rate"]/mass_flow_rate

def get_mass_flow_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat, eps, f0):

    jac = approx_derivative(
        evaluate_throat_mass_flow,
        w_throat,
        abs_step = eps,
        f0 = f0,
        method="3-point",
        args = (
            fluid,
            mass_flow_rate,
            rothalpy,
            s_in,
            A_throat,
            theta_throat, 
            u_throat,
        )
    )

    return jac[0]


def get_residual_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat, eps, f0):
    
    """ 
    Evaluate gradient of the throat residuals.
    """
    
    jac = approx_derivative(
        evaluate_throat_residual,
        w_throat,
        abs_step = eps,
        f0 = f0,
        method="3-point",
        args = (
            fluid,
            mass_flow_rate,
            rothalpy,
            s_in,
            A_throat,
            theta_throat, 
            u_throat,
        )
    )

    return jac[0]

def get_critical_throat(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

    """
    Get gradient of get_res_throat at point x

    Impeller: rothalpy is rothalpy, w2 is relative flow and u_throat is throat blade speed
    Stationary components: rothalpy is the stagnation enthalpy, w2 is absolute velocity and u_throat is zero  
    """

    # Load reference values
    v_max = reference_values["v_max"]

    # Load independent variables
    w_throat = independents[f"w_throat_{component}"] * v_max

    # Load input
    rothalpy = inlet_state["rothalpy"]
    s_in = inlet_state["s_in"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load geometry
    A_throat = geometry["area_throat"]
    theta_throat = geometry["leading_edge_angle"] # Assume throat velocity aligns with inlet angle

    # Load model options
    rel_step_fd = model_options["rel_step_fd"]
    
    # Evaluate gradient
    w_throat = np.array([w_throat])
    critical_throat_plane = evaluate_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat)
    f0 = 1-critical_throat_plane["mass_flow_rate"]/mass_flow_rate
    eps = rel_step_fd * w_throat
    residual_gradient = get_residual_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat, eps, f0)

    # Store results
    throat_residual = {
        "mass_flow_residual_gradient" : residual_gradient,
    }

    throat_results = {
        **critical_throat_plane,
        "mass_flow_residual" : f0,
        "choked" : f0 > 0,
    }

    return throat_residual, throat_results

def get_throat(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

    """
    Get gradient of get_res_throat at point x

    Impeller: rothalpy is rothalpy, w2 is relative flow and u_throat is throat blade speed
    Stationary components: rothalpy is the stagnation enthalpy, w2 is absolute velocity and u_throat is zero  
    """

    # Load reference values
    v_max = reference_values["v_max"]

    # Load independent variables
    w_throat = independents[f"w_throat_{component}"]* v_max

    # Load input
    rothalpy = inlet_state["rothalpy"]
    s_in = inlet_state["s_in"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load geometry
    A_throat = geometry["area_throat"]
    theta_throat = geometry["leading_edge_angle"] # Assume throat velocity aligns with inlet angle

    # Load model options
    rel_step_fd = model_options["rel_step_fd"]
        
    # Evaluate mass flow rate gradient
    w_throat = np.array([w_throat])
    throat_plane = evaluate_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat)
    f0 = throat_plane["mass_flow_rate"]
    eps = rel_step_fd * w_throat
    mass_flow_gradient = get_mass_flow_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, theta_throat, u_throat, eps, f0)

    # Evaluate mass flow residual
    mass_flow_residual = 1-throat_plane["mass_flow_rate"]/mass_flow_rate

    # Store results
    throat_residual = {
        "mass_flow_gradient_product" : mass_flow_gradient*mass_flow_residual,
    }

    throat_results = {
        **throat_plane,
        "mass_flow_residual" : mass_flow_residual,
        "choked" : abs(mass_flow_residual) > 1e-4, # Choked solution if mass flow residual is clearly positive
        "correct_solution" : mass_flow_residual > -1e-4, # Correct solution if mass flow residual is non-negative  
    }

    return throat_residual, throat_results



