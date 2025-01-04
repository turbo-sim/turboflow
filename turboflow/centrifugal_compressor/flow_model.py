
import numpy as np
import pandas as pd
import CoolProp as cp
from . import loss_model as lm
from . import slip_model as sm
from . import choking_criterion as cm
from .. import utilities as utils
from .. import math
import scipy.linalg
import scipy.integrate
from scipy.optimize._numdiff import approx_derivative

def evaluate_centrifugal_compressor(
    variables,
    boundary_conditions,
    geometry,
    fluid,
    model_options,
    reference_values,
):

    # Initialize results structure
    compressor_results = {"residuals" : {},
                          "planes" : pd.DataFrame(),
                          "boundary_conditions" : boundary_conditions}

    # Map functions for each component
    component_function = {
        "impeller" : evaluate_impeller,
        "vaneless_diffuser" : evaluate_vaneless_diffuser,
        "vaned_diffuser" : evaluate_vaned_diffuser,
        "volute" : evaluate_volute,
    }

    # Prepare first set of inputs
    input = {"h0_in" : boundary_conditions["h0_in"],
             "s_in" : boundary_conditions["s_in"],
             "alpha_in" : boundary_conditions["alpha_in"],
             }

    # Evaluate components
    for key in geometry.keys():
        # Prepare set of independent variables
        independents = {var : val for var, val in variables.items() if var.endswith(key)}

        # Evaluate component
        results, residuals = component_function[key](independents, input, boundary_conditions, geometry[key], fluid, model_options, reference_values)

        # Store result in result structure
        compressor_results[key] = results
        compressor_results["planes"].loc[len(compressor_results["planes"]), results["inlet_plane"].keys()] = results["inlet_plane"]
        compressor_results["planes"].loc[len(compressor_results["planes"]), results["exit_plane"].keys()] = results["exit_plane"]

        residuals = utils.add_string_to_keys(residuals, f"_{key}")
        compressor_results["residuals"].update(residuals)

        # Prepare calculation of the subsequent component
        input = compressor_results["planes"].loc[len(compressor_results["planes"])-1].to_dict()

    # Compute overall performance
    compressor_results["overall"] = compute_overall_performance(compressor_results, fluid, geometry)

    return compressor_results

def evaluate_impeller(independents, input, boundary_conditions, geometry, fluid, model_options, reference_values):
    """
    Evaluate impeller performance
    - Deviation calculated according to slip factor
    - Losses calculated according to correlation for loss coefficient
    """

    # Load reference_values
    v_max = reference_values["v_max"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Load independent variables
    v_in = independents["v_in_impeller"]*v_max
    w_out = independents["w_out_impeller"]*v_max 
    beta_out = independents["beta_out_impeller"]*angle_range + angle_min 
    s_out = independents["s_out_impeller"]*s_range + s_min

    # Load input variables
    h0_in = input["h0_in"]
    s_in = input["s_in"]
    alpha_in = input["alpha_in"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]
    omega = boundary_conditions["omega"]

    # Load impeller geometry
    A_in = geometry["area_in"]
    A_out = geometry["area_out"]
    r_mean_in = geometry["radius_mean_in"]
    r_out = geometry["radius_out"]
    theta_out = geometry["trailing_edge_angle"]

    # Load model options
    slip_model = model_options ["slip_model"]
    loss_model = model_options ["loss_model"]["impeller"]

    # Evaluate inlet velocity triangle
    u_in = omega*r_mean_in
    v_m_in = v_in*math.cosd(alpha_in)
    v_t_in = v_in*math.sind(alpha_in)
    w_m_in = v_m_in
    w_t_in = v_t_in - u_in
    w_in = np.sqrt(w_t_in**2 + w_m_in**2)
    beta_in = math.arctand(w_t_in/w_m_in)

    velocity_triangle_in = {"v_t" : v_t_in,
                   "v_m" : v_m_in,
                   "v" : v_in,
                   "alpha" : alpha_in,
                   "w_t" : w_t_in,
                   "w_m" : w_m_in,
                   "w" : w_in,
                   "beta" : beta_in,
                   "blade_speed" : u_in,
                   }

    # Evaluate inlet thermodynamic state
    h_in = h0_in - 0.5*v_in**2
    h0_rel_in = h_in + 0.5*w_in**2
    static_properties_in = fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
    stagnation_properties_in = fluid.get_props(cp.HmassSmass_INPUTS, h0_in, s_in)
    relative_properties_in = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel_in, s_in)

    # Evaluate inlet mass flow rate, rothalpy and mach
    m_in = w_m_in*static_properties_in["d"]*A_in
    rothalpy = h0_rel_in - 0.5*u_in**2
    Ma_in = v_in/static_properties_in["a"]
    Ma_rel_in = w_in/static_properties_in["a"]

    # Evaluate exit velocity triangle
    u_out = omega*r_out
    w_m_out = w_out*math.cosd(beta_out)
    w_t_out = w_out*math.sind(beta_out)
    v_m_out = w_m_out
    v_t_out = w_t_out + u_out
    v_out = np.sqrt(v_t_out**2 + v_m_out**2)
    alpha_out = math.arctand(v_t_out/v_m_out)

    velocity_triangle_out = {"v_t" : v_t_out,
                   "v_m" : v_m_out,
                   "v" : v_out,
                   "alpha" : alpha_out,
                   "w_t" : w_t_out,
                   "w_m" : w_m_out,
                   "w" : w_out,
                   "beta" : beta_out,
                   "blade_speed" : u_out,
                   }

    # Evaluate inlet thermodynamic state
    h0_rel_out = rothalpy + 0.5*u_out**2
    h_out = h0_rel_out - 0.5*w_out**2
    h0_out = h_out + 0.5*v_out**2
    static_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
    stagnation_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h0_out, s_out)
    relative_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel_out, s_out)  
    isentropic_relative_properties_out = fluid.get_props(cp.PSmass_INPUTS, relative_properties_out["p"], s_in)
    isentropic_static_properties_out = fluid.get_props(cp.PSmass_INPUTS, static_properties_out["p"], s_in)
    isentropic_properties_out = {**isentropic_static_properties_out,**utils.add_string_to_keys(isentropic_relative_properties_out, "0")}

    # Evaluate exit mass flow rate and Mach 
    m_out = static_properties_out["d"]*w_m_out*A_out
    Ma_out = v_out/static_properties_out["a"]
    Ma_rel_out = w_out/static_properties_out["a"]

    # Evaluate throat residual
    input_choking = {
        "s_in" : s_in,
        "rothalpy" : rothalpy,
    }
    throat_residual, throat_plane = cm.evaluate_choking(independents, input_choking, boundary_conditions, geometry, fluid, model_options, reference_values, component="impeller", u_throat = u_in)

    # Collect inlet plane and exit plane results 
    inlet_plane = {**velocity_triangle_in,
                   **static_properties_in,
                   **utils.add_string_to_keys(stagnation_properties_in, "0"),
                   **utils.add_string_to_keys(relative_properties_in, "0_rel"),
                   "Ma" : Ma_in,
                   "Ma_rel" : Ma_rel_in,
                   }
    
    exit_plane = {**velocity_triangle_out,
                   **static_properties_out,
                   **utils.add_string_to_keys(stagnation_properties_out, "0"),
                   **utils.add_string_to_keys(relative_properties_out, "0_rel"),
                    "Ma" : Ma_out,
                   "Ma_rel" : Ma_rel_out,
                   }

    # Evaluate loss coefficient
    loss_input = {
        "inlet_plane" : inlet_plane,
        "exit_plane" : exit_plane,
        "throat_plane" : throat_plane,
        "geometry" : geometry,
        "isentropic" : isentropic_properties_out,
        "factors" : model_options["factors"],
        "boundary_conditions" : boundary_conditions,
    }
    loss_dict = lm.evaluate_loss_model(loss_model, loss_input, "impeller")
    exit_plane = {**exit_plane, **loss_dict}

    # Evaluate slip
    slip_velocity = u_out + v_m_out*math.tand(theta_out) - v_t_out
    slip_input = {"u_out" : u_out}
    slip_model = sm.evaluate_slip(slip_model, slip_input, geometry)

    # Evaluate residuals
    resiudals = {"mass_flow_in" : (m_in-mass_flow_rate)/mass_flow_rate,
                 "mass_flow_out" : (m_out-mass_flow_rate)/mass_flow_rate,
                 "losses" : loss_dict["loss_error"],
                 "slip" : (slip_velocity-slip_model)/v_max,
                 **throat_residual}

    # Store impeller results
    results = {"inlet_plane" : inlet_plane,
               "throat_plane" : throat_plane,
               "exit_plane": exit_plane,
    }

    return results, resiudals

def evaluate_vaned_diffuser(independents, input, boundary_conditions, geometry, fluid, model_options, reference_values):

    """
    Evaluate vaned diffuser
    - Deviation calculated according to aungier 
    - Losses calculated according to correlation for loss coefficient
    """

    # Load reference_values
    v_max = reference_values["v_max"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]

    # Load independent variables
    v_out = independents["v_out_vaned_diffuser"]*v_max
    s_out = independents["s_out_vaned_diffuser"]*s_range + s_min
    
    # Load input variables
    alpha_in = input["alpha"]
    h0_in = input["h0"]
    s_in = input["s"]

    # Load diffuser geometry
    theta_in = geometry["leading_edge_angle"]
    theta_out = geometry["trailing_edge_angle"]
    loc_camber_max = geometry["loc_camber_max"]
    camber = geometry["camber_angle"]
    solidity = geometry["solidity"]
    A_out = geometry["area_out"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load model options
    loss_model = model_options["loss_model"]["vaned_diffuser"]

    # Calculate incidence
    alpha_inc = alpha_in - theta_in

    # Calculate deviation angle and absolute flow angle
    # delta_0 = camber*(0.92*loc_camber_max**2 + 0.02*theta_out)/(np.sqrt(solidity) - 0.02*camber)
    # d_delta = np.exp(((1.5-(90-theta_in)/60)**2-3.3)*solidity)
    # alpha_out = theta_out + delta_0 + d_delta*alpha_inc
    # dev = {"delta_0" : delta_0,
    #     "alpha_inc" : alpha_inc,
        # "d_delta" : d_delta,
    #     }
    alpha_out = theta_out

    # Calculate tangential and meridional velocity
    v_m_out = v_out*math.cosd(alpha_out)
    v_t_out = v_out*math.sind(alpha_out)

    velocity_triangle_out = {"v_t" : v_t_out,
                   "v_m" : v_m_out,
                   "v" : v_out,
                   "alpha" : alpha_out,
                   }

    # Calculate thermophysical properties
    h0_out = h0_in
    h_out = h0_out - 0.5*v_out**2
    static_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
    stagnation_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h0_out, s_out)
    isentropic_stagnation_properties_out = fluid.get_props(cp.PSmass_INPUTS, stagnation_properties_out["p"], s_in)
    isentropic_static_properties_out = fluid.get_props(cp.PSmass_INPUTS, static_properties_out["p"], s_in)
    isentropic_properties_out = {**isentropic_static_properties_out,**utils.add_string_to_keys(isentropic_stagnation_properties_out, "0")}

    # Calculate mass flow rate and mach
    m_out = static_properties_out["d"]*v_m_out*A_out
    Ma_out = v_out/static_properties_out["a"]

    # Evaluate throat residual
    input_choking = {
        "s_in" : s_in,
        "rothalpy" : h0_in,
    }
    throat_residual, throat_results = cm.evaluate_choking(independents, input_choking, boundary_conditions, geometry, fluid, model_options, reference_values, component="vaned_diffuser")


    # Collect exit plane results
    exit_plane = {**velocity_triangle_out,
                   **static_properties_out,
                   **utils.add_string_to_keys(stagnation_properties_out, "0"),
                   "Ma" : Ma_out,
                   }

    # Evaluate loss coefficient
    loss_input = {
        "inlet_plane" : input,
        "exit_plane" : exit_plane,
        "geometry" : geometry,
        "isentropic" : isentropic_properties_out,
        "factors" : model_options["factors"],
        "boundary_conditions" : boundary_conditions,
    }
    loss_dict = lm.evaluate_loss_model(loss_model, loss_input, "vaned_diffuser")
    exit_plane = {**exit_plane, **loss_dict}

    # Evaluate residuals
    resiudals = {"mass_flow_out" : (m_out-mass_flow_rate)/mass_flow_rate,
                 "losses" : loss_dict["loss_error"],
                 **throat_residual
    }

    # Store diffuser results
    results = {"inlet_plane" : input,
               "throat_plane" : throat_results,
               "exit_plane": exit_plane,
               }
    
    return results, resiudals


def evaluate_vaneless_diffuser(independents, input, boundary_conditions, geometry, fluid, model_options, reference_values):

    vaneless_diffuser_model = model_options["vaneless_diffuser_model"]

    diffuser_function = {
        "algebraic" : evaluate_vaneless_diffuser_algebraic,
        "1D" : evaluate_vaneless_diffuser_1D,
    }

    if vaneless_diffuser_model in diffuser_function.keys():
        results, residual = diffuser_function[vaneless_diffuser_model](independents, input, boundary_conditions, geometry, fluid, model_options, reference_values)
        return results, residual
    else:
        options = ", ".join(f"'{k}'" for k in diffuser_function.keys())
        raise ValueError(
            f"Invalid diffuser model: '{vaneless_diffuser_model}'. Available options: {options}"
        )

def evaluate_vaneless_diffuser_algebraic(independents, input, boundary_conditions, geometry, fluid, model_options, reference_values):

    """
    Evaluate vaneless diffuser using algerbraic equations
    - Use conservation of angular momentum, accouting for friction by introducing a loss term
    - Pressure losses calculated according to correlation for loss coefficient
    """

    # Load reference_values
    v_max = reference_values["v_max"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Load independent variables
    v_out = independents["v_out_vaneless_diffuser"]*v_max
    s_out = independents["s_out_vaneless_diffuser"]*s_range + s_min 
    alpha_out = independents["alpha_out_vaneless_diffuser"]*angle_range + angle_min

    # Load input variables
    h0_in = input["h0"]
    s_in = input["s"]
    v_in = input["v"]
    v_t_in = input["v_t"]
    alpha_in = input["alpha"]

    # Load diffuser geometry
    b_in = geometry["width_in"]
    r_in = geometry["radius_in"]
    r_out = geometry["radius_out"]
    A_out = geometry["area_out"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load model factors
    Cf = model_options["factors"]["skin_friction"]
    loss_model = model_options["loss_model"]["vaneless_diffuser"]

    # Calculate tangential and meridional velocity
    v_m_out = v_out*math.cosd(alpha_out)
    v_t_out = v_out*math.sind(alpha_out)

    velocity_triangle_out = {"v_t" : v_t_out,
                   "v_m" : v_m_out,
                   "v" : v_out,
                   "alpha" : alpha_out,
                   }
    
    # Calculate thermophysical properties
    h0_out = h0_in
    h_out = h0_out - 0.5*v_out**2
    static_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
    stagnation_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h0_out, s_out)
    isentropic_stagnation_properties_out = fluid.get_props(cp.PSmass_INPUTS, stagnation_properties_out["p"], s_in)
    isentropic_static_properties_out = fluid.get_props(cp.PSmass_INPUTS, static_properties_out["p"], s_in)
    isentropic_properties_out = {**isentropic_static_properties_out,**utils.add_string_to_keys(isentropic_stagnation_properties_out, "0")}

    # Calculate mass flow rate and mach 
    m_out = static_properties_out["d"]*v_m_out*A_out
    Ma_out = v_out/static_properties_out["a"]

    # Collect exit plane results
    exit_plane = {**velocity_triangle_out,
                   **static_properties_out,
                   **utils.add_string_to_keys(stagnation_properties_out, "0"),
                   "Ma" : Ma_out
                   }

    # Evaluate loss coefficient
    loss_input = {
        "inlet_plane" : input,
        "exit_plane" : exit_plane,
        "geometry" : geometry,
        "isentropic" : isentropic_properties_out,
        "factors" : model_options["factors"],
        "boundary_conditions" : boundary_conditions,
    }
    loss_dict = lm.evaluate_loss_model(loss_model, loss_input, "vaneless_diffuser")
    exit_plane = {**exit_plane, **loss_dict}

    # Evaluate change of angular momentum
    phi_is = r_out*v_t_out/(r_in*v_t_in)
    phi_AM = np.exp(-Cf*(r_out-r_in)/(b_in*math.cosd(alpha_in)))

    # Evaluate residuals
    resiudals = {"mass_flow_out" : (m_out-mass_flow_rate)/mass_flow_rate,
                 "losses" : loss_dict["loss_error"],
                 "angular_momentum" : phi_is - phi_AM,
    }

    # Store diffuser results
    results = {"inlet_plane" : input,
               "exit_plane": exit_plane}
    
    return results, resiudals

def evaluate_vaneless_diffuser_1D(independents, input, boundary_conditions, geometry, fluid, model_options, reference_values):
    
    """
    Evaluate vaneless diffuser using 1D transport equations
    """

    # Rename inlet state properties
    v_m_in = input["v_m"]
    v_t_in = input["v_t"]
    d_in = input["d"]
    p_in = input["p"]

    # Rename geometry
    div = geometry["wall_divergence"]
    r_in = geometry["radius_in"]
    r_out = geometry["radius_out"]
    b_in = geometry["width_in"]
    phi = 0 # Diffuser channel is radial

    # Load model options
    Cf = model_options["factors"]["skin_friction"]
    q_w = model_options["factors"]["wall_heat_flux"]

    # Define integration interval
    m_out = r_out - r_in
    
    def odefun(t, y):

        # Rename from ODE terminology to physical variables
        m = t     
        v_m, v_t, d, p = y

        # Calculate velocity
        v = np.sqrt(v_t**2 + v_m**2)
        alpha = math.arctand(v_t/v_m)

        velocity_triangle = {"v_t" : v_t,
                             "v_m" : v_m,
                             "v" : v,
                             "alpha" : alpha}

        # Calculate local radius
        r = r_fun(r_in, phi, m)
        b = b_fun(b_in, div, m)

        # Calculate derivative of area change
        delta = 1e-4
        diff_br = (b_fun(b_in,div,m+delta)*r_fun(r_in,phi,m+delta) - b*r)/delta

        # Calculate derivative of internal energy
        delta = 1e-4
        e1 = fluid.get_props(cp.DmassP_INPUTS, d, p-delta)
        e1 = e1["u"]
        e2 = fluid.get_props(cp.DmassP_INPUTS, d, p+delta)
        e2 = e2["u"]
        dedp_d = (e2 - e1)/(2*delta)

        # Calculate thermodynamic state
        static_properties = fluid.get_props(cp.DmassP_INPUTS, d, p)
        a = static_properties["a"]
        h = static_properties["h"]
        s = static_properties["s"]
        h0 = h + 0.5*v**2
        stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0, s)

        # Stress at the wall
        tau_w = Cf*d*v**2/2

        # Compute coefficient matrix
        M = np.asarray(
            [
                [d, 0.0, v_m, 0],
                [d*v_m, 0.0, 0.0, 1.0],
                [0.0, d*v_m, 0.0, 0.0],
                [0.0, 0.0, -d*v_m*a**2, d*v_m]
            ]
        )

        # Compute source term
        S = np.asarray(
            [
                -d*v_m/(b*r)*diff_br,
                d*v_t**2/r*math.sind(phi) - 2*tau_w/b*math.cosd(alpha),
                -d*v_t*v_m/r*math.sind(phi)-2*tau_w/b*math.sind(alpha),
                2*(tau_w*v + q_w)/(b*dedp_d)
            ]
        ) 

        dy = scipy.linalg.solve(M, S)

        out = {**velocity_triangle,
               **static_properties,
               **utils.add_string_to_keys(stagnation_properties, "0")}

        return dy, out
    
    solution = scipy.integrate.solve_ivp(
        lambda t,y: odefun(t,y)[0],
        [0, m_out],
        [v_m_in, v_t_in, d_in, p_in],
        method = "RK45",
        rtol = 1e-6,
        atol = 1e-6,
    )

    dy, exit_plane = odefun(solution.t[-1], solution.y[:, -1])
    results = {
        "inlet_plane": input,
        "exit_plane" : exit_plane}

    return results, {}

def r_fun(r_in, phi, m):

    "Calculate the radius from the meridonial coordinate"

    return r_in + math.sind(phi)*m

def b_fun(b_in, div, m):

    "Calculate the channel width from the meridonial coordinate"

    return b_in + 2*math.tand(div)*m

def evaluate_volute(independents, input, boundary_conditions, geometry, fluid, model_options, reference_values):

    """
    Evaluate the volute. This include the scroll and concial diffuser.

    Assumptions:
    - Single velocity component at volute exit
    - Incompressible flow in diffuser cone

    """

    # Load reference_values
    v_max = reference_values["v_max"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]

    # Load independent variables
    v_out = independents["v_out_volute"]*v_max
    s_out = independents["s_out_volute"]*s_range + s_min

    # Load input variables
    h0_in = input["h0"]
    s_in = input["s"]

    # Load diffuser geometry
    A_out = geometry["area_out"]
    A_scroll = geometry["area_scroll"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load model options
    loss_model = model_options["loss_model"]["volute"]

    # Define velocity triangle
    velocity_triangle_out = {"v" : v_out}

    # Calculate thermophysical properties
    h0_out = h0_in
    h_out = h0_out - 0.5*v_out**2
    static_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
    stagnation_properties_out = fluid.get_props(cp.HmassSmass_INPUTS, h0_out, s_out)
    isentropic_stagnation_properties_out = fluid.get_props(cp.PSmass_INPUTS, stagnation_properties_out["p"], s_in)
    isentropic_static_properties_out = fluid.get_props(cp.PSmass_INPUTS, static_properties_out["p"], s_in)
    isentropic_properties_out = {**isentropic_static_properties_out,**utils.add_string_to_keys(isentropic_stagnation_properties_out, "0")}

    # Calculate mass flow rate and mach 
    m_out = static_properties_out["d"]*v_out*A_out
    Ma_out = v_out/static_properties_out["a"]

    # Evaluate scroll velocity (Assume incompressible flow in diffuser cone)
    v_scroll = v_out*A_out/A_scroll

    # Collect exit plane results
    exit_plane = {**velocity_triangle_out,
                   **static_properties_out,
                   **utils.add_string_to_keys(stagnation_properties_out, "0"),
                   "Ma" : Ma_out,
                   }

    # Evaluate losses 
    loss_input = {
        "inlet_plane" : input,
        "exit_plane" : exit_plane,
        "geometry" : geometry,
        "isentropic" : isentropic_properties_out,
        "factors" : model_options["factors"],
        "boundary_conditions" : boundary_conditions, 
    }
    loss_dict = lm.evaluate_loss_model(loss_model, loss_input, "volute")
    exit_plane = {**exit_plane, **loss_dict}

    # Evaluate residuals
    resiudals = {"mass_flow_out" : (m_out-mass_flow_rate)/mass_flow_rate,
                 "losses" : loss_dict["loss_error"],
    }

    # Store diffuser results
    results = {"inlet_plane" : input,
               "exit_plane": exit_plane}
    
    return results, resiudals
    

def compute_overall_performance(results, fluid, geometry):
    """
    Calculate the overall performance metrics of the centrifugal compressor.

    This function extracts necessary values from the `results` dictionary, performs calculations to determine
    overall performance metrics, and returns these in a dictionary.

    Parameters
    ----------
    results : dict
        A dictionary containing all necessary information, such as geometry and flow characteristics.

    Returns
    -------
    dict
        An dictionary containing the calculated performance metrics.
    """

    # Load boundary conditions
    p0_in = results["boundary_conditions"]["p0_in"]
    T0_in = results["boundary_conditions"]["T0_in"]
    omega = results["boundary_conditions"]["omega"]
    mass_flow_rate = results["boundary_conditions"]["mass_flow_rate"]

    # Get results for first and final plane
    first_plane = results["planes"].loc[0].to_dict()
    final_plane = results["planes"].loc[len(results["planes"])-1].to_dict()

    # Get impeller results
    impeller = results["impeller"]

    # Calculate enthalpy for isentropic compression
    stagnation_properties = fluid.get_props(cp.PSmass_INPUTS, final_plane["p0"], first_plane["s"])
    h0_out_is = stagnation_properties["h"]
    static_properties = fluid.get_props(cp.PSmass_INPUTS, final_plane["p"], first_plane["s"])
    h_out_is = static_properties["h"]
 
    # Calculate overall performance characteristics
    PR_tt = final_plane["p0"]/p0_in
    PR_ts = final_plane["p"]/p0_in
    power = mass_flow_rate*(final_plane["h0"]-first_plane["h0"])
    torque = mass_flow_rate*(abs(geometry["impeller"]["radius_out"]*impeller["exit_plane"]["v_t"]) - abs(geometry["impeller"]["radius_mean_in"]*impeller["inlet_plane"]["v_t"]))
    efficiency_ts = mass_flow_rate*(h_out_is - first_plane["h0"])/(omega*torque)*100
    efficiency_tt = mass_flow_rate*(h0_out_is - first_plane["h0"])/(omega*torque)*100

    # Calculate nondimensional parameters
    gamma = first_plane["gamma"]
    R = first_plane["cp"] - first_plane["cv"]
    m_non_dim = mass_flow_rate*np.sqrt(T0_in)/p0_in*np.sqrt(R/gamma)/(2*geometry["impeller"]["radius_out"])**2
    N_non_dim = omega*60/2/np.pi/np.sqrt(T0_in)/np.sqrt(gamma*R)*2*geometry["impeller"]["radius_out"]

    # Store all variables in dictionary
    overall = {
        "PR_tt": PR_tt,
        "PR_ts": PR_ts,
        "efficiency_tt": efficiency_tt,
        "efficiency_ts": efficiency_ts,
        "power": power,
        "torque": torque,
        "m_non_dim" : m_non_dim,
        "N_non_dim" : N_non_dim,
    }

    return overall


