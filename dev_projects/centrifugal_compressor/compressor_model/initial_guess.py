
from scipy.stats import qmc
import numpy as np
import turboflow as tf
import CoolProp as cp
from scipy import optimize

initial_guess = {"v_in_impeller" : 0.08,
                 "w_out_impeller" : 0.2,
                 "beta_out_impeller" : (-30+90)/180,
                 "s_out_impeller" : 0.00,
                 "v_out_vaneless_diffuser" : 0.1,
                 "s_out_vaneless_diffuser" : 0.00,
                 "alpha_out_vaneless_diffuser" : (74+90)/180,
                 "v_out_vaned_diffuser" : 0.1,
                 "s_out_vaned_diffuser" : 0.00,

                 }

def latin_hypercube_sampling(bounds, n_samples):
    """
    Generates samples using Latin Hypercube Sampling.

    Parameters:
    bounds (list of tuples): A list of (min, max) bounds for each variable.
    n_samples (int): The number of samples to generate.

    Returns:
    np.ndarray: An array of shape (n_samples, n_variables) containing the samples.
    """
    n_variables = len(bounds)
    # Create a Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=n_variables, seed = 1)
    
    # Generate samples in the unit hypercube
    unit_hypercube_samples = sampler.random(n=n_samples)
    
    # Scale the samples to the provided bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(unit_hypercube_samples, lower_bounds, upper_bounds)

    return scaled_samples

def get_heuristic_guess(eta_tt, phi_out, boundary_conditions, geometry, fluid, model_options, v0):

    # Check vaneless diffuser model
    vaneless_model = model_options["vaneless_diffuser_model"]

    # initialize initial guess dictionary
    initial_guess = {}

    # Prepare first input
    input = {"eta_tt" : eta_tt,
             "phi_out" : phi_out}

    # Get initial guess
    for key in geometry.keys():
        if key == "impeller":
            guess, exit_state = get_impeller_guess(input, boundary_conditions, geometry[key], fluid, v0)
        elif key == "vaneless_diffuser" and vaneless_model == "algebraic":
            guess, exit_state = get_vaneless_diffuser_guess(input, boundary_conditions, geometry[key], fluid, model_options)
        elif key == "vaned_diffuser":
            guess, exit_state = get_vaned_diffuser_guess(input, boundary_conditions, geometry[key], fluid, model_options)
        elif key == "volute":
            guess, exit_state = get_volute_guess(input, boundary_conditions, geometry[key], fluid)

        # Store guess
        initial_guess = {**initial_guess, **guess}

        # Prepare calculation of next component
        input = exit_state

    return initial_guess

def get_vaneless_diffuser_guess(input, boundary_conditions, geometry, fluid, model_options):
    """
    Guess constant alpha
    Simple correlation for static enthalpy loss coefficient
    """

    # Load input
    alpha_in  = input["alpha"]
    v_t_in = input["v_t"]
    v_in = input["v"]
    h_in = input["h"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load geometry
    r_out = geometry["radius_out"]
    r_in = geometry["radius_in"]
    b_in = geometry["width_in"]
    A_out = geometry["area_out"]

    # Load model options
    Cf = model_options["factors"]["skin_friction"]

    # Calculate exit velocity
    alpha_out = alpha_in
    delta_M = np.exp(-Cf*(r_out-r_in)/(b_in*tf.math.cosd(alpha_in)))
    v_t_out = v_t_in*r_out/r_in*delta_M
    v_out = v_t_out/tf.math.sind(alpha_out)
    v_m_out = v_out*tf.math.cosd(alpha_out)
    velocity_triangle_out = {
        "v" : v_out,
        "v_t" : v_t_out,
        "v_m" : v_m_out,
        "alpha" : alpha_out
    }
    
    # Calculate exit entropy
    d_out = mass_flow_rate/(v_out*tf.math.cosd(alpha_out)*A_out)
    h0_out = h_in + 0.5*v_in**2
    h_out = h0_out - 0.5*v_out**2
    static_props_out = fluid.get_props(cp.DmassHmass_INPUTS, d_out, h_out)
    s_out = static_props_out["s"]

    initial_guess = {
                 "v_out_vaneless_diffuser" : v_out,
                 "s_out_vaneless_diffuser" : s_out,
                 "alpha_out_vaneless_diffuser" : alpha_out,
                 }
    
    exit_state = {
        **velocity_triangle_out,
        **static_props_out,
    }

    return initial_guess, exit_state

def get_vaned_diffuser_guess(input, boundary_conditions, geometry, fluid, model_options):

    """
    Similar as vaneless diffuser
    Assume zero deviation at exit 
    """
    
    # Load input
    alpha_in  = input["alpha"]
    v_t_in = input["v_t"]
    v_in = input["v"]
    h_in = input["h"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load geometry
    r_out = geometry["radius_out"]
    r_in = geometry["radius_in"]
    b_in = geometry["width_in"]
    A_out = geometry["area_out"]
    theta_out = geometry["trailing_edge_angle"]

    # Load model options
    Cf = model_options["factors"]["skin_friction"]

    # Calculate exit velocity
    alpha_out = theta_out
    delta_M = np.exp(-Cf*(r_out-r_in)/(b_in*tf.math.cosd(alpha_in)))
    v_t_out = v_t_in*r_out/r_in*delta_M
    v_out = v_t_out/tf.math.sind(alpha_out)
    v_m_out = v_out*tf.math.cosd(alpha_out)

    velocity_triangle_out = {
        "v" : v_out,
        "v_t" : v_t_out,
        "v_m" : v_m_out,
        "alpha" : alpha_out
    }
    
    # Calculate exit entropy
    d_out = mass_flow_rate/(v_out*tf.math.cosd(alpha_out)*A_out)
    h0_out = h_in + 0.5*v_in**2
    h_out = h0_out - 0.5*v_out**2
    static_props_out = fluid.get_props(cp.DmassHmass_INPUTS, d_out, h_out)
    s_out = static_props_out["s"]

    initial_guess = {
                 "v_out_vaned_diffuser" : v_out,
                 "s_out_vaned_diffuser" : s_out,
                 }
    
    exit_state = {
        **velocity_triangle_out,
        **static_props_out,
    }

    return initial_guess, exit_state

def get_volute_guess(input, boundary_conditions, geometry, fluid):
    """
    Guess constant density
    Guess single velocity component at the exit
     
    """
    
    # Load input
    d_in = input["d"]
    h_in = input["h"]
    v_in = input["v"]

    # Load boundary conditions
    mass_flow_rate = boundary_conditions["mass_flow_rate"]

    # Load geometry
    A_out = geometry["area_out"]

    # Calculate exit velocity
    d_out = d_in
    v_out = mass_flow_rate/(d_out*A_out)
    velocity_triangle_out = {
        "v" : v_out,
    }


    # Calculate exit entropy
    h0_out = h_in + 0.5*v_in**2
    h_out = h0_out - 0.5*v_out**2
    static_props_out = fluid.get_props(cp.DmassHmass_INPUTS, d_out, h_out)
    s_out = static_props_out["s"]

    initial_guess = {
                 "v_out_volute" : v_out,
                 "s_out_volute" : s_out,
                 }

    exit_plane = {
        **velocity_triangle_out,
        **static_props_out
    }

    return initial_guess, exit_plane

def get_impeller_guess(input, boundary_conditions, geometry, fluid, v0):

    # Load boundary conditions
    p0_in = boundary_conditions["p0_in"]
    T0_in = boundary_conditions["T0_in"]
    omega = boundary_conditions["omega"]
    mass_flow_rate = boundary_conditions["mass_flow_rate"]
    alpha_in = boundary_conditions["alpha_in"]

    # Load input
    eta_tt = input["eta_tt"]
    phi_out = input["phi_out"]
    
    # Load geometry
    z = geometry["number_of_blades"]
    theta_out = geometry["trailing_edge_angle"]
    r_out = geometry["radius_out"]
    r_in = geometry["radius_mean_in"]
    A_out = geometry["area_out"]
    A_in = geometry["area_in"]

    # Evaluate inlet thermpdynamic state
    stagnation_props_in = fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
    gamma = stagnation_props_in["cp"]/stagnation_props_in["cv"]
    h0_in = stagnation_props_in["h"]
    a0_in = stagnation_props_in["a"]
    s_in = stagnation_props_in["s"]

    # Approximate impeller exit total pressure
    u_out = r_out*omega
    Ma_out = u_out/a0_in
    slip_factor = 1-np.sqrt(tf.math.cosd(theta_out))/(z**(0.7)*(1+phi_out*tf.math.tand(theta_out)))
    p0_out = (1+(gamma-1)*eta_tt*slip_factor*(1+phi_out*tf.math.tand(theta_out))*Ma_out**2)**(gamma/(gamma-1))*p0_in

    # Get impeller exit total enthalpy and entropy
    isentropic_props_out = fluid.get_props(cp.PSmass_INPUTS, p0_out, s_in)
    h0_out_is = isentropic_props_out["h"]
    h0_out = (h0_out_is-h0_in)/eta_tt + h0_in
    stagnation_props_out = fluid.get_props(cp.HmassP_INPUTS, h0_out, p0_out)
    s_out = stagnation_props_out["s"]

    # Get inlet velocity
    def calculate_inlet_residual(v_scaled):

        # Calculate velocity triangle
        u_in = r_in*omega
        v_in = v_scaled*v0
        v_m_in = v_in*tf.math.cosd(alpha_in)
        v_t_in = v_in*tf.math.sind(alpha_in)
        w_m_in = v_m_in
        w_t_in = v_t_in - u_in
        w_in = np.sqrt(w_m_in**2 + w_t_in**2)   
        beta_in = tf.math.arctand(w_t_in/w_m_in)

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

        # Calculate thermodynamic state
        h_in = h0_in - 0.5*v_in**2
        h0_rel_in = h_in + 0.5*w_in**2
        static_props_in = fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
        relative_props_in = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel_in, s_in)

        # Calculate mass flow rate nad rothalpy
        m_in = static_props_in["d"]*v_m_in*A_in
        rothalpy_in = h0_rel_in - 0.5*u_in**2

        # Evaluate mass flow rate residual
        res = m_in - mass_flow_rate 

        check_in = {"mass_flow_rate" : m_in,
                    "rothalpy" : rothalpy_in,
                    "s_in" : s_in}

        return res, velocity_triangle_in, check_in
    
    def calculate_inlet_velocity():

        sol = optimize.root_scalar(lambda x: calculate_inlet_residual(x)[0], method = "secant", x0 = 0.2)
        delta_h, velocity_triangle_in, check_in  = calculate_inlet_residual(sol.root)
        return velocity_triangle_in, check_in

    velocity_triangle_in, check_in = calculate_inlet_velocity()

    # Calculate exit state
    v_m_out = phi_out*u_out
    v_t_out = slip_factor*(1+phi_out*tf.math.tand(theta_out))*u_out
    v_out = np.sqrt(v_t_out**2 + v_m_out**2)
    alpha_out = tf.math.arctand(v_t_out/v_m_out)
    w_m_out = v_m_out
    w_t_out = v_t_out - u_out
    w_out = np.sqrt(w_t_out**2 + w_m_out**2)
    beta_out = tf.math.arctand(w_t_out/w_m_out)

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
    
    # Get thermophysical properties
    h_out = h0_out -0.5*v_out**2
    h0_rel_out = h_out + 0.5*w_out**2
    static_props_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)

    # Calculate mass flow rate and rothalpy
    m_out = static_props_out["d"]*v_m_out*A_out
    rothalpy_out = h0_rel_out - 0.5*u_out**2

    exit_plane = {
        **velocity_triangle_out,
        **static_props_out,
    }

    # Store initial guess
    initial_guess = {"v_in_impeller" : velocity_triangle_in["v"],
                 "w_out_impeller" : w_out,
                 "beta_out_impeller" : beta_out,
                 "s_out_impeller" : s_out,
                 }
    
    return initial_guess, exit_plane

def get_initial_guess(initial_guess, problem, boundary_conditions, geometry, fluid, model_options):

    v0 = problem.reference_values["v_max"]

    valid_keys_1 = ["efficiency_tt", "phi_out"]
    valid_keys_2 = ["efficiency_tt", "phi_out", "n_samples"]
    check = []
    check.append(set(valid_keys_1) == set(list(initial_guess.keys())))
    check.append(set(valid_keys_2) == set(list(initial_guess.keys())))

    if check[0]:
        # Initial guess determined from heurustic guess, with given parameters
        if isinstance(initial_guess["efficiency_tt"], (list, np.ndarray)):
            # Several initial guesses
            initial_guesses = []
            for i in range(len(initial_guess["efficiency_tt"])):
                heuristic_guess = get_heuristic_guess(initial_guess["efficiency_tt"][i], initial_guess["phi_out"][i], boundary_conditions, geometry, fluid, model_options, v0)
                initial_guesses.append(heuristic_guess)
        else:
            # Single initial guess
            heuristic_guess = get_heuristic_guess(initial_guess["efficiency_tt"], initial_guess["phi_out"], boundary_conditions, geometry, fluid, model_options, v0)
            initial_guesses = [heuristic_guess]

    elif check[1]:
        # Generate initial guess using latin hypercube sampling
        bounds = [initial_guess["efficiency_tt"], initial_guess["phi_out"]]

        n_samples = initial_guess["n_samples"]
        heuristic_inputs = latin_hypercube_sampling(bounds, n_samples)
        norm_residuals = np.array([])
        failures = 0
        for heuristic_input in heuristic_inputs:
            try:
                heuristic_guess = get_heuristic_guess(heuristic_input[0], heuristic_input[1], boundary_conditions, geometry, fluid, model_options, v0)
                x = problem.scale_values(heuristic_guess) # TODO: Add scaling
                problem.keys = x.keys() 
                x0 = np.array(list(x.values()))
                residual = problem.residual(x0)
                norm_residuals = np.append(norm_residuals, np.linalg.norm(residual))
            except:
                failures += 1 
                norm_residuals = np.append(norm_residuals, np.nan)

        print(f"Generating heuristic inital guesses from latin hypercube sampling")
        print(f"Number of failures: {failures} out of {n_samples} samples")
        print(f"Least norm of residuals: {np.nanmin(norm_residuals)}")
        heuristic_input = heuristic_inputs[np.nanargmin(norm_residuals)]
        initial_guess = dict(zip(valid_keys_1, heuristic_input))
        initial_guess = get_heuristic_guess(heuristic_input[0], heuristic_input[1], boundary_conditions, geometry, fluid, model_options, v0)
        initial_guesses = [initial_guess]
    else:
        # Simply return the initial guess given
        initial_guesses = [initial_guess]

    return initial_guesses


if __name__ == '__main__':

    operating_point = {
        "fluid_name": "air",
        "T0_in": 273.15 + 15,
        "p0_in": 101325,
        "mass_flow_rate" : 0.45,
        "omega": 52000*2*np.pi/60,
        "alpha_in": 0,
    }

    impeller = {
        "radius_hub_in": 40.9/2*1e-3,
        "radius_tip_in": 85.6/2*1e-3,
        "radius_out" : 143/2*1e-3,
        "width_out" : 6.29e-3,
        "leading_edge_angle" : (43.6+60)/2,
        "trailing_edge_angle" : -24.5,
        "number_of_blades" : 12, 
        "impeller_length" : 45.3e-3,
        "tip_clearance" : 0.2e-3,
        "width_diffuser" : 6.29e-3,
    }
    impeller["radius_mean_in"] = (40.9/2*1e-3+ 85.6/2*1e-3)/2
    impeller["area_out"] = 2*np.pi*impeller["radius_out"]*impeller["width_out"] # Area exit
    impeller["area_in"] = np.pi * (impeller["radius_tip_in"]**2 - impeller["radius_hub_in"]**2) # Area inlet

    vaneless_diffuser = {
        "wall_divergence" : 0,
        "radius_out" : 150.22/2*1e-3,
        "radius_in" : 143/2*1e-3,
        "width_in" : 6.29e-3,
    }
    b_out = vaneless_diffuser["width_in"] + 2*tf.math.tand(vaneless_diffuser["wall_divergence"])*(vaneless_diffuser["radius_out"] - vaneless_diffuser["radius_in"])
    vaneless_diffuser["area_out"] = 2*np.pi*vaneless_diffuser["radius_out"]*b_out

    vaned_diffuser = {
        "radius_out" : 183.76/2*1e-3,
        "radius_in" : 150.22/2*1e-3,
        "width_in" : 6.29e-3,
        "width_out" : 6.29e-3,
        "leading_edge_angle" : 77.77, # From NASA compressors
        "trailing_edge_angle" : 34.0, # From NASA compressors
        "number_of_vanes" : 21,
    }
    vaned_diffuser["area_out"] = 2*np.pi*vaned_diffuser["radius_out"]*vaned_diffuser["width_out"]

    geometry = {
        "impeller" : impeller,
        "vaneless_diffuser" : vaneless_diffuser,
        "vaned_diffuser" : vaned_diffuser,
        }

    loss_model = {
        "impeller" :{
            "model" : "custom",
            "loss_coefficient" : "static_enthalpy_loss"
        },
        "vaneless_diffuser" :{
            "model" : "custom",
            "loss_coefficient" : "static_enthalpy_loss"
        },
        "vaned_diffuser" :{
            "model" : "isentropic",
            "loss_coefficient" : "static_enthalpy_loss"
        },
    }

    # loss_model = {
    #     "loss_model" : {
    #         "model" : "oh",
    #         "loss_coefficient" : "static_enthalpy_loss",
    #     }
    # }



    simulation_options = {
        "slip_model" : "wiesner", 
        "loss_model" : loss_model,
        "vaneless_diffuser_model" : "algebraic", 
        "factors" : {"skin_friction" : 0.02,
                    "wall_heat_flux" : 0.00,
                    "wake_width" : 0.366,
                    } 
    }
    
    problem = tf.centrifugal_compressor.CentrifugalCompressorProblem(geometry, simulation_options)
    problem.update_boundary_conditions(operating_point)

    # initial_guess = get_heuristic_guess(0.9, 0.5, operating_point, geometry, problem.fluid, simulation_options, problem)
    initial_guess = {"efficiency_tt" : (0.6, 0.98), "phi_out" : (0.1, 0.5), "n_samples" : 100}
    # initial_guess = {"efficiency_tt" : 0.98, "phi_out" : 0.2}
    initial_guesses = get_initial_guess(initial_guess, problem, operating_point, geometry, problem.fluid, simulation_options)
    initial_guess = initial_guesses[0]
    # print(initial_guess)
    initial_guess_scaled = problem.scale_values(initial_guess)
    # print(initial_guess_scaled)
    problem.keys = list(initial_guess.keys())
    x0 = np.array(list(initial_guess_scaled.values()))
    # residual = problem.residual(x0)
    # print(residual)

    # print(problem.results["residuals"])


    solver_options = {
        "method": "lm",  
        "tolerance": 1e-6,  
        "max_iterations": 100,  
        "derivative_method": "2-point",  
        "derivative_abs_step": 1e-6,  
        "plot_convergence": False,
    }
    solver = tf.psv.NonlinearSystemSolver(problem, **solver_options)
    solver.solve(x0)









