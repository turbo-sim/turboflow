import numpy as np
import pandas as pd
from scipy.linalg import solve
import CoolProp as cp
from scipy.optimize._numdiff import approx_derivative

from .. import math
from .. import utilities as utils
from . import loss_model as lm
from . import deviation_model as dm
from . import choking_model as cm

# List of valid options
BLOCKAGE_MODELS = ["flat_plate_turbulent"]

# Keys of the information that should be stored in results
KEYS_KINEMATIC = [
    "u",
    "v",
    "v_m",
    "v_t",
    "alpha",
    "w",
    "w_m",
    "w_t",
    "beta",
]

KEYS_PROPS_STATIC = ["p", "T", "h", "s", "d", "Z", "a", "mu", "k", "cp", "cv", "gamma"]
KEYS_PROPS_STAG_ABS = [f"{key}0" for key in KEYS_PROPS_STATIC]
KEYS_PROPS_STAG_REL = [f"{key}0_rel" for key in KEYS_PROPS_STATIC]
KEYS_LOSSES = lm.KEYS_LOSSES

KEYS_PLANE = (
    KEYS_KINEMATIC
    + KEYS_PROPS_STATIC
    + KEYS_PROPS_STAG_ABS
    + KEYS_PROPS_STAG_REL
    + KEYS_LOSSES
    + [
        "Ma",
        "Ma_rel",
        "Re",
        "mass_flow",
        "rothalpy",
        "blockage",
    ]
)
"""List of keys for the plane performance metrics of the turbine. 
This list is used to ensure the structure of the 'plane' dictionary in various functions."""

KEYS_CASCADE = [
    "loss_total",
    "loss_profile",
    "loss_clearance",
    "loss_secondary",
    "loss_trailing",
    "loss_incidence",
    "dh_s",
    "Ma_crit_throat",
    "mass_flow_throat",
    "w_throat",
    "d_throat",
    "Ma_throat",
    "beta_throat",
    "mass_flow_crit_throat",
    "incidence",
    "beta_subsonic" ,
    "beta_supersonic",
]

"""List of keys for the cascade performance metrics of the turbine. 
This list is used to ensure the structure of the 'cascade' dictionary in various functions."""

KEYS_OVERALL = [
    "PR_tt",
    "PR_ts",
    "mass_flow_rate",
    "efficiency_tt",
    "efficiency_ts",
    "efficiency_ts_drop_kinetic",
    "efficiency_ts_drop_losses",
    "power",
    "torque",
    "angular_speed",
    "exit_flow_angle",
    "exit_velocity",
    "spouting_velocity",
    "last_blade_velocity",
    "blade_jet_ratio",
    "h0_in",
    "h0_out",
    "h_out_s",
]
"""List of keys for the overall performance metrics of the turbine. 
This list is used to ensure the structure of the 'overall' dictionary in various functions."""

KEYS_STAGE = [
    "reaction",
]
"""List of keys for the stage performance metrics of the turbine. 
This list is used to ensure the structure of the 'stage' dictionary in various functions."""

def evaluate_axial_turbine(
    variables,
    boundary_conditions,
    geometry,
    fluid,
    model_options,
    reference_values,
):
    """
    Compute the performance of an axial turbine by evaluating a series of cascades.

    This function evaluates each cascade in the series.
    It begins by loading essential inputs such as geometry, boundary conditions, and reference values, which are integral to the assessment process.

    The evaluation proceeds cascade by cascade, employing two key functions in a structured sequence:

    1. `evaluate_cascade`: Compute flow at the inlet station, throat, and exit planes of each cascade. In addition, this function evaluates
    the turbine performance at the critical point, which is essential to model the correct behavior under choking conditions.
    2. `evaluate_cascade_interspace`: Calculate the flow conditions after the interspace between consecutive cascades to obtain conditions at the inlet of the next cascade.

    As the function iterates through each cascade, it accumulates data in a structured format. The results are organized into a dictionary that includes:

    - 'cascade': Contains data specific to each cascade in the series, including loss coefficients and critical conditions.
    - 'plane': Contains data specific to each flow station, including thermodynamic properties and velocity triangles.
    - 'stage': Contains data specific to each turbine stage, including degree of reaction.
    - 'overall': Summarizes the overall performance of the turbine, including total mass flow rate, efficiency, power output, and other key performance indicators.
    - 'residuals': Summarizes the mismatch in the nonlinear system of equations used to model the turbine.

    .. note::

        The output of this function can only be regarded as a physically meaningful solution when the equations are fully closed (i.e., all residuals are close to zero).
        Therefore, this function is intended to be used within a root-finding or optimization algorithm that systematically adjusts the input variables to drive the residuals to zero and close the system of equations.

    Parameters
    ----------
    variables : dict
        Dictionary containing variable for the cascades.
    boundary_conditions : dict
        Dictionary containing boundary conditions for the series of cascades.
    geometry : dict
        Dictionary containing geometric parameters of the series of cascades.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    model_options : dict
        Dictionary containing various model options.
    reference_values : dict
        Dictionary containing reference values for normalization.

    Returns
    -------
    results : dict
        Dictionary containing the evaluated results, including planes, cascades, stage, overall,
        and geometry information.


    """

    # Load variables
    number_of_cascades = geometry["number_of_cascades"]
    h0_in = boundary_conditions["h0_in"]
    s_in = boundary_conditions["s_in"]
    alpha_in = boundary_conditions["alpha_in"]
    angular_speed = boundary_conditions["omega"]

    # Load reference_values
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Initialize results structure
    results = {
        "plane": pd.DataFrame(columns=KEYS_PLANE),
        "cascade": pd.DataFrame(columns=KEYS_CASCADE),
        "stage": {},
        "overall": {},
        "geometry": geometry,
        "reference_values": reference_values,
        "boundary_conditions": boundary_conditions,
    }

    # initialize residual arrays
    residuals = {}

    # Rename turbine inlet velocity
    v_in = variables["v_in"] * v0
    
    for i in range(number_of_cascades):
        # Update angular speed
        angular_speed_cascade = angular_speed * (i % 2)

        # update geometry for current cascade #FIXME for new geometry model
        geometry_cascade = {
            key: values[i]
            for key, values in geometry.items()
            if key not in ["number_of_cascades", "number_of_stages"]
        }

        # Rename variables
        cascade = "_" + str(i + 1)
        w_out = variables["w_out" + cascade] * v0
        s_out = variables["s_out" + cascade] * s_range + s_min # TODO: change scaling
        beta_out = variables["beta_out" + cascade] * angle_range + angle_min
        
        # Evaluate current cascade
        cascade_inlet_input = {
            "h0": h0_in,
            "s": s_in,
            "alpha": alpha_in,
            "v": v_in,
        }
        cascade_exit_input = {
            "w": w_out,
            "beta": beta_out,
            "s": s_out,
        }
        
        choking_input = {key.replace(cascade, "") : val for key, val in variables.items() if ("*" and cascade in key) or key == "v*_in"}

        cascade_residuals = evaluate_cascade(
            cascade_inlet_input,
            cascade_exit_input,
            choking_input,
            fluid,
            geometry_cascade,
            angular_speed_cascade,
            results,
            model_options,
            reference_values,
        )

        # Add cascade residuals to residual arrays
        cascade_residuals = utils.add_string_to_keys(cascade_residuals, f"_{i+1}")
        residuals.update(cascade_residuals)

        # Calculate input of next cascade (Assume no change in density)
        if i != number_of_cascades - 1:
            (
                h0_in,
                s_in,
                alpha_in,
                v_in,
            ) = evaluate_cascade_interspace(
                results["plane"]["h0"].values[-1],
                results["plane"]["v_m"].values[-1],
                results["plane"]["v_t"].values[-1],
                results["plane"]["d"].values[-1],
                geometry_cascade["radius_mean_out"],
                geometry_cascade["A_out"],
                results["plane"]["blockage"].values[-1],
                geometry["radius_mean_in"][i + 1],
                geometry["A_in"][i + 1],
                fluid,
            )

    # Add exit pressure error to residuals
    p_calc = results["plane"]["p"].values[-1]
    p_error = (p_calc - boundary_conditions["p_out"]) / boundary_conditions["p0_in"]
    residuals["p_out"] = p_error

    # Additional calculations
    loss_fractions = compute_efficiency_breakdown(results)
    results["cascade"] = pd.concat([results["cascade"], loss_fractions], axis=1)

    # Compute stage performance
    # results["stage"] = pd.DataFrame([compute_stage_performance(results)])

    # Compute overall perfrormance
    results["overall"] = pd.DataFrame([compute_overall_performance(results)])

    # Store all residuals for export
    results["residuals"] = residuals

    # Store the input variables
    results["independent_variables"] = variables

    # Retain only variables defined per cascade
    geom_cascades = {
        key: value
        for key, value in geometry.items()
        if len(utils.ensure_iterable(value)) == number_of_cascades
    }
    results["geometry"] = pd.DataFrame([geom_cascades])
    
    
    return results

def evaluate_cascade(
    cascade_inlet_input,
    cascade_exit_input,
    choking_input,
    fluid,
    geometry,
    angular_speed,
    results,
    model_options,
    reference_values,
):
    
    """
    Evaluate the performance of a cascade configuration.
    
    This function evaluates the cascade performance by considering inlet, throat, and exit planes.
    It also evaluates the cascade at point of choking. The results are stored in a dictionary. The function 
    The function returns a dictionary of residuals that includes the mass balance error at both the throat and exit, loss coefficient errors, 
    and the residuals related to the critical state and choking condition.
    
    This function relies on auxiliary functions like evaluate_cascade_inlet, evaluate_cascade_exit,
    evaluate_cascade_critical, compute_residual_flow_angle, and compute_residual_mach_throat.
    
    Parameters
    ----------
    cascade_inlet_input : dict
        Input conditions at the cascade inlet.
    cascade_throat_input : dict
        Input conditions at the cascade throat.
    cascade_exit_input : dict
        Input conditions at the cascade exit.
    critical_cascade_input : dict
        Input conditions for critical state evaluation.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Dictionary containing geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    results : dict
        Dictionary to store the evaluation results.
    model_options : dict
        Dictionary containing various model options.
    reference_values : dict
        Dictionary containing reference values for normalization.
    
    Returns
    -------
    residuals : dict
        A dictionary containing the residuals of the evaluation, which are key indicators of the model's accuracy and physical realism.

    """

    # Define model options
    # TODO Add warnings that some settings have been assumed. This is a dangerous silent behavior as it is now
    # TODO It would make sense to define default values for these options at a higher level, for instance after processing the configuration file
    # TODO an advantage of the approach above is that it is possible to print a warning that will not be shown for each iteration of the solvers
    loss_model = model_options["loss_model"]
    deviation_model = model_options["deviation_model"]

    # Load reference values
    mass_flow_reference = reference_values["mass_flow_ref"]

    # Evaluate inlet plane
    inlet_plane = evaluate_cascade_inlet(
        cascade_inlet_input, fluid, geometry, angular_speed
    )

    # Evaluate exit plane
    cascade_exit_input["rothalpy"] = inlet_plane["rothalpy"]
    exit_plane, loss_dict = evaluate_cascade_exit(
        cascade_exit_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        model_options["blockage_model"],
        loss_model,
    )

    # Evaluate isentropic enthalpy change
    props_is = fluid.get_props(cp.PSmass_INPUTS, exit_plane["p"], inlet_plane["s"])
    dh_is = exit_plane["h"] - props_is["h"]

    # Evaluate critical state
    residuals_critical, critical_state = cm.evaluate_choking(
        choking_input,
        inlet_plane, 
        exit_plane,
        fluid,
        geometry,
        angular_speed,
        model_options,
        reference_values,
    )

    # Create dictionary with equation residuals
    mass_error_exit = inlet_plane["mass_flow"] - exit_plane["mass_flow"]
    residuals = {
        "loss_error_exit": exit_plane["loss_error"],
        "mass_error_exit": mass_error_exit / mass_flow_reference,
        **residuals_critical,
    }

    # Return plane data in dataframe
    # TODO: add critical state as another plane?
    planes = [inlet_plane, exit_plane]
    for plane in planes:
        results["plane"].loc[len(results["plane"])] = plane

    # Return cascade data in dataframe
    # TODO: how much critical point data do we want to retrieve?
    cascade_data = {
        **loss_dict,
        "dh_s": dh_is,
        "Ma_crit_throat": critical_state["throat_plane"]["Ma_rel"],
        # "Ma_throat": throat_plane["Ma_rel"],
        # "beta_throat" : throat_plane["beta"],
        "mass_flow_crit_throat": critical_state["throat_plane"]["mass_flow"],
        # "mass_flow_throat": throat_plane["mass_flow"],
        # "w_throat" : throat_plane["w"],
        # "Y_crit_out" : critical_state["exit_plane"]["loss_total"],
        # "d_throat": throat_plane["d"],
        # "d_crit_out": critical_state["exit_plane"]["d"],
        # "mass_flux_out" : exit_plane["w"]*exit_plane["d"],
        "incidence": inlet_plane["beta"] - geometry["metal_angle_le"],
        # "w_crit" : critical_state["throat_plane"]["w"],
        # "beta_subsonic" : beta_sub,
        # "beta_supersonic" : beta_sup,
    }
    results["cascade"].loc[len(results["cascade"])] = cascade_data

    return residuals

def evaluate_cascade_inlet(cascade_inlet_input, fluid, geometry, angular_speed):
    """
    Evaluate the inlet plane parameters of a cascade including velocity triangles,
    thermodynamic properties, and flow characteristics.

    This function calculates performance data at the inlet of a cascade based on the cascade geometry,
    fluid, and flow conditions. It computes velocity triangles, static and stagnation properties,
    Reynolds and Mach numbers, and the mass flow rate at the inlet.

    Parameters
    ----------
    cascade_inlet_input : dict
        Input parameters specific to the cascade inlet, including stagnation enthalpy ('h0'),
        entropy ('s'), absolute velocity ('v'), and flow angle ('alpha').
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade such as radius at the mean inlet ('radius_mean_in'),
        chord length, and inlet area ('A_in').
    angular_speed : float
        Angular speed of the cascade.

    Returns
    -------
    dict
        A dictionary of calculated parameters at the cascade inlet.
    """
    # Load cascade inlet input
    h0 = cascade_inlet_input["h0"]
    s = cascade_inlet_input["s"]
    v = cascade_inlet_input["v"]
    alpha = cascade_inlet_input["alpha"]

    # Load geometry
    radius = geometry["radius_mean_in"]
    chord = geometry["chord"]
    area = geometry["A_in"]

    # Calculate velocity triangles
    blade_speed = radius * angular_speed
    velocity_triangle = evaluate_velocity_triangle_in(blade_speed, v, alpha)
    w = velocity_triangle["w"]
    w_m = velocity_triangle["w_m"]

    # Calculate static properties
    h = h0 - 0.5 * v**2
    static_properties = fluid.get_props(cp.HmassSmass_INPUTS, h, s)
    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

    # Calculate stagnation properties
    stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    # Calculate relative stagnation properties
    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(
        relative_stagnation_properties, "0_rel"
    )

    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    m = rho * w_m * area
    rothalpy = h0_rel - 0.5 * blade_speed**2

    # Store results in dictionary
    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
        **{key: np.nan for key in KEYS_LOSSES},
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": m,
        "rothalpy": rothalpy,
        "blockage": np.nan,
    }

    return plane

def evaluate_cascade_exit(
    cascade_exit_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    """
    Evaluate the parameters at the exit (or throat) of a cascade including velocity triangles,
    thermodynamic properties, and loss coefficients.

    This function calculates the performance data at the exit of a cascade based on the cascade geometry,
    fluid, and flow conditions. It computes velocity triangles, static and stagnation
    properties, Reynolds and Mach numbers, mass flow rate, and loss coefficients. The calculations
    of the mass flow rate considers the blockage induced by the boundary layer displacement thickness.

    Parameters
    ----------
    cascade_exit_input : dict
        Input parameters specific to the cascade exit, including relative velocity ('w'),
        relative flow angle ('beta'), entropy ('s'), and rothalpy.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade such as chord length, opening, and area.
    inlet_plane : dict
        performance data at the inlet plane of the cascade (needed for loss model calculations).
    angular_speed : float
        Angular speed of the cascade.
    blockage : str or float or None
        The method or value for determining the throat blockage. It can be
        a string specifying a model name ('flat_plate_turbulent'), a numeric
        value between 0 and 1 representing the blockage factor directly, or
        None to use a default calculation method.
    loss_model : str
        The loss model used for calculating loss coefficients.
    radius : float
        Mean radius at the cascade exit.
    area : float
        Area of the cascade exit plane.

    Returns
    -------
    tuple
        A tuple containing:

        - plane (dict): A dictionary of calculated performance data at the cascade exit including
          velocity triangles, thermodynamic properties, Mach and Reynolds numbers, mass flow rate,
          and loss coefficients.
        - loss_dict (dict): A dictionary of loss breakdwon as calculated by the loss model.

    Warnings
    --------
    The current implementation of this calculation has limitations regarding the relationship between
    the throat and exit areas. It does not function correctly unless these areas are related according
    to the cosine rule. Specifically, issues arise when:

    1. The blockage factor at the throat and the exit are not the same. The code may converge for minor
       discrepancies (about <1%), but larger differences lead to convergence issues.
    2. The "throat_location_fraction" parameter is less than one, implying a change in the throat radius
       or height relative to the exit plane.

    These limitations restrict the code's application to cases with constant blade radius or height, or
    when blockage factors at the throat and exit planes are identical. Future versions aim to address
    these issues, enhancing the code's generality for varied geometrical configurations and differing
    blockage factors at the throat and exit planes.

    """

    # Load cascade exit variables
    w = cascade_exit_input["w"]
    beta = cascade_exit_input["beta"]
    s = cascade_exit_input["s"]
    rothalpy = cascade_exit_input["rothalpy"]

    # Load geometry
    chord = geometry["chord"]
    opening = geometry["opening"]
    area = geometry["A_out"]
    radius = geometry["radius_mean_out"]

    # Calculate velocity triangles
    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]
    w_m = velocity_triangle["w_m"]

    # Calculate static properties
    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    static_properties = fluid.get_props(cp.HmassSmass_INPUTS, h, s)
    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

    # Calculate stagnation properties
    h0 = h + 0.5 * v**2
    stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    # Calculate relative stagnation properties
    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(
        relative_stagnation_properties, "0_rel"
    )

    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    rothalpy = h0_rel - 0.5 * blade_speed**2
    # TODO: is rothalpy necessary? See calculations above
    # TODO: not really. But it needs to be assigned either way
    # TODO: Could be assignes as nan, but might as well calculate it
    # TODO: Can also easiliy check that rothalpy is conserved in this way

    # Compute mass flow rate
    blockage_factor = compute_blockage_boundary_layer(blockage, Re, chord, opening)
    mass_flow = rho * w_m * area * (1 - blockage_factor)

    # Evaluate loss coefficient
    # TODO: why not give all variables to the loss model? 
    # Introduce safeguard to prevent negative values for Re and Ma
    # Useful to avoid invalid operations during convergence
    min_val = 1e-3
    loss_model_input = {
        "geometry": geometry,
        "loss_model": loss_model,
        "flow": {
            "p0_rel_in": inlet_plane["p0_rel"],
            "p0_rel_out": relative_stagnation_properties["p0_rel"],
            "p_in": inlet_plane["p"],
            "p_out": static_properties["p"],
            "beta_out": beta,
            "beta_in": inlet_plane["beta"],
            "Ma_rel_in": max(min_val, inlet_plane["Ma_rel"]),
            "Ma_rel_out": max(min_val, Ma_rel),
            "Re_in": max(min_val, inlet_plane["Re"]),
            "Re_out": max(min_val, Re),
            "gamma_out": static_properties["gamma"],
        },
    }

    # Compute loss coefficient from loss model
    loss_dict = lm.evaluate_loss_model(loss_model, loss_model_input)

    # Store results in dictionary
    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
        **loss_dict,
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": mass_flow,
        "displacement_thickness": np.nan,  # Not relevant for exit/throat plane
        "rothalpy": rothalpy,
        "blockage": blockage_factor,
    }
    return plane, loss_dict

def evaluate_cascade_throat(
    cascade_throat_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    blockage,
    loss_model,
):
    """
    Evaluate the parameters at the exit (or throat) of a cascade including velocity triangles,
    thermodynamic properties, and loss coefficients.

    This function calculates the performance data at the exit of a cascade based on the cascade geometry,
    fluid, and flow conditions. It computes velocity triangles, static and stagnation
    properties, Reynolds and Mach numbers, mass flow rate, and loss coefficients. The calculations
    of the mass flow rate considers the blockage induced by the boundary layer displacement thickness.

    Parameters
    ----------
    cascade_exit_input : dict
        Input parameters specific to the cascade exit, including relative velocity ('w'),
        relative flow angle ('beta'), entropy ('s'), and rothalpy.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade such as chord length, opening, and area.
    inlet_plane : dict
        performance data at the inlet plane of the cascade (needed for loss model calculations).
    angular_speed : float
        Angular speed of the cascade.
    blockage : str or float or None
        The method or value for determining the throat blockage. It can be
        a string specifying a model name ('flat_plate_turbulent'), a numeric
        value between 0 and 1 representing the blockage factor directly, or
        None to use a default calculation method.
    loss_model : str
        The loss model used for calculating loss coefficients.
    radius : float
        Mean radius at the cascade exit.
    area : float
        Area of the cascade exit plane.

    Returns
    -------
    tuple
        A tuple containing:

        - plane (dict): A dictionary of calculated performance data at the cascade exit including
          velocity triangles, thermodynamic properties, Mach and Reynolds numbers, mass flow rate,
          and loss coefficients.
        - loss_dict (dict): A dictionary of loss breakdwon as calculated by the loss model.

    Warnings
    --------
    The current implementation of this calculation has limitations regarding the relationship between
    the throat and exit areas. It does not function correctly unless these areas are related according
    to the cosine rule. Specifically, issues arise when:

    1. The blockage factor at the throat and the exit are not the same. The code may converge for minor
       discrepancies (about <1%), but larger differences lead to convergence issues.
    2. The "throat_location_fraction" parameter is less than one, implying a change in the throat radius
       or height relative to the exit plane.

    These limitations restrict the code's application to cases with constant blade radius or height, or
    when blockage factors at the throat and exit planes are identical. Future versions aim to address
    these issues, enhancing the code's generality for varied geometrical configurations and differing
    blockage factors at the throat and exit planes.

    """

    # Load cascade exit variables
    w = cascade_throat_input["w"]
    beta = cascade_throat_input["beta"]
    s = cascade_throat_input["s"]
    rothalpy = cascade_throat_input["rothalpy"]

    # Load geometry
    chord = geometry["chord"]
    opening = geometry["opening"]
    area = geometry["A_throat"]
    radius = geometry["radius_mean_throat"]


    # Calculate velocity triangles
    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]

    # Calculate static properties
    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    static_properties = fluid.get_props(cp.HmassSmass_INPUTS, h, s)
    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

    # Calculate stagnation properties
    h0 = h + 0.5 * v**2
    stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0, s)
    stagnation_properties = utils.add_string_to_keys(stagnation_properties, "0")

    # Calculate relative stagnation properties
    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.get_props(cp.HmassSmass_INPUTS, h0_rel, s)
    relative_stagnation_properties = utils.add_string_to_keys(
        relative_stagnation_properties, "0_rel"
    )

    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    rothalpy = h0_rel - 0.5 * blade_speed**2
    # TODO: is rothalpy necessary? See calculations above
    # TODO: not really. But it needs to be assigned either way
    # TODO: Could be assignes as nan, but might as well calculate it
    # TODO: Can also easiliy check that rothalpy is conserved in this way

    # Compute mass flow rate
    blockage_factor = compute_blockage_boundary_layer(blockage, Re, chord, opening)
    mass_flow = rho * w * area * (1 - blockage_factor)

    # Evaluate loss coefficient
    # TODO: why not give all variables to the loss model? 
    # Introduce safeguard to prevent negative values for Re and Ma
    # Useful to avoid invalid operations during convergence
    min_val = 1e-3
    loss_model_input = {
        "geometry": geometry,
        "loss_model": loss_model,
        "flow": {
            "p0_rel_in": inlet_plane["p0_rel"],
            "p0_rel_out": relative_stagnation_properties["p0_rel"],
            "p_in": inlet_plane["p"],
            "p_out": static_properties["p"],
            "beta_out": beta,
            "beta_in": inlet_plane["beta"],
            "Ma_rel_in": max(min_val, inlet_plane["Ma_rel"]),
            "Ma_rel_out": max(min_val, Ma_rel),
            "Re_in": max(min_val, inlet_plane["Re"]),
            "Re_out": max(min_val, Re),
            "gamma_out": static_properties["gamma"],
        },
    }

    # Compute loss coefficient from loss model
    loss_dict = lm.evaluate_loss_model(loss_model, loss_model_input)

    # Store results in dictionary
    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
        **loss_dict,
        "Ma": Ma,
        "Ma_rel": Ma_rel,
        "Re": Re,
        "mass_flow": mass_flow,
        "displacement_thickness": np.nan,  # Not relevant for exit/throat plane
        "rothalpy": rothalpy,
        "blockage": blockage_factor,
    }
    return plane, loss_dict

def evaluate_cascade_interspace(
    h0_exit,
    v_m_exit,
    v_t_exit,
    rho_exit,
    radius_exit,
    area_exit,
    blockage_exit,
    radius_inlet,
    area_inlet,
    fluid,
):
    """
    Calculate the inlet conditions for the next cascade based on the exit conditions of the previous cascade.

    This function computes the inlet thermodynamic and velocity conditions the next cascade using the exit conditions
    from the previous cascade and the flow equations for the interspace between cascades.

    Assumptions:

    1. No heat transfer (conservation of stagnation enthalpy)
    2. No friction (conservation of angular momentum)
    3. No density variation (incompressible limit)


    Parameters
    ----------
    h0_exit : float
        Stagnation enthalpy at the exit of the previous cascade.
    v_m_exit : float
        Meridional component of velocity at the exit of the previous cascade.
    v_t_exit : float
        Tangential component of velocity at the exit of the previous cascade.
    rho_exit : float
        Fluid density at the exit of the previous cascade.
    radius_exit : float
        Radius at the exit of the previous cascade.
    area_exit : float
        Flow area at the exit of the previous cascade.
    radius_inlet : float
        Radius at the inlet of the next cascade.
    area_inlet : float
        Flow area at the inlet of the next cascade.
    fluid : object
        A fluid object with methods for thermodynamic property calculations (e.g., CoolProp fluid).

    Returns
    -------
    tuple
        A tuple containing:

        - h0_in (float): Stagnation enthalpy at the inlet of the next cascade.
        - s_in (float): Entropy at the inlet of the next cascade.
        - alpha_in (float): Flow angle at the inlet of the next cascade (in radians).
        - v_in (float): Magnitude of the velocity vector at the inlet of the next cascade.

    Warnings
    --------
    The assumption of constant density leads to a small inconsistency in the thermodynamic state,
    manifesting as a slight variation in entropy across the interspace. In future versions of the
    code, it is recommended to evaluate the interspace with a 1D model for the flow in annular ducts
    to improve accuracy and consistency in the analysis.
    """

    # Assume no heat transfer
    h0_in = h0_exit

    # Assume no friction (angular momentum is conserved)
    v_t_in = v_t_exit * radius_exit / radius_inlet

    # Assume density variation is negligible
    v_m_in = v_m_exit * area_exit / area_inlet *(1-blockage_exit)

    # Compute velocity vector
    v_in = np.sqrt(v_t_in**2 + v_m_in**2)
    alpha_in = math.arctand(v_t_in / v_m_in)

    # Compute thermodynamic state
    h_in = h0_in - 0.5 * v_in**2
    rho_in = rho_exit
    stagnation_properties = fluid.get_props(cp.DmassHmass_INPUTS, rho_in, h_in)
    s_in = stagnation_properties["s"]

    return h0_in, s_in, alpha_in, v_in

def evaluate_velocity_triangle_in(u, v, alpha):
    """
    Compute the velocity triangle at the inlet of the cascade.

    This function calculates the components of the velocity triangle at the
    inlet of a cascade, based on the blade speed, absolute velocity, and
    absolute flow angle. It computes both the absolute and relative velocity
    components in the meridional and tangential directions, as well as the
    relative flow angle.

    Parameters
    ----------
    u : float
        Blade speed.
    v : float
        Absolute velocity.
    alpha : float
        Absolute flow angle in radians.

    Returns
    -------
    dict
        A dictionary containing the following properties:

        - "u" (float): Blade velocity.
        - "v" (float): Absolute velocity.
        - "v_m" (float): Meridional component of absolute velocity.
        - "v_t" (float): Tangential component of absolute velocity.
        - "alpha" (float): Absolute flow angle in radians.
        - "w" (float): Relative velocity magnitude.
        - "w_m" (float): Meridional component of relative velocity.
        - "w_t" (float): Tangential component of relative velocity.
        - "beta" (float): Relative flow angle in radians.
    """

    # Absolute velocities
    v_t = v * math.sind(alpha)
    v_m = v * math.cosd(alpha)

    # Relative velocities
    w_t = v_t - u
    w_m = v_m
    w = np.sqrt(w_t**2 + w_m**2)

    # Relative flow angle
    beta = math.arctand(w_t / w_m)

    # Store in dict
    vel_in = {
        "u": u,
        "v": v,
        "v_m": v_m,
        "v_t": v_t,
        "alpha": alpha,
        "w": w,
        "w_m": w_m,
        "w_t": w_t,
        "beta": beta,
    }

    return vel_in

def evaluate_velocity_triangle_out(u, w, beta):
    """
    Compute the velocity triangle at the outlet of the cascade.

    This function calculates the components of the velocity triangle at the
    outlet of a cascade, based on the blade speed, relative velocity, and
    relative flow angle. It computes both the absolute and relative velocity
    components in the meridional and tangential directions, as well as the
    absolute flow angle.

    Parameters
    ----------
    u : float
        Blade speed.
    w : float
        Relative velocity.
    beta : float
        Relative flow angle in radians.

    Returns
    -------
    dict
        A dictionary containing the following properties:

        - "u" (float): Blade velocity.
        - "v" (float): Absolute velocity.
        - "v_m" (float): Meridional component of absolute velocity.
        - "v_t" (float): Tangential component of absolute velocity.
        - "alpha" (float): Absolute flow angle in radians.
        - "w" (float): Relative velocity magnitude.
        - "w_m" (float): Meridional component of relative velocity.
        - "w_t" (float): Tangential component of relative velocity.
        - "beta" (float): Relative flow angle in radians.
    """

    # Relative velocities
    w_t = w * math.sind(beta)
    w_m = w * math.cosd(beta)

    # Absolute velocities
    v_t = w_t + u
    v_m = w_m
    v = np.sqrt(v_t**2 + v_m**2)

    # Absolute flow angle
    alpha = math.arctand(v_t / v_m)

    # Store in dict
    vel_out = {
        "u": u,
        "v": v,
        "v_m": v_m,
        "v_t": v_t,
        "alpha": alpha,
        "w": w,
        "w_m": w_m,
        "w_t": w_t,
        "beta": beta,
    }

    return vel_out

def compute_blockage_boundary_layer(blockage_model, Re, chord, opening):
    r"""
    Calculate the blockage factor due to boundary layer displacement thickness.

    This function computes the blockage factor caused by the boundary layer
    displacement thickness at a given flow station. The blockage factor affects
    mass flow rate calculations effectively reducing the flow area.

    The blockage factor can be determined through various methods, including:

        1. Calculation based on a correlation for the displacement thickness of turbulent boundary layer over a flat plate with zero pressure gradient.
        2. Using a numerical value specified directly by the user.

    The correlation for turbulent boundary layer displacement thickness over a flat plate is given by :cite:`cengel_fluid_2014`:
    
    .. math::
        \delta^* = \frac{0.048}{Re^{1/5}} \times 0.9 \times \text{chord}
        
    From this the blockage factor is calculated as
    
    .. math::
        \text{blockage_factor} = 2 \times \frac{\delta^*}{\text{opening}}

    Parameters
    ----------
    blockage_model : str or float or None
        The method or value for determining the blockage factor. It can be
        a string specifying a model name ('flat_plate_turbulent'), a numeric
        value between 0 and 1 representing the blockage factor directly, or
        None to use a default calculation method.
    Re : float, optional
        Reynolds number, used if `blockage_model` is 'flat_plate_turbulent'.
    chord : float, optional
        Chord length, used if `blockage_model` is 'flat_plate_turbulent'.
    opening : float, optional
        Throat opening size, used to calculate the blockage factor when
        `blockage_model` is None or a numeric value.

    Returns
    -------
    float
        The calculated blockage factor, a value between 0 and 1, where 1
        indicates full blockage and 0 indicates no blockage.

    Raises
    ------
    ValueError
        If `blockage_model` is an invalid option, or required parameters
        for the chosen method are missing.
    """

    if blockage_model == BLOCKAGE_MODELS[0]:
        displacement_thickness = 0.048 / Re ** (1 / 5) * 0.9 * chord
        blockage_factor = 2 * displacement_thickness / opening

    elif isinstance(blockage_model, (float, int)) and 0 <= blockage_model <= 1:
        blockage_factor = blockage_model

    elif blockage_model is None:
        blockage_factor = 0.00

    else:
        raise ValueError(
            f"Invalid throat blockage option: '{blockage_model}'. "
            "Valid options are 'flat_plate_turbulent', a numeric value between 0 and 1, or None."
        )

    return blockage_factor


def compute_efficiency_breakdown(results):
    """
    Compute the loss of total-to-static efficiency due to each various loss component in cascades.

    This function calculates the fraction of total-to-static efficiency drop attributed to each loss component
    in a turbine cascade. A correction factor for the re-heating effect is applied to align the sum of individual
    cascade losses with the total observed change in enthalpy and ensure consistency with
    the overall total-to-static efficiency.

    Parameters
    ----------
    results : dict
        The data for the cascades, including plane and cascade specific parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing efficiency drop fractions for each loss type in each cascade.
        Columns are named as 'efficiency_drop_{loss_type}', where 'loss_type' includes:

        - 'profile'
        - 'incidence'
        - 'secondary'
        - 'clearance'
        - 'trailing'

    """
    # Load parameters
    h_out_s = results["reference_values"]["h_out_s"]
    h0_in = results["plane"]["h0"].values[0]
    h_out = results["plane"]["h"].values[-1]
    cascade = results["cascade"]
    number_of_cascades = results["geometry"]["number_of_cascades"]

    # Compute a correction factor due to re-heating effect
    dhs_total = h_out - h_out_s
    dhs_sum = cascade["dh_s"].sum()
    correction = dhs_total / dhs_sum

    # Initialize DataFrame
    loss_types = ["profile", "incidence", "secondary", "clearance", "trailing"]
    breakdown = pd.DataFrame(columns=[f"efficiency_drop_{type}" for type in loss_types])

    # Loss breakdown in each cascade
    for i in range(number_of_cascades):
        # Construct column names dynamically based on loss_types
        col_names = [f"loss_{type}" for type in loss_types]

        # Retrieve values for each specified column and compute the fractions
        fracs = cascade.loc[i, col_names] / cascade.loc[i, "loss_total"]
        dh_s = cascade["dh_s"][i]
        efficiency_drop = correction * dh_s / (h0_in - h_out_s)

        # Append efficiency drop breakdown
        breakdown.loc[len(breakdown)] = (fracs * efficiency_drop).values

    return breakdown


def compute_stage_performance(results):
    """
    Calculate the stage performance metrics of the turbine

    Parameters
    ----------
    number_of_stages : int
        The number of stages in the cascading system.
    planes : dict
        A dictionary containing performance data at each plane in the cascading system.

    Returns
    -------
    dict
        A dictionary with calculated stage parameters. Currently, this includes:
        - 'R' (numpy.ndarray): An array of degree of reaction values for each stage.

    Warnings
    --------
    This function currently only computes the degree of reaction. Other stage parameters,
    which might be relevant for a comprehensive analysis of the cascading system, are not
    computed in this version of the function. Future implementations may include additional
    parameters for a more detailed stage analysis.

    """

    # Only proceed if there are stages
    number_of_stages = results["geometry"]["number_of_stages"]
    if number_of_stages == 0:
        return {}

    # Calculate the degree of reaction for each stage using list comprehension
    h = results["plane"]["h"].values
    R = np.array(
        [
            (h[i * 6 + 2] - h[i * 6 + 5]) / (h[i * 6] - h[i * 6 + 5])
            for i in range(number_of_stages)
        ]
    )

    # Store all variables in dictionary
    stages = {"reaction": R}

    # Check the dictionary has the expected keys
    utils.validate_keys(stages, KEYS_STAGE, KEYS_STAGE)

    return stages


def compute_overall_performance(results):
    """
    Calculate the overall performance metrics of the turbine

    This function extracts necessary values from the 'results' dictionary, performs calculations to determine
    overall performance metrics, and stores these metrics in the "overall" dictionary. The function also checks
    that the overall dictionary has all the expected keys and does not contain any unexpected keys.

    The keys for the 'overall' dictionary are defined in the :obj:`KEYS_OVERALL` list. Refer to its documentation
    for the complete list of keys.

    Parameters
    ----------
    results : dict
        A dictionary containing various operational parameters at each flow station

    Returns
    -------
    dict
        An updated dictionary with a new key 'overall' containing the calculated performance metrics.
        The keys for these metrics are defined in :obj:`KEYS_OVERALL`.
    """

    # TODO: Refactor so results is the only required argument
    # TODO: recalculate h_out_s and v0 from results data might be a simple solution
    # TODO: angular speed could be saved at each cascade. This would be needed anyways if we want to extend the code to machines with multiple angular speeds in the future (e.g., a multi-shaft gas turbine)
    angular_speed = results["boundary_conditions"]["omega"]
    v0 = results["reference_values"]["v0"]
    h_out_s = results["reference_values"]["h_out_s"]  # FIXME: bad plaement of h_out_s variable?

    # Calculation of overall performance
    p = results["plane"]["p"].values
    p0 = results["plane"]["p0"].values
    h0 = results["plane"]["h0"].values
    v_out = results["plane"]["v"].values[-1]
    u_out = results["plane"]["u"].values[-1]
    mass_flow = results["plane"]["mass_flow"].values[-1]
    exit_flow_angle = results["plane"]["alpha"].values[-1]
    PR_tt = p0[0] / p0[-1]
    PR_ts = p0[0] / p[-1]
    h0_in = h0[0]
    h0_out = h0[-1]
    efficiency_tt = (h0_in - h0_out) / (h0_in - h_out_s - 0.5 * v_out**2)*100
    efficiency_ts = (h0_in - h0_out) / (h0_in - h_out_s)*100
    efficiency_ts_drop_kinetic = 0.5 * v_out**2 / (h0_in - h_out_s)
    efficiency_ts_drop_losses = 1.0 - efficiency_ts - efficiency_ts_drop_kinetic
    power = mass_flow * (h0_in - h0_out)
    torque = power / angular_speed

    # Store all variables in dictionary
    overall = {
        "PR_tt": PR_tt,
        "PR_ts": PR_ts,
        "mass_flow_rate": mass_flow,
        "efficiency_tt": efficiency_tt,
        "efficiency_ts": efficiency_ts,
        "efficiency_ts_drop_kinetic": efficiency_ts_drop_kinetic,
        "efficiency_ts_drop_losses": efficiency_ts_drop_losses,
        "power": power,
        "torque": torque,
        "angular_speed": angular_speed,
        "exit_flow_angle": exit_flow_angle,
        "exit_velocity": v_out,
        "spouting_velocity": v0,
        "last_blade_velocity": u_out,
        "blade_jet_ratio": u_out / v0,
        "h0_in": h0_in,
        "h0_out": h0_out,
        "h_out_s": h_out_s,
    }

    # Check the dictionary has the expected keys
    utils.validate_keys(overall, KEYS_OVERALL, KEYS_OVERALL)

    return overall

# TODO:
# Change blade speed from u to 'blade speed'
# Now it is overwritten by internal energy in the ouput file
