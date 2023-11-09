import numpy as np
import pandas as pd
from scipy import optimize
import CoolProp as CP
from ..properties import FluidCoolProp_2Phase
from ..math import smooth_max, sind, cosd, tand
from . import loss_model as lm
from . import deviation_model as dm
from scipy.optimize._numdiff import approx_derivative

from .. import math

# Keys of the information that should be stored in results
keys_plane = [
    "v",
    "v_m",
    "v_t",
    "alpha",
    "w",
    "w_m",
    "w_t",
    "beta",
    "p",
    "T",
    "h",
    "s",
    "d",
    "Z",
    "a",
    "mu",
    "k",
    "cp",
    "cv",
    "gamma",
    "p0",
    "T0",
    "h0",
    "s0",
    "d0",
    "Z0",
    "a0",
    "mu0",
    "k0",
    "cp0",
    "cv0",
    "gamma0",
    "p0rel",
    "T0rel",
    "h0rel",
    "s0rel",
    "d0rel",
    "Z0rel",
    "a0rel",
    "mu0rel",
    "k0rel",
    "cp0rel",
    "cv0rel",
    "gamma0rel",
    "Ma",
    "Ma_rel",
    "Re",
    "m",
    "delta",
    "rothalpy",
    "Y_err",
    "Y",
    "Y_p",
    "Y_cl",
    "Y_s",
    "Y_te",
    "Y_inc",
    "blockage",
]

keys_cascade = [
    "Y_tot",
    "Y_p",
    "Y_s",
    "Y_cl",
    "dh_s",
    "Ma_crit",
    "m_crit",
    "incidence",
    "density_correction",
]


def evaluate_cascade_series(
    variables_values, BC, geometry, fluid, model_options, reference_values, results
):
    
    # TODO
    # Convert all angles from radians to

    # Load variables
    n_cascades = geometry["n_cascades"]
    h0_in = BC["h0_in"]
    s_in = BC["s_in"]
    alpha_in = BC["alpha_in"]
    angular_speed = BC["omega"]

    # Load reference_values
    h_out_s = reference_values["h_out_s"]  # FIXME: bad plaement of h_out_s variable?
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Initialize cascades data
    df = pd.DataFrame(columns=keys_plane)
    results["plane"] = df
    df = pd.DataFrame(columns=keys_cascade)
    results["cascade"] = df

    # initialize residual arrays
    residuals_values = np.array([])
    residuals_keys = np.array([])

    # Define degrees of freedom #FIXME: independant variables could be handled differently
    n_dof = 5
    n_dof_crit = 3
    variables_actual = variables_values[0 : n_dof * n_cascades + 1]
    variables_critical = variables_values[n_dof * n_cascades + 1 :]
    v_in = variables_actual[0] * v0
    variables_actual = np.delete(variables_actual, 0)

    for i in range(n_cascades):
        # Update angular speed
        angular_speed_cascade = angular_speed * (i % 2)

        # update geometry for current cascade
        geometry_cascade = {
            key: values[i]
            for key, values in geometry.items()
            if key not in ["n_cascades", "n_stages"]
        }

        # Rename independant variables #FIXME: independant variables could be handled differently
        variables_actual_cascade = variables_actual[i * n_dof : (i + 1) * n_dof]
        variables_critical_cascade = variables_critical[
            i * n_dof_crit : (i + 1) * n_dof_crit
        ]
        w_throat = variables_actual_cascade[0] * v0
        w_out = variables_actual_cascade[1] * v0
        s_throat = variables_actual_cascade[2] * s_range + s_min
        s_out = variables_actual_cascade[3] * s_range + s_min
        beta_out = variables_actual_cascade[4] * angle_range + angle_min  # TODO convert to radians
        v_crit_in = variables_critical_cascade[0]
        w_crit_out = variables_critical_cascade[1]
        s_crit_out = variables_critical_cascade[2]

        # Evaluate current cascade
        cascade_inlet_input = {
            "h0": h0_in,
            "s": s_in,
            "alpha": alpha_in,
            "v": v_in,
        }
        cascade_throat_input = {"w": w_throat, "s": s_throat}
        cascade_exit_input = {
            "w": w_out,
            "beta": beta_out,
            "s": s_out,
        }
        critical_cascade_input = {
            "v*_in": v_crit_in,
            "w*_out": w_crit_out,
            "s*_out": s_crit_out,
        }
        cascade_residual_values, cascade_residual_keys = evaluate_cascade(
            cascade_inlet_input,
            cascade_throat_input,
            cascade_exit_input,
            critical_cascade_input,
            fluid,
            geometry_cascade,
            angular_speed_cascade,
            results,
            model_options,
            reference_values,
        )

        # Add cascade residuals to residual arrays
        residuals_values = np.concatenate((residuals_values, cascade_residual_values))
        residuals_keys = np.concatenate((residuals_keys, cascade_residual_keys))

        # Calculate input of next cascade (Assume no change in density)
        if i != n_cascades - 1:
            (
                h0_in,
                s_in,
                alpha_in,
                v_in,
            ) = evaluate_inter_cascade_space(
                results["plane"]["h0"].values[-1],
                results["plane"]["v_m"].values[-1],
                results["plane"]["v_t"].values[-1],
                results["plane"]["d"].values[-1],
                fluid,
                geometry_cascade["r_out"],
                geometry_cascade["A_out"],
                geometry["r_in"][i + 1],
                geometry["A_in"][i + 1],
            )

    # Add exit pressure error to residuals
    p_calc = results["plane"]["p"].values[-1]
    p_error = (p_calc - BC["p_out"]) / BC["p0_in"]
    residuals_values = np.append(residuals_values, p_error)
    residuals_keys = np.append(residuals_keys, "p_out")

    # Store global variables
    v_out = results["plane"]["v"].values[-1]
    m = results["plane"]["m"].values[-1]
    h0_in = results["plane"]["h0"].values[0]
    h0_out = results["plane"]["h0"].values[-1]

    results["overall"] = {}
    results["overall"]["mass_flow_rate"] = m

    h0 = results["plane"]["h0"].values
    eta_ts = (h0[0] - h0[-1]) / (h0[0] - h_out_s) * 100
    eta_tt = (h0[0] - h0[-1]) / (h0[0] - h_out_s - 0.5 * v_out**2) * 100
    results["overall"]["eta_ts"] = eta_ts
    results["overall"]["eta_tt"] = eta_tt
    results["overall"]["power"] = m * (h0_in - h0_out)
    results["overall"]["torque"] = m * (h0_in - h0_out) / angular_speed
    results["overall"]["pr_tt"] = BC["p0_in"] / results["plane"]["p0"].values[-1]
    results["overall"]["pr_ts"] = BC["p0_in"] / results["plane"]["p"].values[-1]

    loss_fracs = calculate_eta_drop_fractions(results, n_cascades, reference_values)
    results["loss_fracs"] = loss_fracs
    results["cascade"] = pd.concat([results["cascade"], loss_fracs], axis=1)

    # Add stage results to the results tructure
    if geometry["n_stages"] != 0:
        # Calculate stage variables
        stage_results = calculate_stage_parameters(
            geometry["n_stages"], results["plane"]
        )
        results["stage"] = pd.DataFrame(stage_results)

    return residuals_values, residuals_keys


def evaluate_cascade(
    cascade_inlet_input,
    cascade_throat_input,
    cascade_exit_input,
    critical_cascade_input,
    fluid,
    geometry,
    angular_speed,
    results,
    model_options,
    reference_values,
):
    # Define model options
    displacement_thickness = model_options.get("displacement_thickness", 0)
    loss_model = model_options.get("loss_model", "benner")
    choking_condition = model_options.get("choking_condition", "deviation")
    deviation_model = model_options.get("deviation_model", "aungier")

    # Load reference values
    m_ref = reference_values["m_ref"]
    delta_ref = reference_values["delta_ref"]

    # Define residual array and residual keys array
    residuals_cascade = np.array([])
    keys_cascade = np.array([])

    # Evaluate inlet plane
    inlet_plane = evaluate_inlet(
        cascade_inlet_input, fluid, geometry, angular_speed, delta_ref
    )
    results["plane"].loc[len(results["plane"])] = inlet_plane

    # Evaluate throat plane
    cascade_throat_input["rothalpy"] = inlet_plane["rothalpy"]
    cascade_throat_input["beta"] = geometry["theta_out"]
    throat_plane, Y_info = evaluate_exit(
        cascade_throat_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        displacement_thickness,
        loss_model,
    )
    results["plane"].loc[len(results["plane"])] = throat_plane

    # Evaluate exit plane
    cascade_exit_input["rothalpy"] = inlet_plane["rothalpy"]
    exit_plane, Y_info = evaluate_exit(
        cascade_exit_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        displacement_thickness,
        loss_model,
    )
    results["plane"].loc[len(results["plane"])] = exit_plane

    # Add loss coefficient error to residual array
    residuals_cascade = np.concatenate(
        (residuals_cascade, [throat_plane["Y_err"], exit_plane["Y_err"]])
    )
    keys_cascade = np.concatenate((keys_cascade, ["Y_err_throat", "Y_err_exit"]))

    # Add mass flow rate error
    m_error_throat = inlet_plane["m"] - throat_plane["m"]
    m_error_exit = inlet_plane["m"] - exit_plane["m"]
    residuals_m = np.array([m_error_throat, m_error_exit]) / m_ref
    residuals_cascade = np.concatenate((residuals_cascade, residuals_m))
    keys_cascade = np.concatenate((keys_cascade, ["m_err_throat", "m_err_exit"]))

    # Calculate critical state
    critical_cascade_input["h0_in"] = cascade_inlet_input["h0"]
    critical_cascade_input["s_in"] = cascade_inlet_input["s"]
    critical_cascade_input["alpha_in"] = cascade_inlet_input["alpha"]
    x_crit = np.array(
        [
            critical_cascade_input["v*_in"],
            critical_cascade_input["w*_out"],
            critical_cascade_input["s*_out"],
        ]
    )
    res_critical, res_keys, critical_state = evaluate_lagrange_gradient(
        x_crit,
        critical_cascade_input,
        fluid,
        geometry,
        angular_speed,
        model_options,
        reference_values,
    )

    # Add residuals of calculation of critical state
    residuals_cascade = np.concatenate((residuals_cascade, res_critical))
    keys_cascade = np.concatenate((keys_cascade, res_keys))

    # Add final closing equation (deviaton model or mach error)
    if choking_condition == "deviation":
        density_correction = (
            throat_plane["d"] / critical_state["d"]
        )  # density correction due to nested finite differences
        residual = calculate_deviation_equation(
            geometry, critical_state, exit_plane, density_correction, deviation_model
        )

    elif choking_condition == "mach_critical" or choking_condition == "mach_unity":
        if choking_condition == "mach_unity":
            critical_state["Ma_rel"] = 1
        residual = calculate_mach_equation(
            critical_state["Ma_rel"], exit_plane["Ma_rel"], throat_plane["Ma_rel"]
        )
        density_correction = np.nan
    else:
        raise Exception(
            "choking_condition must be 'deviation', 'mach_critical' or 'mach_unity'"
        )
    residuals_cascade = np.append(residuals_cascade, residual)
    keys_cascade = np.append(keys_cascade, choking_condition)

    # Add cascade results to results structure
    static_properties_isentropic_expansion = fluid.compute_properties_meanline(
        CP.PSmass_INPUTS, exit_plane["p"], inlet_plane["s"]
    )
    dhs = exit_plane["h"] - static_properties_isentropic_expansion["h"]
    cascade_data = {
        "Y_tot": Y_info["Total"],
        "Y_p": Y_info["Profile"],
        "Y_s": Y_info["Secondary"],
        "Y_cl": Y_info["Clearance"],
        "dh_s": dhs,
        "Ma_crit": critical_state["Ma_rel"],
        "m_crit": critical_state["m"],
        "incidence": inlet_plane["beta"] - geometry["theta_in"],
        "density_correction": density_correction,
    }
    results["cascade"].loc[len(results["cascade"])] = cascade_data

    return residuals_cascade, keys_cascade


def evaluate_inlet(cascade_inlet_input, fluid, geometry, angular_speed, delta_ref):
    
    # Load cascade inlet input
    h0 = cascade_inlet_input["h0"]
    s = cascade_inlet_input["s"]
    v = cascade_inlet_input["v"]
    alpha = cascade_inlet_input["alpha"]

    # Load geometry
    radius = geometry["r_in"]
    chord = geometry["c"]
    area = geometry["A_in"]

    # Calculate velocity triangles
    blade_speed = radius * angular_speed
    velocity_triangle = evaluate_velocity_triangle_in(blade_speed, v, alpha)
    w = velocity_triangle["w"]
    w_m = velocity_triangle["w_m"]

    # Calculate static properties
    h = h0 - 0.5 * v**2
    static_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h, s)
    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

    # Calculate stagnation properties
    stagnation_properties = fluid.compute_properties_meanline(
        CP.HmassSmass_INPUTS, h0, s
    )
    stagnation_properties = add_string_to_keys(stagnation_properties, "0")

    # Calculate relatove stagnation properties
    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.compute_properties_meanline(
        CP.HmassSmass_INPUTS, h0_rel, s
    )
    relative_stagnation_properties = add_string_to_keys(
        relative_stagnation_properties, "0_rel"
    )

    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    m = rho * w_m * area
    rothalpy = h0_rel - 0.5 * blade_speed**2

    # Calculate inlet displacement thickness to blade height ratio based on reference vale
    delta = delta_ref * Re ** (-1 / 7)

    # Store result
    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
    }
    plane["Ma"] = Ma
    plane["Ma_rel"] = Ma_rel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = delta
    plane["rothalpy"] = rothalpy
    plane["Y_err"] = np.nan
    plane["Y"] = np.nan
    plane["Y_p"] = np.nan
    plane["Y_cl"] = np.nan
    plane["Y_s"] = np.nan
    plane["Y_te"] = np.nan
    plane["Y_inc"] = np.nan
    plane["blockage"] = np.nan

    return plane


def evaluate_exit(
    cascade_exit_input,
    fluid,
    geometry,
    inlet_plane,
    angular_speed,
    displacement_thickness,
    loss_model,
):
    # Load cascade exit variables
    w = cascade_exit_input["w"]
    beta = cascade_exit_input["beta"]
    s = cascade_exit_input["s"]
    rothalpy = cascade_exit_input["rothalpy"]

    # Load geometry
    radius = geometry["r_out"]
    area = geometry["A_out"]
    chord = geometry["c"]
    opening = geometry["o"]

    # Calculate velocity triangles
    blade_speed = angular_speed * radius
    velocity_triangle = evaluate_velocity_triangle_out(blade_speed, w, beta)
    v = velocity_triangle["v"]
    w_m = velocity_triangle["w_m"]

    # Calculate static properties
    h = rothalpy + 0.5 * blade_speed**2 - 0.5 * w**2
    static_properties = fluid.compute_properties_meanline(CP.HmassSmass_INPUTS, h, s)
    rho = static_properties["d"]
    mu = static_properties["mu"]
    a = static_properties["a"]

    # Calculate stagnation properties
    h0 = h + 0.5 * v**2
    stagnation_properties = fluid.compute_properties_meanline(
        CP.HmassSmass_INPUTS, h0, s
    )
    stagnation_properties = add_string_to_keys(stagnation_properties, "0")

    # Calculate relatove stagnation properties
    h0_rel = h + 0.5 * w**2
    relative_stagnation_properties = fluid.compute_properties_meanline(
        CP.HmassSmass_INPUTS, h0_rel, s
    )
    relative_stagnation_properties = add_string_to_keys(
        relative_stagnation_properties, "0_rel"
    )

    # Calculate mach, reynolds and mass flow rate for cascade inlet
    Ma = v / a
    Ma_rel = w / a
    Re = rho * w * chord / mu
    m = rho * w_m * area
    rothalpy = h0_rel - 0.5 * blade_speed**2

    # Account for blockage effect due to boundary layer displacement thickness
    if displacement_thickness == None:
        displacement_thickness = 0.048 / Re ** (1 / 5) * 0.9 * chord
    correction = 1 - 2 * displacement_thickness / opening
    m *= correction

    # Evaluate loss coefficient
    loss_model_input = {
        "geometry": geometry,
        "flow": {},
        "loss_model": loss_model,
        "type": "stator" * (blade_speed == 0) + "rotor" * (blade_speed != 0),
    }

    loss_model_input["flow"]["p0_rel_in"] = inlet_plane["p0_rel"]
    loss_model_input["flow"]["p0_rel_out"] = relative_stagnation_properties["p0_rel"]
    loss_model_input["flow"]["p_in"] = inlet_plane["p"]
    loss_model_input["flow"]["p_out"] = static_properties["p"]
    loss_model_input["flow"]["beta_out"] = beta
    loss_model_input["flow"]["beta_in"] = inlet_plane["beta"]
    loss_model_input["flow"]["Ma_rel_in"] = inlet_plane["Ma_rel"]
    loss_model_input["flow"]["Ma_rel_out"] = Ma_rel
    loss_model_input["flow"]["Re_in"] = inlet_plane["Re"]
    loss_model_input["flow"]["Re_out"] = Re
    loss_model_input["flow"]["delta"] = inlet_plane["delta"]
    loss_model_input["flow"]["gamma_out"] = static_properties["gamma"]

    Y, Y_err, Y_info = evaluate_loss_model(loss_model_input)

    # Store result
    plane = {
        **velocity_triangle,
        **static_properties,
        **stagnation_properties,
        **relative_stagnation_properties,
    }

    plane["Ma"] = Ma
    plane["Ma_rel"] = Ma_rel
    plane["Re"] = Re
    plane["m"] = m
    plane["delta"] = np.nan  # Not relevant for exit/throat plane
    plane["rothalpy"] = rothalpy
    plane["Y_err"] = Y_err
    plane["Y"] = Y
    plane["Y_p"] = Y_info["Profile"]
    plane["Y_cl"] = Y_info["Clearance"]
    plane["Y_s"] = Y_info["Secondary"]
    plane["Y_te"] = Y_info["Trailing"]
    plane["Y_inc"] = Y_info["Incidence"]
    plane["blockage"] = correction

    return plane, Y_info


def evaluate_lagrange_gradient(
    x_crit,
    critical_cascade_input,
    fluid,
    geometry,
    angular_speed,
    model_options,
    reference_values,
):
    """
    Evaluate the gradient of the Lagrange function of the critical mass flow rate function.

    The function assumes specific structures in the critical_cascade_data dictionary.
    Please ensure the required keys are present for accurate evaluation.

    This function evaluates the gradient of the Lagrange function of the critical mass flow rate function.
    It calculates the Lagrange multipliers explicitly and returns the residuals of the Lagrange gradient.

    
    Parameters
    ----------
        x (numpy.ndarray): Array containing [v_in*, V_throat*, s_throat*].
        critical_cascade_data (dict): Dictionary containing critical cascade data.

    Returns
    -------
    numpy.ndarray:
        Residuals of the Lagrange function.

    """

    # Load reference values
    m_ref = reference_values["m_ref"]

    # Define critical state dictionary to store information
    critical_state = {}

    # Evaluate the current cascade at critical conditions
    f0 = evaluate_critical_cascade(
        x_crit,
        critical_cascade_input,
        fluid,
        geometry,
        angular_speed,
        critical_state,
        model_options,
        reference_values,
    )

    # Evaluate the Jacobian of the evaluate_critical_cascade function
    J = compute_critical_cascade_jacobian(
        x_crit,
        critical_cascade_input,
        fluid,
        geometry,
        angular_speed,
        critical_state,
        model_options,
        reference_values,
        f0,
    )

    # Rename gradients
    a11, a12, a21, a22, b1, b2 = (
        J[1, 0],
        J[2, 0],
        J[1, 1 + 1],
        J[2, 1 + 1],
        -1 * J[0, 0],
        -1 * J[0, 1 + 1],
    )  # For isentropic

    # Calculate the Lagrange multipliers explicitly
    l1 = (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21)
    l2 = (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)

    # Evaluate the last equation
    df, dg1, dg2 = J[0, 2 - 1], J[1, 2 - 1], J[2, 2 - 1]  # for isentropic
    grad = (df + l1 * dg1 + l2 * dg2) / m_ref

    # Return last 3 equations of the Lagrangian gradient (df/dx2+l1*dg1/dx2+l2*dg2/dx2 and g1, g2)
    g = f0[1:]  # The two constraints
    residual_values = np.insert(g, 0, grad)
    residual_keys = ["L*", "m*", "Y*"]

    return residual_values, residual_keys, critical_state


def evaluate_critical_cascade(
    x_crit,
    critical_cascade_input,
    fluid,
    geometry,
    angular_speed,
    critical_state,
    model_options,
    reference_values,
):
    # Load reference values
    m_ref = reference_values["m_ref"]
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    delta_ref = reference_values["delta_ref"]

    # Load geometry
    theta_out = geometry["theta_out"]

    # Load cinput for critical cascade
    s_in = critical_cascade_input["s_in"]
    h0_in = critical_cascade_input["h0_in"]
    alpha_in = critical_cascade_input["alpha_in"]

    v_in, w_throat, s_throat = (
        x_crit[0] * v0,
        x_crit[1] * v0,
        x_crit[2] * s_range + s_min,
    )

    # Define model options
    displacement_thickness = model_options.get("displacement_thickness", 0)
    loss_model = model_options.get("loss_model", "benner")

    # Evaluate inlet plane
    critical_inlet_input = {
        "v": v_in,
        "s": s_in,
        "h0": h0_in,
        "alpha": alpha_in,
    }
    inlet_plane = evaluate_inlet(
        critical_inlet_input, fluid, geometry, angular_speed, delta_ref
    )

    # Evaluate throat plane
    critical_exit_input = {
        "w": w_throat,
        "s": s_throat,
        "beta": theta_out,
        "rothalpy": inlet_plane["rothalpy"],
    }
    throat_plane, Y_info = evaluate_exit(
        critical_exit_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        displacement_thickness,
        loss_model,
    )

    # Add residuals
    residuals = np.array(
        [(inlet_plane["m"] - throat_plane["m"]) / m_ref, throat_plane["Y_err"]]
    )

    critical_state["m"] = throat_plane["m"]
    critical_state["Ma_rel"] = throat_plane["Ma_rel"]
    critical_state["d"] = throat_plane["d"]

    output = np.insert(residuals, 0, throat_plane["m"])

    return output


def compute_critical_cascade_jacobian(
    x,
    critical_cascade_input,
    fluid,
    geometry,
    angular_speed,
    critical_state,
    model_options,
    reference_values,
    f0,
):
    eps = 1e-3 * x
    J = approx_derivative(
        evaluate_critical_cascade,
        x,
        method="2-point",
        f0=f0,
        abs_step=eps,
        args=(
            critical_cascade_input,
            fluid,
            geometry,
            angular_speed,
            critical_state,
            model_options,
            reference_values,
        ),
    )

    return J


def evaluate_velocity_triangle_in(u, v, alpha):
    """
    Compute the velocity triangle at the inlet of the cascade.

    Args:
        u (float): Blade speed.
        v (float): Absolute velocity.
        alpha (float): Absolute flow angle (in radians).

    Returns:
        dict: Dictionary containing the following properties:
            v (float): Absolute velocity.
            v_m (float): Meridional component of absolute velocity.
            v_t (float): Tangential component of absolute velocity.
            alpha (float): Absolute flow angle.
            w (float): Relative velocity magnitude.
            w_m (float): Meridional component of relative velocity.
            w_t (float): Tangential component of relative velocity.
            beta (float): Relative flow angle (in radians).
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

    Args:
        u (float): Blade speed.
        w (float): Relative velocity.
        beta (float): Relative flow angle (in radians).

    Returns:
        dict: Dictionary containing the following properties:
            v (float): Absolute velocity.
            v_m (float): Meridional component of absolute velocity.
            v_t (float): Tangential component of absolute velocity.
            alpha (float): Absolute flow angle (in radians).
            w (float): Relative velocity magnitude.
            w_m (float): Meridional component of relative velocity.
            w_t (float): Tangential component of relative velocity.
            beta (float): Relative flow angle.
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


def evaluate_loss_model(loss_model_input, is_throat=False):
    """
    Evaluate the loss according to both loss correlation and definition.
    Return the loss coefficient, error, and breakdown of losses.

    Args:
        cascade_data (dict): Data for the cascade.

    Returns:
        tuple: A tuple containing the loss coefficient, error, and additional information.
    """
    # Load necessary parameters
    lossmodel = loss_model_input["loss_model"]
    p0rel_in = loss_model_input["flow"]["p0_rel_in"]
    p0rel_out = loss_model_input["flow"]["p0_rel_out"]
    p_out = loss_model_input["flow"]["p_out"]

    # Compute the loss coefficient from its definition
    Y_def = (p0rel_in - p0rel_out) / (p0rel_out - p_out)

    # Compute the loss coefficient from the correlations
    Y, Y_info = lm.loss(loss_model_input, lossmodel, is_throat)

    # Compute loss coefficient error
    Y_err = Y_def - Y

    return Y, Y_err, Y_info


def evaluate_inter_cascade_space(
    h0_exit, v_m_exit, v_t_exit, rho_exit, fluid, r_out, a_out, r_in, a_in
):
    """
    Function that calculates the inlet condition of the next cascade based on the exit state of the previous cascade
    """

    h0_in = h0_exit
    v_t_in = v_t_exit * r_out / r_in
    v_m_in = v_m_exit * a_out / a_in
    v_in = np.sqrt(v_t_in**2 + v_m_in**2)
    alpha_in = math.arctand(v_t_in / v_m_in)
    h_in = h0_in - 0.5 * v_in**2
    rho_in = rho_exit
    stagnation_properties = fluid.compute_properties_meanline(
        CP.DmassHmass_INPUTS, rho_in, h_in
    )
    s_in = stagnation_properties["s"]

    return h0_in, s_in, alpha_in, v_in


def calculate_mach_equation(Ma_crit, Ma_exit, Ma_throat, alpha=-100):
    xi = np.array([Ma_crit, Ma_exit])

    actual_mach = smooth_max(xi, method="boltzmann", alpha=alpha)

    res = Ma_throat - actual_mach

    return res


def calculate_deviation_equation(
    geometry, critical_state, exit_plane, density_correction, deviation_model
):
    # Load cascade geometry
    theta = geometry["theta_out"]
    area = geometry["A_out"]
    opening = geometry["o"]
    pitch = geometry["s"]

    # Load calculated critical condition
    m_crit = critical_state["m"]
    Ma_crit = critical_state["Ma_rel"]

    # Load exit plane
    Ma = exit_plane["Ma_rel"]
    rho = exit_plane["d"]
    w = exit_plane["w"]
    beta = exit_plane["beta"]
    blockage = exit_plane["blockage"]

    if Ma < Ma_crit:
        beta_model = dm.get_subsonic_deviation(Ma, Ma_crit, opening/pitch, deviation_model)
    else:
        beta_model = math.arccosd(m_crit / rho / w / area / blockage * density_correction)

    # Compute error of guessed beta and deviation model
    res = math.cosd(beta_model) - math.cosd(beta)

    return res


def calculate_eta_drop_fractions(results, n_cascades, reference_values):
    """
    Calculate efficiency penalty for each loss component.

    Args:
        results (dict): The data for the cascades.

    Returns:
        pd.DataFrame: A DataFrame containing efficiency drop fractions.
    """
    # Load parameters
    h_out_s = reference_values["h_out_s"]
    h0_in = results["plane"]["h0"].values[0]
    h_out = results["plane"]["h"].values[-1]
    v_out = results["plane"]["v"].values[-1]
    cascade = results["cascade"]

    dhs_sum = cascade["dh_s"].sum()
    dhss = h_out - h_out_s
    a = dhss / dhs_sum

    # Initialize DataFrame
    loss_fractions = pd.DataFrame(columns=["eta_drop_p", "eta_drop_s", "eta_drop_cl"])

    for i in range(n_cascades):
        fractions_temp = cascade.loc[i, "Y_p":"Y_cl"] / cascade.loc[i, "Y_tot"]

        dhs = cascade["dh_s"][i]
        eta_drop = a * dhs / (h0_in - h_out_s)
        loss_fractions_temp = fractions_temp * eta_drop
        loss_fractions.loc[len(loss_fractions)] = loss_fractions_temp.values

    eta_drop_kinetic = 0.5 * v_out**2 / (h0_in - h_out_s)
    results["overall"]["eta_drop_kinetic"] = eta_drop_kinetic

    return loss_fractions


def get_reference_values(BC, fluid):
    # Renamme variables
    p0_in = BC["p0_in"]
    T0_in = BC["T0_in"]
    p_out = BC["p_out"]

    # Compute stagnation properties at inlet
    stagnation_properties = fluid.compute_properties_meanline(
        CP.PT_INPUTS, p0_in, T0_in
    )
    h0_in = stagnation_properties["h"]
    s_in = stagnation_properties["s"]

    # Calculate exit static properties for a isentropic expansion
    isentropic_expansion_properties = fluid.compute_properties_meanline(
        CP.PSmass_INPUTS, p_out, s_in
    )
    h_is = isentropic_expansion_properties["h"]
    d_is = isentropic_expansion_properties["d"]

    # Calculate exit static properties for a isenthalpic process to the given exit pressure
    isenhalpic_process_properties = fluid.compute_properties_meanline(
        CP.HmassP_INPUTS, h0_in, p_out
    )
    s_isenthalpic = isenhalpic_process_properties["s"]

    # Calculate spouting velocity
    v0 = np.sqrt(2 * (h0_in - h_is))

    # Add values to BC
    BC["h0_in"] = h0_in
    BC["s_in"] = s_in

    # Define reference_values
    reference_values = {
        "s_range": s_isenthalpic - s_in,
        "s_min": s_in,
        "v0": v0,
        "h_out_s": h_is,
        "d_out_s": d_is,
        "angle_range": 180,
        "angle_min": -90,
        "delta_ref": 0.011 / 3e5 ** (-1 / 7),
    }

    return reference_values


def calculate_stage_parameters(n_stages, planes):
    """
    Calculate parameters relevant for the whole stage, e.g. reaction.

    Args:
        cascades_data (dict): The data for the cascades.

    Returns:
        None
    """

    h_vec = planes["h"].values

    # Degree of reaction
    R = np.zeros(n_stages)

    for i in range(n_stages):
        h1 = h_vec[i * 6]
        h2 = h_vec[i * 6 + 2]
        h3 = h_vec[i * 6 + 5]
        R[i] = (h2 - h3) / (h1 - h3)

    stages = {"R": R}

    return stages


def get_number_of_stages(n_cascades):
    """
    Calculate the number of stages based on the number of cascades.

    Args:
        results (dict): The data for the cascades.

    Raises:
        Exception: If the number of cascades is not valid.

    Returns:
        None
    """

    if n_cascades == 1:
        # If only one cascade, there are no stages
        n_stages = 0

    elif (n_cascades % 2) == 0:
        # If n_cascades is divisible by two, n_stages = n_cascades/2
        n_stages = int(n_cascades / 2)

    else:
        # If n_cascades is not 1 or divisible by two, it's an invalid configuration
        raise Exception("Invalid number of cascades")

    return n_stages


def add_string_to_keys(input_dict, suffix):
    """
    Add a suffix to each key in the input dictionary.

    Args:
        input_dict (dict): The input dictionary.
        suffix (str): The string to add to each key.

    Returns:
        dict: A new dictionary with modified keys.

    Example:
        >>> input_dict = {'a': 1, 'b': 2, 'c': 3}
        >>> add_string_to_keys(input_dict, '_new')
        {'a_new': 1, 'b_new': 2, 'c_new': 3}
    """
    return {f"{key}{suffix}": value for key, value in input_dict.items()}
