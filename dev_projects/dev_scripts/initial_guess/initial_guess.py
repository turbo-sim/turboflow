

import turboflow as tf
from scipy import optimize
import CoolProp.CoolProp as cp
import numpy as np
import time
import os
import latin_hypercube_sampling as lhs
import matplotlib.pyplot as plt

def calculate_enthalpy_residual_1(prop1, scale, h0, Ma, fluid, call, prop2):

    props = fluid.get_props(call, prop1*scale, prop2)

    return  props["h"] - h0 + 0.5*Ma**2*props["speed_sound"]**2 

def get_unkown(prop1, scale, h0, Ma, fluid, call, prop2):

    "call is either cp.DmassP_INPUTS (find Dmass based on a pressure) or cp.PSmass_INPUTS (find p based on a Smass)"

    sol = optimize.root_scalar(calculate_enthalpy_residual_1, args = (scale, h0, Ma, fluid, call, prop2), method = "secant", x0 = prop1)

    return sol.root*scale

def get_heuristic_guess_mach(eta_tt, eta_ts,  mach, boundary_conditions, geometry, fluid, deviation_model):

    p0_first = boundary_conditions["p0_in"]
    T0_first = boundary_conditions["T0_in"]
    p_final = boundary_conditions["p_out"]
    angular_speed = boundary_conditions["omega"]
    alpha_first = boundary_conditions["alpha_in"]

    number_of_cascades = geometry["number_of_cascades"]

    # Initialize fluid object
    fluid = tf.properties.Fluid(boundary_conditions["fluid_name"], exceptions=True)

    # Calculate first stagnation properties
    stag_first = fluid.get_props(cp.PT_INPUTS, p0_first, T0_first)
    h0_first = stag_first["h"]
    s_first = stag_first["s"]
    d0_first = stag_first["d"]

    # Calculate final exit enthalpy for isentropic expansion
    static_is = fluid.get_props(cp.PSmass_INPUTS, p_final, s_first)
    h_final_s = static_is["h"]
    a_final_s = static_is["speed_sound"]

    # Calculate spouting velocity
    v0 = np.sqrt(2*(h0_first-h_final_s))

    # Calculate exit enthalpy
    h0_final = h0_first - eta_ts * (h0_first - h_final_s)
    v_final = np.sqrt(2 * (h0_first - h_final_s - (h0_first - h0_final) / eta_tt))
    h_final = h0_final - 0.5 * v_final**2

    # Calculate exit static state for expansion with guessed efficiency
    static_properties_exit = fluid.get_props(cp.HmassP_INPUTS, h_final, p_final)
    s_final = static_properties_exit["s"]

    # Assume linear entropy distribution
    entropy_distribution = np.linspace(s_first, s_final, number_of_cascades + 1)[1:]

    # Define initial guess dictionary
    initial_guess = {}

    # Initialize inlet calculation
    s_in = s_first
    rothalpy = h0_first
    alpha_in = alpha_first
    d_in = d0_first

    for i in range(number_of_cascades):
        geometry_cascade = {
            key: values[i]
            for key, values in geometry.items()
            if key not in ["number_of_cascades", "number_of_stages"]
        }

        radius_mean_in = geometry_cascade["radius_mean_in"]
        radius_mean_throat = geometry_cascade["radius_mean_throat"]
        radius_mean_out = geometry_cascade["radius_mean_out"]
        A_throat = geometry_cascade["A_throat"]
        A_out = geometry_cascade["A_out"]
        A_in = geometry_cascade["A_in"]

        # Load entropy from assumed entropy distribution
        s_out = entropy_distribution[i]
        ma_out = mach[i]

        # Calculate exit pressure
        blade_speed_out = angular_speed * (i % 2) * radius_mean_out
        h0_rel_out = rothalpy + 0.5*blade_speed_out**2
        p_out = get_unkown(1.0, p0_first, h0_rel_out, ma_out, fluid, cp.PSmass_INPUTS, s_out)

        # Calculate exit state
        static_out = fluid.get_props(cp.PSmass_INPUTS, p_out, s_out)
        h_out = static_out["h"]
        a_out = static_out["speed_sound"]
        d_out = static_out["d"]
        gamma_out = static_out["gamma"]

        # Calculate exit velocity
        w_out = np.sqrt(2*(h0_rel_out - h_out))
        
        # Calculate critical mach
        static_props_is = fluid.get_props(cp.PSmass_INPUTS, p_out, s_in)
        h_out_s = static_props_is["h"]
        eta = (h0_rel_out-h_out)/(h0_rel_out-h_out_s)
        ma_crit = tf.get_mach_crit(gamma_out, eta)

        # Calculate exit flow angle 
        beta_out = (-1)**i*tf.get_subsonic_deviation(
                ma_out, ma_crit, {"A_throat" : A_throat, "A_out" : A_out}, deviation_model
            )
        
        # Calculate mass flow rate
        mass_flow = d_out * w_out*tf.cosd(beta_out) * A_out
        
         # Calculate critical state
        w_throat_crit = a_out * ma_crit
        h_throat_crit = h0_rel_out - 0.5 * w_throat_crit**2
        s_throat_crit = s_out
        static_state_throat_crit = fluid.get_props(cp.HmassSmass_INPUTS, h_throat_crit, s_throat_crit)
        rho_throat_crit = static_state_throat_crit["d"]
        m_crit = w_throat_crit * rho_throat_crit * A_throat
        w_m_in_crit = m_crit / d_in / A_in
        v_in_crit = w_m_in_crit/tf.cosd(alpha_in)

        # Store initial guess
        index = f"_{i+1}"
        initial_guess.update(
            {
                "w_out" + index: w_out,
                "s_out" + index: s_out,
                "beta_out"
                + index: (-1)**i * tf.arccosd(A_throat / A_out),
                "v_crit_in" + index: v_in_crit,
                "beta_crit_throat"
                + index: (-1)**i * tf.arccosd(A_throat / A_out),
                "w_crit_throat" + index: w_throat_crit,
                "s_crit_throat" + index: s_throat_crit,
            }
        )

        # Update variables for next cascade
        if i != (number_of_cascades - 1):
            A_next = geometry["A_in"][i + 1]
            radius_mean_next = geometry["radius_mean_in"][i + 1]
            velocity_triangle_out = tf.evaluate_velocity_triangle_out(blade_speed_out, w_out, beta_out)
            v_m_in = velocity_triangle_out["v_m"] * A_out / A_next
            v_t_in = velocity_triangle_out["v_t"] * radius_mean_out / radius_mean_next
            v_in = np.sqrt(v_m_in**2 + v_t_in**2)
            alpha_in = tf.arctand(v_t_in / v_m_in)
            blade_speed_in = angular_speed * ((i+1) % 2) * radius_mean_next
            velocity_triangle_in = tf.evaluate_velocity_triangle_in(blade_speed_in, v_in, alpha_in)
            h0_in = h_out + 0.5*velocity_triangle_out["v"]**2
            h_in = h0_in - 0.5 * v_in**2
            rothalpy = h_in + 0.5*velocity_triangle_in["w"]**2 - 0.5*blade_speed_in**2
            s_in = s_out
            static_in = fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
            d_in = static_in["d"]

    # Calculate inlet velocity from
    initial_guess["v_in"] = mass_flow / (d0_first * geometry["A_in"][0] * tf.cosd(alpha_first))

    return initial_guess


def get_heuristic_guess_enthalpy_distribution(
        eta_tt, eta_ts, Ma_crit, boundary_conditions, geometry, fluid,
    ):
        """
        Compute the heuristic initial guess for the performance analysis based on the given parameters.

        This function calculates the heuristic initial guess based on the provided enthalpy loss fractions for each cascade,
        total-to-static and total-to-total efficiencies, and critical Mach number.

        Parameters
        ----------
        enthalpy_loss_fractions : array-like
            Enthalpy loss fractions for each cascade.
        eta_tt : float
            Total-to-total efficiency.
        eta_ts : float
            Total-to-static efficiency.
        Ma_crit : float
            Critical Mach number.

        Returns
        -------
        dict
            Heuristic initial guess for the performance analysis.

        """


        # Rename variables
        number_of_cascades = geometry["number_of_cascades"]
        p0_in = boundary_conditions["p0_in"]
        T0_in = boundary_conditions["T0_in"]
        alpha_in = boundary_conditions["alpha_in"]
        angular_speed = boundary_conditions["omega"]
        p_out = boundary_conditions["p_out"]


        # Calculate inlet stagnation state
        stagnation_properties_in = fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        h0_in = stagnation_properties_in["h"]
        s_in = stagnation_properties_in["s"]
        rho0_in = stagnation_properties_in["d"]

        # Calculate for isentropic expansion
        isentropic_expansion = fluid.get_props(cp.PSmass_INPUTS, p_out, s_in)
        h_out_s = isentropic_expansion["h"]

        # Calculate exit enthalpy
        h0_out = h0_in - eta_ts * (h0_in - h_out_s)
        v_out = np.sqrt(2 * (h0_in - h_out_s - (h0_in - h0_out) / eta_tt))
        h_out = h0_out - 0.5 * v_out**2

        # Calculate exit static state for expansion with guessed efficiency
        static_properties_exit = fluid.get_props(cp.HmassP_INPUTS, h_out, p_out)
        s_out = static_properties_exit["s"]

        # Define entropy distribution
        entropy_distribution = np.linspace(s_in, s_out, number_of_cascades + 1)[1:]

        # Define enthalpy distribution
        enthalpy_distribution = np.linspace(h0_in, h_out, number_of_cascades + 1)[1:]

        # Assums h0_in approx h_in for first inlet
        h_in = h0_in

        # Define initial guess dictionary
        initial_guess = {}

        for i in range(number_of_cascades):
            geometry_cascade = {
                key: values[i]
                for key, values in geometry.items()
                if key not in ["number_of_cascades", "number_of_stages"]
            }

            # Load enthalpy from initial guess
            h_out = enthalpy_distribution[i]

            # Load entropy from assumed entropy distribution
            s_out = entropy_distribution[i]

            # Rename necessary geometry
            theta_in = geometry_cascade["leading_edge_angle"]
            theta_out = geometry_cascade["gauging_angle"]
            A_out = geometry_cascade["A_out"]
            A_throat = geometry_cascade["A_throat"]
            A_in = geometry_cascade["A_in"]
            radius_mean_in = geometry_cascade["radius_mean_in"]
            radius_mean_throat = geometry_cascade["radius_mean_throat"]
            radius_mean_out = geometry_cascade["radius_mean_out"]
            
            # Calculate rothalpy at inlet of cascade
            blade_speed_in = angular_speed * (i % 2) * radius_mean_in
            if i == 0:
                h0_rel_in = h_in
                m_temp = rho0_in * A_in * tf.cosd(alpha_in)
            else:
                v_in = np.sqrt(2 * (h0_in - h_in))
                velocity_triangle_in = tf.evaluate_velocity_triangle_in(
                    blade_speed_in, v_in, alpha_in
                )
                w_in = velocity_triangle_in["w"]
                h0_rel_in = h_in + 0.5 * w_in**2

            rothalpy = h0_rel_in - 0.5 * blade_speed_in**2

            # Calculate static state at cascade inlet
            static_state_in = fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
            rho_in = static_state_in["d"]

            # Calculate exit velocity from rothalpy and enthalpy distirbution
            blade_speed_out = angular_speed * (i % 2) * radius_mean_out
            h0_rel_out = rothalpy + 0.5 * blade_speed_out**2
            w_out = np.sqrt(2 * (h0_rel_out - h_out))
            velocity_triangle_out = tf.evaluate_velocity_triangle_out(
                blade_speed_out, w_out, theta_out
            )
            v_t_out = velocity_triangle_out["v_t"]
            v_m_out = velocity_triangle_out["v_m"]
            v_out = velocity_triangle_out["v"]
            h0_out = h_out + 0.5 * v_out**2

            # Calculate static state at cascade exit
            static_state_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
            a_out = static_state_out["a"]
            rho_out = static_state_out["d"]

            # Calculate mass flow rate
            mass_flow = rho_out * v_m_out * A_out

            # Calculate throat velocity depending on subsonic or supersonic conditions
            if w_out < a_out * Ma_crit:
                w_throat = w_out
                s_throat = s_out
            else:
                w_throat = a_out * Ma_crit
                blade_speed_throat = angular_speed * (i % 2) * radius_mean_throat
                h0_rel_throat = rothalpy + 0.5 * blade_speed_throat**2
                h_throat = h0_rel_throat - 0.5 * w_throat**2
                rho_throat = rho_out
                static_state_throat = fluid.get_props(
                    cp.DmassHmass_INPUTS, rho_throat, h_throat
                )
                s_throat = static_state_throat["s"]

            # Calculate critical state
            w_throat_crit = a_out * Ma_crit
            h_throat_crit = (
                h0_rel_in - 0.5 * w_throat_crit**2
            )  # FIXME: h0_rel_in works less good?
            static_state_throat_crit = fluid.get_props(
                cp.HmassSmass_INPUTS, h_throat_crit, s_throat
            )
            rho_throat_crit = static_state_throat_crit["d"]
            m_crit = w_throat_crit * tf.cosd(theta_out) * rho_throat_crit * A_throat
            w_m_in_crit = m_crit / rho_in / A_in
            w_in_crit = w_m_in_crit / tf.cosd(
                theta_in
            )  # XXX Works better with metal angle than inlet flow angle?
            velocity_triangle_crit_in = tf.evaluate_velocity_triangle_out(
                blade_speed_in, w_in_crit, theta_in
            )
            v_in_crit = velocity_triangle_crit_in["v"]

            rho_out_crit = rho_throat_crit
            w_out_crit = m_crit / (rho_out_crit * A_out * tf.cosd(theta_out))

            # Store initial guess
            index = f"_{i+1}"
            initial_guess.update(
                {
                    "w_out" + index: w_out,
                    "s_out" + index: s_out,
                    "beta_out"
                    + index: np.sign(theta_out) * tf.arccosd(A_throat / A_out),
                    "v_crit_in" + index: v_in_crit,
                    "beta_crit_throat"
                    + index: np.sign(theta_out) * tf.arccosd(A_throat / A_out),
                    "w_crit_throat" + index: w_throat_crit,
                    "s_crit_throat" + index: s_throat,
                }
            )

            # Update variables for next cascade
            if i != (number_of_cascades - 1):
                A_next = geometry["A_in"][i + 1]
                radius_mean_next = geometry["radius_mean_in"][i + 1]
                v_m_in = v_m_out * A_out / A_next
                v_t_in = v_t_out * radius_mean_out / radius_mean_next
                v_in = np.sqrt(v_m_in**2 + v_t_in**2)
                alpha_in = tf.arctand(v_t_in / v_m_in)
                h0_in = h0_out
                h_in = h0_in - 0.5 * v_in**2
                s_in = s_out

        # Calculate inlet velocity from
        initial_guess["v_in"] = mass_flow / m_temp

        return initial_guess

def get_heuristic_guess_improved(
        eta_tt, eta_ts, boundary_conditions, geometry, fluid,
    ):
        """
        Compute the heuristic initial guess for the performance analysis based on the given parameters.

        This function calculates the heuristic initial guess based on the provided enthalpy loss fractions for each cascade,
        total-to-static and total-to-total efficiencies, and critical Mach number.

        Parameters
        ----------
        enthalpy_loss_fractions : array-like
            Enthalpy loss fractions for each cascade.
        eta_tt : float
            Total-to-total efficiency.
        eta_ts : float
            Total-to-static efficiency.
        Ma_crit : float
            Critical Mach number.

        Returns
        -------
        dict
            Heuristic initial guess for the performance analysis.

        """


        # Rename variables
        number_of_cascades = geometry["number_of_cascades"]
        p0_first = boundary_conditions["p0_in"]
        T0_first = boundary_conditions["T0_in"]
        alpha_first = boundary_conditions["alpha_in"]
        angular_speed = boundary_conditions["omega"]
        p_final = boundary_conditions["p_out"]


        # Calculate inlet stagnation state
        stagnation_properties_in = fluid.get_props(cp.PT_INPUTS, p0_first, T0_first)
        h0_first = stagnation_properties_in["h"]
        s_first = stagnation_properties_in["s"]
        rho0_first = stagnation_properties_in["d"]

        # Calculate for isentropic expansion
        isentropic_expansion = fluid.get_props(cp.PSmass_INPUTS, p_final, s_first)
        h_final_s = isentropic_expansion["h"]

        # Calculate exit enthalpy
        h0_final = h0_first - eta_ts * (h0_first - h_final_s)
        v_final = np.sqrt(2 * (h0_first - h_final_s - (h0_first - h0_final) / eta_tt))
        h_final = h0_final - 0.5 * v_final**2

        # Calculate exit static state for expansion with guessed efficiency
        static_properties_exit = fluid.get_props(cp.HmassP_INPUTS, h_final, p_final)
        s_final = static_properties_exit["s"]

        # Define entropy distribution
        entropy_distribution = np.linspace(s_first, s_final, number_of_cascades + 1)[1:]

        # Define enthalpy distribution
        enthalpy_distribution = np.linspace(h0_first, h_final, number_of_cascades + 1)[1:]

        # Assums h0_in approx h_in for first inlet
        h_in = h0_first
        s_in = s_first
        rothalpy = h0_first
        alpha_in = alpha_first

        # Define initial guess dictionary
        initial_guess = {}

        for i in range(number_of_cascades):
            geometry_cascade = {
                key: values[i]
                for key, values in geometry.items()
                if key not in ["number_of_cascades", "number_of_stages"]
            }

            # Load enthalpy from initial guess
            h_out = enthalpy_distribution[i]

            # Load entropy from assumed entropy distribution
            s_out = entropy_distribution[i]

            # Rename necessary geometry
            theta_in = geometry_cascade["leading_edge_angle"]
            theta_out = geometry_cascade["gauging_angle"]
            A_out = geometry_cascade["A_out"]
            A_throat = geometry_cascade["A_throat"]
            A_in = geometry_cascade["A_in"]
            radius_mean_in = geometry_cascade["radius_mean_in"]
            radius_mean_throat = geometry_cascade["radius_mean_throat"]
            radius_mean_out = geometry_cascade["radius_mean_out"]
            
            # Calculate rothalpy at inlet of cascade
            blade_speed_in = angular_speed * (i % 2) * radius_mean_in
            if i == 0:
                h0_rel_in = h_in
                m_temp = rho0_first * A_in * tf.cosd(alpha_in)
            else:
                v_in = np.sqrt(2 * (h0_in - h_in))
                velocity_triangle_in = tf.evaluate_velocity_triangle_in(blade_speed_in, v_in, alpha_in)
                w_in = velocity_triangle_in["w"]
                h0_rel_in = h_in + 0.5 * w_in**2

            rothalpy = h0_rel_in - 0.5 * blade_speed_in**2

            # Calculate static state at cascade inlet
            static_state_in = fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
            rho_in = static_state_in["d"]

            # Calculate static state at cascade exit
            static_state_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
            a_out = static_state_out["a"]
            rho_out = static_state_out["d"]
            gamma_out = static_state_out["gamma"]

            # Calculate exit velocity from rothalpy and enthalpy distirbution
            blade_speed_out = angular_speed * (i % 2) * radius_mean_out
            h0_rel_out = rothalpy + 0.5 * blade_speed_out**2
            w_out = np.sqrt(2 * (h0_rel_out - h_out))
            ma_crit = 1.0
            beta_out = (-1)**i*tf.get_subsonic_deviation(
                w_out/a_out, ma_crit, {"A_throat" : A_throat, "A_out" : A_out}, deviation_model
            )
            velocity_triangle_out = tf.evaluate_velocity_triangle_out(
                blade_speed_out, w_out, theta_out
            )
            v_t_out = velocity_triangle_out["v_t"]
            v_m_out = velocity_triangle_out["v_m"]
            v_out = velocity_triangle_out["v"]
            h0_out = h_out + 0.5 * v_out**2

            # Calculate mass flow rate
            mass_flow = rho_out * v_m_out * A_out

            # Calculate critical state
            w_throat_crit = a_out * ma_crit
            h_throat_crit = h0_rel_out - 0.5 * w_throat_crit**2
            s_throat_crit = s_out
            static_state_throat_crit = fluid.get_props(cp.HmassSmass_INPUTS, h_throat_crit, s_throat_crit)
            rho_throat_crit = static_state_throat_crit["d"]
            m_crit = w_throat_crit * rho_throat_crit * A_throat
            w_m_in_crit = m_crit / rho_in / A_in
            w_in_crit = w_m_in_crit / tf.cosd(alpha_in) 
            blade_speed_in = angular_speed * (i % 2) * radius_mean_in
            velocity_triangle_crit_in = tf.evaluate_velocity_triangle_out(blade_speed_in, w_in_crit, theta_in)
            v_in_crit = velocity_triangle_crit_in["v"]

            # Store initial guess
            index = f"_{i+1}"
            initial_guess.update(
                {
                    "w_out" + index: w_out,
                    "s_out" + index: s_out,
                    "beta_out"
                    + index: (-1)**i * tf.arccosd(A_throat / A_out),
                    "v_crit_in" + index: v_in_crit,
                    "beta_crit_throat"
                    + index: (-1)**i * tf.arccosd(A_throat / A_out),
                    "w_crit_throat" + index: w_throat_crit,
                    "s_crit_throat" + index: s_throat_crit,
                }
            )

            # Update variables for next cascade
            if i != (number_of_cascades - 1):
                A_next = geometry["A_in"][i + 1]
                radius_mean_next = geometry["radius_mean_in"][i + 1]
                v_m_in = v_m_out * A_out / A_next
                v_t_in = v_t_out * radius_mean_out / radius_mean_next
                v_in = np.sqrt(v_m_in**2 + v_t_in**2)
                alpha_in = tf.arctand(v_t_in / v_m_in)
                h0_in = h0_out
                h_in = h0_in - 0.5 * v_in**2
                s_in = s_out

        # Calculate inlet velocity from
        initial_guess["v_in"] = mass_flow / m_temp

        return initial_guess

def get_heuristic_guess_first_mach(eta_tt, ma_first, ma_final, boundary_conditions, geometry, fluid, deviation_model):

    """
    Actual state
    - Guess first mach to get mass flow rate
    - Guess mach final to get s_final
    - 1D equation for each cascade to calculate exit velocity
    - Satisfies conservatino of mass
    - Exit pressure wrong

    Critical state
    - Negligable change in speed of sound from out condition
    - Negligable change in entropy from out condition
    """

    p0_first = boundary_conditions["p0_in"]
    T0_first = boundary_conditions["T0_in"]
    p_final = boundary_conditions["p_out"]
    angular_speed = boundary_conditions["omega"]
    alpha_first = boundary_conditions["alpha_in"]

    number_of_cascades = geometry["number_of_cascades"]

    # Initialize fluid object
    fluid = tf.properties.Fluid(boundary_conditions["fluid_name"], exceptions=True)

    # Calculate first stagnation properties
    stag_first = fluid.get_props(cp.PT_INPUTS, p0_first, T0_first)
    h0_first = stag_first["h"]
    s_first = stag_first["s"]
    d0_first = stag_first["d"]

    # Calculate final exit enthalpy for isentropic expansion
    static_is = fluid.get_props(cp.PSmass_INPUTS, p_final, s_first)
    h_final_s = static_is["h"]
    a_final_s = static_is["speed_sound"]

    # Calculate spouting velocity
    v0 = np.sqrt(2*(h0_first-h_final_s))

    # Calculate final total enthalpy
    h0_final_s = h_final_s + 0.5*ma_final**2*a_final_s**2
    h0_final = h0_first - eta_tt * (h0_first - h0_final_s)

    # Calculate final density
    d_final = get_unkown(1.0, d0_first, h0_final, ma_final, fluid, cp.DmassP_INPUTS, p_final)

    # Calculate final state
    static_final = fluid.get_props(cp.DmassP_INPUTS, d_final, p_final)
    s_final = static_final["s"]
    h_final = static_final["h"]
    a_final = static_final["speed_sound"]

    # Calculate first state
    p_first = get_unkown(1.0, p0_first, h0_first, ma_first, fluid, cp.PSmass_INPUTS, s_first)
    static_first = fluid.get_props(cp.PSmass_INPUTS, p_first, s_first)
    h_first = static_first["h"]
    d_first = static_first["d"]

    # Calculate first inlet velocity
    v_first = np.sqrt(2*(h0_first-h_first))

    # Check mass flow rate
    mass_flow_rate = v_first*tf.cosd(alpha_first)*d_first*geometry["A_in"][0]

    # Assume linear entropy distribution
    entropy_distribution = np.linspace(s_first, s_final, number_of_cascades + 1)[1:]

    # Define initial guess dictionary
    initial_guess = {"v_in" : v_first}

    # Initialize inlet calculation
    s_in = s_first
    rothalpy = h0_first
    h0_in = h0_first

    for i in range(number_of_cascades):

        s_out = entropy_distribution[i]

        # Rename geometrical variables
        geometry_cascade = {
                    key: values[i]
                    for key, values in geometry.items()
                    if key not in ["number_of_cascades", "number_of_stages"]
                }
        
        A_out = geometry_cascade["A_out"]
        A_throat = geometry_cascade["A_throat"]
        A_in = geometry_cascade["A_in"]
        radius_mean_in = geometry_cascade["radius_mean_in"]
        radius_mean_throat = geometry_cascade["radius_mean_throat"]
        radius_mean_out = geometry_cascade["radius_mean_out"]

        # Calculate exit velocity and static enthalpy
        blade_speed_in = angular_speed * (i % 2) * radius_mean_in
        blade_speed_out = angular_speed * (i % 2) * radius_mean_out
        h0_rel_out = rothalpy + 0.5*blade_speed_out**2
        w_out = get_velocity(0.5, v0, mass_flow_rate, s_out, h0_rel_out, A_throat, A_out, fluid)
        h_out = h0_rel_out - 0.5*w_out**2

        # Calculate exit properties
        static_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
        a_out = static_out["speed_sound"]
        gamma_out = static_out["gamma"]
        p_out = static_out["p"]
        d_out = static_out["d"]
        ma_out = w_out/a_out

        # Calculate critical mach
        static_props_is = fluid.get_props(cp.PSmass_INPUTS, p_out, s_in)
        h_out_s = static_props_is["h"]
        eta = (h0_rel_out-h_out)/(h0_rel_out-h_out_s)
        ma_crit = tf.get_mach_crit(gamma_out, eta)

        # Calculate critical state
        a_out = static_out["speed_sound"]
        w_throat_crit = ma_crit*a_out
        blade_speed_throat = angular_speed * (i % 2) * radius_mean_throat
        h0_rel_throat = rothalpy + 0.5 * blade_speed_throat**2
        h_throat_crit = h0_rel_out - 0.5*w_throat_crit**2
        s_throat_crit = s_out
        static_crit = fluid.get_props(cp.HmassSmass_INPUTS, h_throat_crit, s_throat_crit)
        d_throat_crit = static_crit["d"]
        m_crit = d_throat_crit*w_throat_crit*A_throat 
        v_in_crit = get_velocity(0.5, v0, m_crit, s_in, h0_in, A_throat, A_out, fluid)

        # Calculate deviation
        beta_out = (-1)**i*tf.get_subsonic_deviation(
                ma_out, ma_crit, {"A_throat" : A_throat, "A_out" : A_out}, deviation_model
            )

        # Store initial guess
        index = f"_{i+1}"
        initial_guess.update(
                {
                    "w_out" + index: w_out,
                    "s_out" + index: s_out,
                    "beta_out" + index : beta_out,
                    "v_crit_in" + index : v_in_crit,
                    "w_crit_throat" + index: w_throat_crit,
                    "s_crit_throat" + index: s_throat_crit,
                }
            )
        
        # # Calculate cascade interspace
        if i is not number_of_cascades-1:

            # Calculate exit velocity triangle
            velocity_triangle_out = tf.evaluate_velocity_triangle_out(blade_speed_out, w_out, beta_out)

            # Assume no heat transfer
            h0_in = h_out + 0.5*velocity_triangle_out["v"]**2

            # Assume no friction (angular momentum is conserved)
            v_t_in = velocity_triangle_out["v_t"] * radius_mean_out / geometry["radius_mean_in"][i+1]
            # Assume density variation is negligible
            v_m_in = velocity_triangle_out["v_m"] * A_out / geometry["A_in"][i+1]

            # Compute velocity vector
            v_in = np.sqrt(v_t_in**2 + v_m_in**2)
            alpha_in = tf.arctand(v_t_in / v_m_in)

            # Compute thermodynamic state
            h_in = h0_in - 0.5 * v_in**2
            d_in = d_out
            static_props = fluid.get_props(cp.DmassHmass_INPUTS, d_in, h_in)
            s_in = static_props["s"]

            # Calculate velocity triangles in
            blade_speed_in = angular_speed * ((i+1) % 2) * geometry["radius_mean_in"][i+1]
            velocity_triangle_in = tf.evaluate_velocity_triangle_in(blade_speed_in, v_in, alpha_in)
            h0_rel_in = h_in + 0.5*velocity_triangle_in["w"]**2 
            rothalpy = h0_rel_in - 0.5 *blade_speed_in**2

    return initial_guess

def get_simple_guess(eta_tt, eta_ts, boundary_conditions, geometry, fluid):

    p0_first = boundary_conditions["p0_in"]
    T0_first = boundary_conditions["T0_in"]
    p_final = boundary_conditions["p_out"]
    angular_speed = boundary_conditions["omega"]
    alpha_first = boundary_conditions["alpha_in"]

    number_of_cascades = geometry["number_of_cascades"]

    # Initialize fluid object
    fluid = tf.properties.Fluid(boundary_conditions["fluid_name"], exceptions=True)

    # Calculate first stagnation properties
    stag_first = fluid.get_props(cp.PT_INPUTS, p0_first, T0_first)
    h0_first = stag_first["h"]
    s_first = stag_first["s"]
    d0_first = stag_first["d"]

    # Calculate final exit enthalpy for isentropic expansion
    static_is = fluid.get_props(cp.PSmass_INPUTS, p_final, s_first)
    h_final_s = static_is["h"]
    a_final_s = static_is["speed_sound"]

    # Calculate spouting velocity
    v0 = np.sqrt(2*(h0_first-h_final_s))

    # Calculate exit enthalpy
    h0_final = h0_first - eta_ts * (h0_first - h_final_s)
    v_final = np.sqrt(2 * (h0_first - h_final_s - (h0_first - h0_final) / eta_tt))
    h_final = h0_final - 0.5 * v_final**2

    # Calculate exit static state for expansion with guessed efficiency
    static_properties_exit = fluid.get_props(cp.HmassP_INPUTS, h_final, p_final)
    s_final = static_properties_exit["s"]

    # Assume linear entropy distribution
    entropy_distribution = np.linspace(s_first, s_final, number_of_cascades + 1)[1:]

    # initialize initial guess
    initial_guess = {}

    for i in range(number_of_cascades):

        s_out = entropy_distribution[i]
        w_out = 0.6*v0
        v_in = 0.2*v0

        # Store initial guess
        index = f"_{i+1}"
        initial_guess.update(
                {
                    "w_out" + index: w_out,
                    "s_out" + index: s_out,
                    "beta_out" + index :  (-1)**i * tf.arccosd(geometry["A_throat"][i] / geometry["A_throat"][i]),
                    "v_crit_in" + index : v_in,
                    "w_crit_throat" + index: w_out,
                    "s_crit_throat" + index: s_out,
                }
            )
        
    initial_guess["v_in"] = 0.1*v0
    return initial_guess

        
def get_residual(w, v0, m, s, h0_rel, A_throat, A_out, fluid):

    w = w*v0
    h = h0_rel - 0.5*w**2

    # Calculate exit properties
    static = fluid.get_props(cp.HmassSmass_INPUTS, h, s)
    a = static["speed_sound"]
    gamma = static["gamma"]
    p = static["p"]
    d = static["d"]
    ma = w/a

    # Calculate critical mach
    static_props_is = fluid.get_props(cp.PSmass_INPUTS, p, s)
    h_s = static_props_is["h"]
    eta = (h0_rel-h)/(h0_rel-h_s)
    ma_crit = tf.get_mach_crit(gamma, eta)

    # Calculate deviation
    beta_out = tf.get_subsonic_deviation(
            ma, ma_crit, {"A_throat" : A_throat, "A_out" : A_out}, deviation_model
        )
    
    # Check mass flow rate
    m_calc = w*tf.cosd(beta_out)*d*A_out

    return (m-m_calc)/m

def get_velocity(x0, v0, m, s, h0_rel, A_throat, A_out, fluid):

    sol = optimize.root_scalar(get_residual, args = (v0, m, s, h0_rel, A_throat, A_out, fluid), method = "secant", x0 = x0)

    # print(sol)

    return sol.root*v0

def evaluate_residual(initial_guess, problem):

    x = problem.scale_values(initial_guess)
    problem.keys = x.keys()
    x0 = np.array(list(x.values()))
    residual = problem.residual(x0)

    return np.linalg.norm(residual)


        
if __name__ == "__main__":

    # Simulation options
    deviation_model = 'aungier'

    # geometry
    geometry = {"cascade_type": np.array(["stator", "rotor"]),
                "radius_hub_in": np.array([0.084785, 0.084785]),
                "radius_hub_out": np.array([0.084785, 0.081875]),
                "radius_tip_in": np.array([0.118415, 0.118415]),
                "radius_tip_out": np.array([0.118415, 0.121325]),
                "pitch": np.array([1.8294e-2, 1.524e-2]),
                "chord": np.array([2.616e-2, 2.606e-2]),
                "stagger_angle": np.array([+43.03, -31.05]),
                "opening": np.array([0.747503242e-2, 0.735223377e-2]),
                "leading_edge_angle" : np.array([0.00, 29.60]),
                "leading_edge_wedge_angle" : np.array([50.00, 50.00]),
                "leading_edge_diameter" : np.array([2*0.127e-2, 2*0.081e-2]),
                "trailing_edge_thickness" : np.array([0.050e-2, 0.050e-2]),
                "maximum_thickness" : np.array([0.505e-2, 0.447e-2]),
                "tip_clearance": np.array([0.00, 0.030e-2]),
                "throat_location_fraction": np.array([1, 1]),}
    tf.validate_turbine_geometry(geometry)
    
    # Load config_files
    CONFIG_FILE_1972 = os.path.abspath("kofskey1972_1stage.yaml")
    CONFIG_FILE_1974 = os.path.abspath("kofskey1974.yaml")
    config_1972 = tf.load_config(CONFIG_FILE_1972, print_summary=False)
    config_1974 = tf.load_config(CONFIG_FILE_1974, print_summary=False)

    # Heuristic guess 
    guess_1 = {"eta_tt" : 0.9,
                "ma_first" : 0.227,
                "ma_final" : 0.8}

    guess_2 = {"eta_tt" : 0.9,
                "eta_ts" : 0.8,
                "Ma_crit" : 0.95}
    
    guess_1_bounds = {"eta_tt" : (0.6, 1.0),
                      "ma_first" : (0.0, 0.3),
                      "ma_final" : (0.5, 1.5)}
    
    guess_2_bounds = {"eta_tt" : (0.5, 0.95),
                      "eta_ts" : (0.4, 0.95),
                      "Ma_crit" : (0.9, 1.0)}
    
    guess_3_bounds = {"eta_tt" : (0.6, 1.0),
                "eta_ts" : (0.5, 0.95),
                "mach_1" :(0.5, 1.2),
                "mach_2" :(0.5, 1.2)}
    
    guess_4_bounds = {"eta_tt" : (0.5, 0.95),
                      "eta_ts" : (0.4, 0.95)}

    # Define case
    case = 1974
    if case == 1972:
        config = config_1972
    elif case == 1974:
        config = config_1974

    N = 100

    # Initialize problem
    problem = tf.CascadesNonlinearSystemProblem(config)
    problem.update_boundary_conditions(config["operation_points"])

    # Initialize structure for storing data
    results = {"method_1" : {"norm_residual" : np.array([]), "time" : np.array([]), "failures" : 0},
               "method_2" : {"norm_residual" : np.array([]), "time" : np.array([]),"failures" : 0}}


    # Generate guesses
    sample_1 = lhs.latin_hypercube_sampling(list(guess_1_bounds.values()), N)
    sample_2 = lhs.latin_hypercube_sampling(list(guess_2_bounds.values()), N)
    sample_3 = lhs.latin_hypercube_sampling(list(guess_3_bounds.values()), N)
    sample_4 = lhs.latin_hypercube_sampling(list(guess_4_bounds.values()), N)
    for i in range(N):

        try:
            start_time = time.time()
            # initial_guess = get_heuristic_guess_first_mach(sample_1[i, 0], sample_1[i, 1], sample_1[i, 2], problem.boundary_conditions, problem.geometry, problem.fluid, deviation_model)
            # initial_guess = get_heuristic_guess_mach(sample_3[i, 0], sample_3[i, 1], [sample_3[i, 2],sample_3[i, 3]], problem.boundary_conditions, problem.geometry, problem.fluid, deviation_model)
            # initial_guess = get_heuristic_guess_improved(sample_4[i, 0], sample_4[i, 1], problem.boundary_conditions, problem.geometry, problem.fluid)
            initial_guess = get_simple_guess(sample_4[i, 0], sample_4[i, 1], problem.boundary_conditions, problem.geometry, problem.fluid)
            end_time = time.time() - start_time
            results["method_1"]["time"] = np.append(results["method_1"]["time"], end_time)
            norm = evaluate_residual(initial_guess, problem)
            results["method_1"]["norm_residual"] = np.append(results["method_1"]["norm_residual"], norm)
        except:
            results["method_1"]["failures"] += 1

        try:
            start_time = time.time()
            initial_guess = get_heuristic_guess_enthalpy_distribution(sample_2[i, 0], sample_2[i, 1], sample_2[i, 2], problem.boundary_conditions, problem.geometry, problem.fluid)
            end_time = time.time() - start_time
            results["method_2"]["time"] = np.append(results["method_2"]["time"], end_time)
            norm = evaluate_residual(initial_guess, problem)
            results["method_2"]["norm_residual"] = np.append(results["method_2"]["norm_residual"], norm)

        except:
            results["method_2"]["failures"] += 1


    # filter
    # results["method_1"]["norm_residual"] = np.array([val for val in results["method_1"]["norm_residual"] if val < 10])
    print(results["method_1"]["norm_residual"])
    # Print results
    print(f"Number of failures for guess mach method: {results['method_1']['failures']}")
    print(f"Number of failures for guess enthalpy distribution method: {results['method_2']['failures']}")
    print(f"Average norm of resiudal method_1: {np.average(results['method_1']['norm_residual'])}")
    print(f"Average norm of resiudal method_2: {np.average(results['method_2']['norm_residual'])}")
    fig1, ax1 = plt.subplots()
    ax1.scatter(range(len(results["method_1"]["norm_residual"])), results["method_1"]["norm_residual"], color = 'b', label = 'guess mach')
    ax1.scatter(range(len(results["method_2"]["norm_residual"])), results["method_2"]["norm_residual"], color = 'r', marker = 'x', label = 'enthalpy distribution')
    ax1.legend()
    ax1.set_title("Norm of residuals")
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(range(len(results["method_1"]["time"])), results["method_1"]["time"], color = 'b', label = 'guess mach')
    ax2.scatter(range(len(results["method_2"]["time"])), results["method_2"]["time"], color = 'r', marker = 'x', label = 'enthalpy distribution')
    ax2.legend()
    ax2.set_title("Time")

    plt.show()



