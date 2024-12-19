
import scipy.linalg
import scipy.integrate
import scipy.optimize
import numpy as np
import CoolProp as cp
import matplotlib.pyplot as plt


def evaluate_vaneless_diffuser_1d(
        p0_in,
        T0_in,
        Ma_in,
        alpha_in,
        Cf,
        q_w,
        fluid_name,
        diffuser_geometry,
        number_of_points=None,
        tol=1e-6,
        ):
    """Evaluate one-dimensional flow in an generic annular duct"""

    # Create fluid object
    fluid = cp.AbstractState("HEOS", fluid_name)

    # Compute inlet static state
    p_in, s_in = compute_inlet_static_state(p0_in, T0_in, Ma_in, fluid)
    fluid.update(cp.PSmass_INPUTS, p_in, s_in)
    d_in = fluid.rhomass()
    p_in = fluid.p()
    a_in = fluid.speed_sound()
    v_in = Ma_in*a_in
    v_m_in = v_in*np.cos(alpha_in)
    v_t_in = v_in*np.sin(alpha_in)

    # Rename geometry
    phi = diffuser_geometry["phi"]
    div = diffuser_geometry["div"]
    r_in = diffuser_geometry["r_in"]
    r_out = diffuser_geometry["r_out"]
    b_in = diffuser_geometry["b_in"]

    # Define integration interval
    m_out = r_out - r_in
    
    def odefun(t, y):

        # Rename from ODE terminology to physical variables
        m = t     
        v_m, v_t, d, p, s_gen, theta = y

        # Calculate velocity
        v = np.sqrt(v_t**2 + v_m**2)
        alpha = np.arctan(v_t/v_m)

        # Calculate local radius
        r = r_fun(r_in, phi, m)
        b = b_fun(b_in, div, m)

        # Calculate derivative of area change
        delta = 1e-4
        diff_br = (b_fun(b_in,div,m+delta)*r_fun(r_in,phi,m+delta) - b*r)/delta

        # Calculate derivative of internal energy
        delta = 1e-4
        fluid.update(cp.DmassP_INPUTS, d, p-delta)
        e1 = fluid.umass()
        fluid.update(cp.DmassP_INPUTS, d, p+delta)
        e2 = fluid.umass()
        dedp_d = (e2 - e1)/(2*delta)

        # Calculate thermodynamic state
        fluid.update(cp.DmassP_INPUTS, d, p)
        a = fluid.speed_sound()
        h = fluid.hmass()
        s = fluid.smass()
        T = fluid.T()
        h0 = h + 0.5*v**2

        # Stress at the wall
        # TODO implement function for friction factor
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
                d*v_t**2/r*np.sin(phi) - 2*tau_w/b*np.cos(alpha),
                -d*v_t*v_m/r*np.sin(phi)-2*tau_w/b*np.sin(alpha),
                2*(tau_w*v + q_w)/(b*dedp_d)
            ]
        ) 

        dy = scipy.linalg.solve(M, S)


        # Compute entropy generation rate
        sigma = 2/b*(tau_w*v)
        dy = np.append(dy, sigma/(d*v_m)/T)
        
        # Compute streamline wrapping) angle (only for phi=pi/2)
        dy = np.append(dy, (v_t/v_m)/r)

        # Store data in dictionary
        out = {"v_t" : v_t,
               "v_m" : v_m,
               "v" : v,
               "Ma" : v / a,
               "Ma_m" : v_m / a,
               "Ma_t" : v_t / a,
               "alpha" : alpha,
               "d" : d,
               "p" : p,
               "s" : s,
               "T" : T,
               "s_gen" : s_gen,
               "h" : h,
               "h0" : h0,
               "theta" : theta,
               "r": r,
               "b": b,
               "m": m,
               "radius_ratio": r/r_in,
               "area_ratio": (b*r)/(b_in*r_in),
               "Cp": (p-p_in)/(p0_in-p_in),
               "Re": "",
               }

        return dy, out

    m_eval = np.linspace(0.0, m_out, number_of_points) if number_of_points else None
    ode_sol = scipy.integrate.solve_ivp(
        fun=lambda t,y: odefun(t,y)[0],
        t_span=[0, m_out],
        t_eval=m_eval,
        y0=[v_m_in, v_t_in, d_in, p_in, 0.0, 0.0],
        method = "RK45",
        rtol = tol,
        atol = tol,
    )

    # Postprocess solution
    states = postprocess_ode(ode_sol.t, ode_sol.y, odefun)

    return states, ode_sol


def compute_inlet_static_state(p0, T0, Ma, fluid):
    """
    Calculate the static pressure from stagnation conditions and Mach number.
    
    Parameters
    ----------
    p0 : float
        Stagnation pressure (Pa).
    T0 : float
        Stagnation temperature (K).
    Ma : float
        Mach number.
    fluid : str
        Name of the fluid (compatible with CoolProp).
    
    Returns
    -------
    float
        Static pressure (Pa).    

    """
    # Compute stagnation state properties
    fluid.update(cp.PT_INPUTS, float(p0), float(T0))
    s0 = fluid.smass()
    h0 = fluid.hmass()

    # Define the function for fsolve
    def stagnation_definition_error(p):
        fluid.update(cp.PSmass_INPUTS, float(p), float(s0))
        a = fluid.speed_sound()  # Speed of sound
        h = fluid.hmass()  # Enthalpy
        v = a * Ma  # Velocity
        return h0 - h - v**2 / 2  # Residual for stagnation definition
    
    # Solve for static pressure using fsolve
    p_static = scipy.optimize.fsolve(stagnation_definition_error, p0)[0]
    return p_static, s0



def r_fun(r_in, phi, m):
    """Calculate the radius from the meridional coordinate"""
    return r_in + np.sin(phi)*m

def b_fun(b_in, div, m):
    """Calculate the channel width from the meridional coordinate"""
    return b_in + 2*np.tan(div)*m


def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored (can be None) and the second element
        is a dictionary representing the processed state of the system.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
    """
    # Initialize ode_out as a dictionary
    ode_out = {}
    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(t_i, y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    return ode_out



if __name__ == '__main__':

    # # Create working fluid
    # fluid_name = "air"

    # # Set inlet conditions
    # p0_in = 101325
    # T0_in = 273.15+20
    # Ma_in = 0.75
    # alpha_in = 65*np.pi/180

    # Create working fluid
    fluid_name = "CO2"

    # Set inlet conditions
    p0_in = 80e5
    T0_in = 273.15+50
    Ma_in = 0.75
    alpha_in = 65*np.pi/180


    # Set friction coefficient and wall heat flux
    Cf_array = [0.0, 0.01, 0.02, 0.03]
    q_w = 0
    
    # Define diffuser geometry
    diffuser_geometry = {
        "phi" : 90*np.pi/180,  # Use pi/2 for a radial channel
        "div" : 0*np.pi/180,   # Use 0 for a constant width channel
        "r_in" : 1.0,
        "r_out" : 3.0,
        "b_in" : 0.25
    }

    # Plot the pressure recovery coefficient distribution
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(True)
    ax_1.set_xlabel('Radius ratio')
    ax_1.set_ylabel('Pressure recovery coefficient\n')

    # Plot the Mach number distribution
    fig_2, ax_2 = plt.subplots()
    ax_2.grid(True)
    ax_2.set_xlabel('Radius ratio')
    ax_2.set_ylabel('Mach number\n')

    # Plot streamlines
    number_of_streamlines = 5
    fig_3, ax_3 = plt.subplots()
    ax_3.set_aspect('equal', adjustable='box')
    ax_3.grid(False)
    ax_3.set_xlabel('x coordinate')
    ax_3.set_ylabel('y coordinate\n')
    ax_3.set_title('Diffuser streamlines\n')
    ax_3.axis(1.1 * diffuser_geometry["r_out"] * np.array([-1, 1, -1, 1]))
    theta = np.linspace(0, 2 * np.pi, 100)
    x_in = diffuser_geometry['r_in'] * np.cos(theta)
    y_in = diffuser_geometry['r_in'] * np.sin(theta)
    x_out = diffuser_geometry['r_out'] * np.cos(theta)
    y_out = diffuser_geometry['r_out'] * np.sin(theta)
    ax_3.plot(x_in, y_in, 'k', label=None)  # HandleVisibility='off'
    ax_3.plot(x_out, y_out, 'k', label=None)  # HandleVisibility='off'
    theta = np.linspace(0, 2 * np.pi, number_of_streamlines + 1)




    
    # Compute diffuser performance for different friction factors
    colors = plt.cm.magma(np.linspace(0, 1, len(Cf_array)+1))  # Generate colors
    for i, Cf in enumerate(Cf_array):
    
        # Evaluate diffuser model
        states, ode_sol = evaluate_vaneless_diffuser_1d(
            p0_in,
            T0_in,
            Ma_in,
            alpha_in,
            Cf,
            q_w,
            fluid_name,
            diffuser_geometry,
            number_of_points=200,
        )

        
        # Plot the pressure recovery coefficient distribution
        ax_1.plot(states['radius_ratio'], states['T'], label=f"$C_f = {Cf:0.3f}$", color=colors[i])
        ax_1.legend(loc='lower right')

        # # Plot the Mach number distribution
        # ax_2.plot(states['radius_ratio'], states['Ma'], label=f"$C_f = {Cf:0.3f}$", color=colors[i])
        # ax_2.legend(loc='upper right')

        # # Plot streamlines
        # for j in range(len(theta)):
        #     x = states['r'] * np.cos(states['theta'] + theta[j])
        #     y = states['r'] * np.sin(states['theta'] + theta[j])
        #     if j == 0:
        #         plt.plot(x, y, label=f"$C_f = {Cf:0.3f}$", color=colors[i])
        #     else:
        #         plt.plot(x, y, color=colors[i])



    # Add legend and show plot
    plt.legend()
    plt.show()


    # print(p_in)
    # print(solution.t)
    # print(solution.y)
