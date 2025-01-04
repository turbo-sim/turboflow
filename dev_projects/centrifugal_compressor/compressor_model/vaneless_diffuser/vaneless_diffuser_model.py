
import numpy as np
import CoolProp as cp
import scipy.linalg
import scipy.integrate


def radial_vaneless_diffuser_1d(
        fluid,
        Cf,
        q_w,
        inlet_state,
        diffuser_geometry,
        ):
    
    # Rename inlet state properties
    v_m_in = inlet_state["v_m"]
    v_t_in = inlet_state["v_t"]
    d_in = inlet_state["d"]
    p_in = inlet_state["p"]

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
        v_m, v_t, d, p = y

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
        h0 = h + 0.5*v**2

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
                d*v_t**2/r*np.sin(phi) - 2*tau_w/b*np.cos(alpha),
                -d*v_t*v_m/r*np.sin(phi)-2*tau_w/b*np.sin(alpha),
                2*(tau_w*v + q_w)/(b*dedp_d)
            ]
        ) 

        dy = scipy.linalg.solve(M, S)

        out = {"v_t" : v_t,
               "v_m" : v_m,
               "v" : v,
               "alpha" : alpha,
               "d" : d,
               "p" : p,
               "s" : s,
               "h" : h,
               "h0" : h0}

        return dy, out
    
    solution = scipy.integrate.solve_ivp(
        lambda t,y: odefun(t,y)[0],
        [0, m_out],
        [v_m_in, v_t_in, d_in, p_in],
        method = "RK45",
        rtol = 1e-6,
        atol = 1e-6,
    )

    exit_state = odefun(solution.t[-1], solution.y[:, -1])

    return solution, exit_state

def r_fun(r_in, phi, m):

    "Calculate the radius from the meridonial coordinate"

    return r_in + np.sin(phi)*m

def b_fun(b_in, div, m):

    "Calculate the channel width from the meridonial coordinate"

    return b_in + 2*np.tan(div)*m


if __name__ == '__main__':

    fluid_name = "air"
    fluid = cp.AbstractState("HEOS", fluid_name)

    # Set inlet conditions
    p_in = 101325
    T_in = 273.15+20
    Ma_in = 0.75
    alpha_in = 65*np.pi/180

    # Set friction coefficient and wall heat flux
    Cf = 0.0
    q_w = 0
    
    # Calculate remaining input conditions
    fluid.update(cp.PT_INPUTS, p_in, T_in)
    d_in = fluid.rhomass()
    a_in = fluid.speed_sound()
    v_in = Ma_in*a_in
    v_m_in = v_in*np.cos(alpha_in)
    v_t_in = v_in*np.sin(alpha_in)

    inlet_state = {
        "v_m" : v_m_in,
        "v_t" : v_t_in,
        "p" : p_in,
        "d" : d_in,
    }

    diffuser_geometry = {
        "phi" : 90*np.pi/180,
        "div" : 0*np.pi/180,
        "r_in" : 1.0,
        "r_out" : 3.0,
        "b_in" : 0.25}

    solution, exit_state = radial_vaneless_diffuser_1d(
        fluid,
        Cf,
        q_w,
        inlet_state,
        diffuser_geometry
    )

    print(p_in)
    print(solution.t)
    print(solution.y)
