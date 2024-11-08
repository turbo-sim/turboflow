
import numpy as np
import CoolProp as cp
from scipy.optimize._numdiff import approx_derivative
from .. import math

CHOKING_CRITERIONS = [
    "no_throat",
    "minimize_throat_residual",
    "evaluate_throat"
]


def evaluate_choking(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

    """
    Evaluate throat residuals
    """

    choking_criterion = model_options ["choking_criterion"]

    throat_functions = {
        "no_throat" : lambda *args: ({}, {}),
        "minimize_throat_residual" : get_throat_gradient,
        "evaluate_throat" : evaluate_throat,
    }

    residuals, results = throat_functions[choking_criterion](independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat)

    return residuals, results

def get_res_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat):

        """ 
        Get mass flow rate residual at the throat for normalized velocity x
        """

        w_throat = w_throat[0]

        # Throat
        h0_rel_throat = rothalpy + 0.5*u_throat**2 # Assume same radius at throat and inlet
        h_throat = h0_rel_throat - 0.5*w_throat**2
        s_throat = s_in # Assume isentropic flow
        statsic_properties_throat = fluid.get_props(cp.HmassSmass_INPUTS, h_throat, s_throat)
        d_throat = statsic_properties_throat["d"]
        m_throat = d_throat*w_throat*A_throat

        res = 1-m_throat/mass_flow_rate

        return np.array([res, math.smooth_abs(res, method="logarithmic", epsilon=1e-1)])

def get_res_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat, eps, f0):
    jac = approx_derivative(
        get_res_throat,
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
            u_throat,
        )
    )

    return jac[:, 0]

def get_throat_gradient(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

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

    # Load model options
    rel_step_fd = model_options["rel_step_fd"]
    
    # Evaluate gradient
    w_throat = np.array([w_throat])
    f0 = get_res_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat)
    eps = rel_step_fd * w_throat
    grad = get_res_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat, eps, f0)

    # Store results
    throat_residual = {
        "gradient_throat_residual" : grad[0],
    }

    throat_results = {
        "mass_flow_residual" : f0[0],
        "choked" : f0[0] > 0,
        "w_crit" : w_throat[0],
    }

    return throat_residual, throat_results

def evaluate_throat(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

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

    # Load model options
    rel_step_fd = model_options["rel_step_fd"]

    # def get_res_throat(x, rothalpy, u_throat, s_in, mass_flow_rate):

    #     """ 
    #     Get mass flow rate residual at the throat for normalized velocity x
    #     """

    #     w = x[0]

    #     # Throat
    #     h0_rel = rothalpy + 0.5*u_throat**2 # Assume same radius at throat and inlet
    #     h = h0_rel - 0.5*w**2
    #     s = s_in # Assume isentropic flow
    #     statsic_properties = fluid.get_props(cp.HmassSmass_INPUTS, h, s)
    #     d = statsic_properties["d"]
    #     m = d*w*A_throat

    #     res = 1-m/mass_flow_rate

    #     return math.smooth_abs(res, method="logarithmic", epsilon=1e-1)
    #     # return res
                
    # def get_gradient(w_throat, rothalpy, u_throat, s_in, mass_flow_rate, eps):

    #     jac = approx_derivative(
    #         get_res_throat,
    #         w_throat,
    #         abs_step = eps,
    #         # f0 = f0,
    #         method="3-point",
    #         args = (
    #             rothalpy,
    #             u_throat,
    #             s_in,
    #             mass_flow_rate,
    #         )
    #     )

    #     return jac[0]
        
    # Evaluate gradient
    w_throat = np.array([w_throat])
    f0 = get_res_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat)
    eps = rel_step_fd * w_throat
    gradient = get_res_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat, eps, f0)

    # Evaluate second derivative
    second_deriv = approx_derivative(
        get_res_gradient, 
        w_throat,
        abs_step = eps,
        # f0 = gradient,
        method = "3-point",
        args = (
            fluid,
            mass_flow_rate,
            rothalpy,
            s_in,
            A_throat,
            u_throat,
            eps,
            f0,
    )
    )

    throat_residual = {
        "gradient_throat_residual" : gradient[1],
    }

    throat_results = {
        "mass_flow_residual" : f0[1],
        "choked" : f0[1]-1e-4 > 0,
        "second_deriv" : second_deriv[:, 0][1],
        "minima" : second_deriv[:, 0][1] > 0, 
        "w_throat" : w_throat[0],
    }

    return throat_residual, throat_results



