
import numpy as np
import CoolProp as cp
from scipy.optimize._numdiff import approx_derivative

CHOKING_CRITERIONS = [
    "no_throat",
    "critical_isentropic_throat",
]


def evaluate_choking(independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat = 0):

    """
    Evaluate throat residuals
    """

    choking_criterion = model_options ["choking_criterion"]

    throat_functions = {
        "no_throat" : lambda *args: ({}, {}),
        "critical_isentropic_throat" : get_throat_gradient,
    }

    residuals, results = throat_functions[choking_criterion](independents, inlet_state, boundary_conditions, geometry, fluid, model_options, reference_values, component, u_throat)

    return residuals, results

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

    def get_res_throat(w, rothalpy, u_throat, s_in, mass_flow_rate):

        """ 
        Get mass flow rate residual at the throat for normalized velocity x
        """

        # Throat
        h0_rel = rothalpy + 0.5*u_throat**2 # Assume same radius at throat and inlet
        h = h0_rel - 0.5*w**2
        s = s_in # Assume isentropic flow
        statsic_properties = fluid.get_props(cp.HmassSmass_INPUTS, h, s)
        d = statsic_properties["d"]
        m = d*w*A_throat

        res = 1-m/mass_flow_rate

        return res
    
    # Evaluate gradient
    x = np.array([w_throat])
    res = get_res_throat(w_throat, rothalpy, u_throat, s_in, mass_flow_rate)
    eps = rel_step_fd * x
    jac = approx_derivative(
        get_res_throat,
        w_throat,
        abs_step = eps,
        f0 = res,
        method="3-point",
        args = (
            rothalpy,
            u_throat,
            s_in,
            mass_flow_rate,
        )
    )

    throat_residual = {
        "gradient_throat_residual" : jac[0],
    }

    throat_results = {
        "mass_flow_residual" : res,
        "choked" : res > 0,
    }

    return throat_residual, throat_results