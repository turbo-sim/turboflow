from scipy.optimize._numdiff import approx_derivative
from scipy import interpolate
import numpy as np

from .. import math
from . import flow_model as fm
from . import deviation_model as dm

CHOKING_CRITERIONS = [
    "critical_mass_flow_rate",
    "critical_mach_number",
    "critical_isentropic_throat",
]

REGRESSION_COEFFICIENTS = {'air' : [0.999984354, -0.368037583, 0.202442269, -0.096333086, 0.023783628],
                           'R125' : [0.999983586, -0.319079827, 0.165487456, -0.078246564, 0.01996359]}


def evaluate_choking(
    choking_input,
    inlet_plane,
    exit_plane,
    fluid,
    geometry,
    angular_speed,
    model_options,
    reference_values,
):
    r"""

    Calculate condition for choking and evaluate wheter or not the cascade is choked, based on selected choking model.

    Parameters
    ----------
    choking_input : dict
        Dictionary containing necessary input parameters required for selected choking model. See each models documentation for specifics.
    inlet_plane : dict
        Dictionary containing data on the inlet plane for the actual cascade operating condition.
    exit_plane : dict
        Dictionary containing data on the exit plane for the actual cascade operating condition.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    model_options : dict
        Options for the model used to evaluate choking.
    reference_values : dict
        Reference values used in the calculation, including the reference mass flow rate.

    Returns
    -------
    dict
        Dictionary containing the residuals of choking model
    dict
        Dictionary containing state information on the critical conditions.

    Raises
    ------
    ValueError
        If an invalid choking model is provided.

    """

    critical_cascade_functions = {
        CHOKING_CRITERIONS[0]: critical_mass_flow_rate,
        CHOKING_CRITERIONS[1]: critical_mach_number,
        CHOKING_CRITERIONS[2]: critical_isentropic_throat,
    }

    # Evaluate loss model
    model = model_options["choking_criterion"]
    if model in critical_cascade_functions:
        return critical_cascade_functions[model](
            choking_input,
            inlet_plane,
            exit_plane,
            fluid,
            geometry,
            angular_speed,
            model_options,
            reference_values,
        )
    else:
        options = ", ".join(f"'{k}'" for k in CHOKING_CRITERIONS)
        raise ValueError(
            f"Invalid critical cascade model '{model}'. Available options: {options}"
        )

def critical_mach_number(
    choking_input,
    inlet_plane,
    exit_plane,
    fluid,
    geometry,
    angular_speed,
    model_options,
    reference_values,
):
    r"""

    Calculate condition for choking and evaluate wheter or not the cascade is choked, based on the critical_mach_number choking model.

    This choking model evaluates the cascade throat and checks if the throat mach number exceed the critical. The critical mach number is calculated
    from a correlation depending on the throat loss coefficient. The exit flow angle is calculated by the selected deviation model at subsonic condition,
    and by ensuring that the mach at the throat equals the critical at supercritical conditions.

    Parameters
    ----------
    choking_input : dict
        Dictionary containing scaled input parameters required for selected choking model. Required items are:

        - `w_crit_throat` : throat velocity.
        - `s_crit_throat` : throat entropy.
        - `beta_crit_throat` : throat relative flow angle.
    inlet_plane : dict
        Dictionary containing data on the inlet plane for the actual cascade operating condition.
    exit_plane : dict
        Dictionary containing data on the exit plane for the actual cascade operating condition.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    model_options : dict
        Options for the model used to evaluate choking.
    reference_values : dict
        Reference values used in the calculation, including the reference mass flow rate.

    Returns
    -------
    dict
        Dictionary containing the residuals of choking model:

        - `m*`: Mass flow rate residual.
        - `Y*`: Loss error residual.
        - `beta*`: Residual of the flow angle.
        - `choking`: Choking residual.
    dict
        Dictionary containing relevant information on the critical state and throat plane.

    """

    # Rename variables
    loss_model = model_options["loss_model"]
    blockage = model_options["blockage_model"]
    deviation_model = model_options["deviation_model"]
    A_throat = geometry["A_throat"]
    A_out = geometry["A_out"]
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Evaluate throat
    cascade_throat_input = {
        "w": choking_input["w_crit_throat"] * v0,
        "s": choking_input["s_crit_throat"] * s_range + s_min,
        "beta" : np.sign(exit_plane["beta"])
        * math.arccosd(geometry["A_throat"] / geometry["A_out"]),
        "rothalpy": inlet_plane["rothalpy"],
    }
    throat_plane, loss_dict = fm.evaluate_cascade_throat(
        cascade_throat_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        blockage,
        loss_model,
    )

    # Evaluate critical mach
    Y_tot = loss_dict["loss_total"]
    eta = (throat_plane["h0_rel"]-throat_plane["h"])/(throat_plane["h0_rel"]-throat_plane["h_is"])
    critical_mach = get_mach_crit(throat_plane["gamma"], eta)

    # Evaluate if flow cascade is choked or not an add choking residual
    if exit_plane["Ma_rel"] <= critical_mach:
        beta_model = np.sign(exit_plane["beta"]) * dm.get_subsonic_deviation(
            exit_plane["Ma_rel"], critical_mach, geometry, deviation_model
        )

        # Compute error of guessed beta and deviation model
        choking_residual = math.cosd(beta_model) - math.cosd(exit_plane["beta"])
    else:
        choking_residual = throat_plane["Ma_rel"] - critical_mach

    # Evaluate resiudals
    residual_values = np.array(
        [
            (inlet_plane["mass_flow"] - throat_plane["mass_flow"])
            / reference_values["mass_flow_ref"],
            throat_plane["loss_error"],
            choking_residual,
        ]
    )
    residual_keys = ["m*", "Y*", "beta*", "choking"]
    residuals_critical = dict(zip(residual_keys, residual_values))

    # Define output values
    critical_state = {
        "critical_mach": critical_mach,
    }
    throat_plane = {f"{key}_throat": val for key, val in throat_plane.items()}

    return residuals_critical, {**critical_state, **throat_plane}


def get_mach_crit(gamma, eta):
    r"""
    Compute the critical Mach number for non-isentropic flow in a nozzle.
 
    This function calculates the critical Mach number (:math:`\text{Ma}_\text{crit}`) using the
    given specific heat ratio (:math:`\gamma`) and nozzle efficiency (:math:`\eta`).
   
    .. math::
        \text{Ma}_\text{crit}^{2} = \left(\frac{2}{\gamma-1}\right)\left(\frac{1}{\hat{T}_\text{crit}}-1\right)
        = \left(\frac{2}{\gamma-1}\right)\left[\frac{4 \alpha-2}{(2 \alpha+\eta-3) + \sqrt{(1+\eta)^{2}+4 \alpha(1+\alpha-3 \eta)}} - 1\right]
 
    where:
 
    .. math::
        \alpha = \frac{\gamma}{\gamma-1}
 
    and
 
    .. math::
        \eta = 1 - \Delta \phi^2
 
    Here, :math:`\Delta \phi^2` is the kinetic energy loss coefficient.
       
    Parameters
    ----------
    gamma : float
        Specific heat ratio of the gas.
    eta : float
        Nozzle efficiency, related to the kinetic energy loss coefficient.
 
    Returns
    -------
    float
        Critical Mach number.
    """
    alpha = gamma / (gamma - 1)
    T_hat_crit = (2*alpha + eta - 3 + np.sqrt((1+eta)**2 + 4*alpha*(1+alpha-3*eta))) / (4*alpha - 2)
    Ma_crit = np.sqrt(2 / (gamma - 1) * (1 / T_hat_crit - 1))
    return Ma_crit


def get_flow_capacity(Ma, gamma, eta):
    r"""
    Compute dimensionless mass flow rate for non-isentropic flow in a nozzle.

    This function calculates the dimensionless mass flow rate (:math:`\Phi`) using the 
    given Mach number (:math:`\text{Ma}`), specific heat ratio (:math:`\gamma`), and efficiency (:math:`\eta`).
    
    .. math::
        \Phi = \left(\frac{\dot{m} \sqrt{R T_{0}}}{p_{0}  A}\right) = \left(\frac{2 \gamma}{\gamma-1}\right)^{1 / 2} \cdot \frac{\sqrt{1-\hat{T}}}{\hat{T}}
        \left[1-\frac{1}{\eta}(1-\hat{T})\right]^{\frac{\gamma}{\gamma-1}}

    where:

    .. math::
        \hat{T} = \left(\frac{T}{T_{0}}\right)

    and

    .. math::
        \eta = 1 - \Delta \phi^2

    Here, :math:`\Delta \phi^2` is the kinetic energy loss coefficient.
    
    Parameters
    ----------
    Ma : float
        Mach number of the flow.
    gamma : float
        Specific heat ratio of the gas.
    eta : float
        Nozzle efficiency, related to the kinetic energy loss coefficient.

    Returns
    -------
    float
        Dimensionless mass flow rate.
    """
    T_hat = (1 + (gamma - 1) / 2 * Ma**2) ** (-1)
    Phi = np.sqrt(2 * gamma / (gamma - 1)) * np.sqrt(1 - T_hat) / T_hat * (1 - 1 / eta * (1 - T_hat)) ** (gamma / (gamma - 1))
    return Phi



def critical_mass_flow_rate(
    choking_input,
    inlet_plane,
    exit_plane,
    fluid,
    geometry,
    angular_speed,
    model_options,
    reference_values,
):
    r"""

    Calculate condition for choking and evaluate wheter or not the cascade is choked, based on the `critical_mass_flow_rate` choking model.

    This choking model evaluate the critical state by optimizing the mass flow rate at the throat through the method of lagrange multipliers, and checks if the throat mach number exceed the critical. 
    The exit flow angle is calculated by the selected devaition model at subsonic condition, and from the critical mass flow rate at supersonic conditions. 

    Compute the gradient of the Lagrange function of the critical mass flow rate and the residuals of mass
    conservation and loss computation equations at the throat.

    The constrained optimization problem to maximize the mass flow rate can be converted to a set of equations through the method of lagrange multipliers. The method
    utlize that at the optimal point, the gradient of the constraints are proportional to the gradient fo the objective function. Thus the solution to the optimization problem 
    eaulas the solution as the following set of equations: 
        
    .. math::

        \nabla L = \begin{bmatrix}
        \frac{\partial f}{\partial x_1} + \lambda_1 \cdot \frac{\partial g_1}{\partial x_1} + \lambda_2 \cdot \frac{\partial g_2}{\partial x_1} \\
        \frac{\partial f}{\partial x_2} + \lambda_1 \cdot \frac{\partial g_1}{\partial x_2} + \lambda_2 \cdot \frac{\partial g_2}{\partial x_2} \\
        \frac{\partial f}{\partial x_3} + \lambda_1 \cdot \frac{\partial g_1}{\partial x_3} + \lambda_2 \cdot \frac{\partial g_2}{\partial x_3} \\
        g_1(x_1, x_2, x_3) \\
        g_2(x_1, x_2, x_3) \\
        \end{bmatrix} = 0
        
    This function returns the value of the three last equations in the set above, while the two first equations are used to 
    explicitly calculate the lagrange multipliers.
    
    .. math::
        \lambda_1 = \frac{\frac{\partial g_2}{\partial x_2}\cdot\frac{-\partial f}{\partial x_1} - \frac{\partial g_2}{\partial x_1}\cdot-\frac{\partial f}{\partial x_2}} 
                     {\frac{\partial g_1}{\partial x_1}\cdot\frac{\partial g_2}{\partial x_2} - \frac{\partial g_2}{\partial x_1}\cdot\frac{\partial g_1}{\partial x_2}} \\                     
        \lambda_2 = \frac{\frac{\partial g_1}{\partial x_1}\cdot\frac{-\partial f}{\partial x_2} - \frac{\partial g_1}{\partial x_2}\cdot-\frac{\partial f}{\partial x_1}} 
                     {\frac{\partial g_1}{\partial x_1}\cdot\frac{\partial g_2}{\partial x_2} - \frac{\partial g_2}{\partial x_1}\cdot\frac{\partial g_1}{\partial x_2}} 

    By transforming the problem into a system of equations, this approach allows the evaluation of the critical
    point without directly solving an optimization problem. One significant advantage of this
    equation-oriented method is that it enables the coupling of these critical condition equations with the
    other modeling equations. This integrated system of equations can then be efficiently solved using gradient-based
    root finding algorithms (e.g., Newton-Raphson solvers).

    Such a coupled solution strategy, as opposed to segregated approaches where nested systems are solved
    sequentially and iteratively, offers superior computational efficiency. This method thus provides
    a more direct and computationally effective way of determining the critical conditions in a cascade.

    Parameters
    ----------
    critical_cascade_input : dict
        Dictionary containing scaled input parameters required for selected choking model. Required items are:

        - `v_crit_in` : inlet velocity at critical state.  
        - `w_crit_throat` : throat velocity at critical state. 
        - `s_crit_throat` : throat entropy at critical state.
    inlet_plane : dict
        Dictionary containing data on the inlet plane for the actual cascade operating condition.
    exit_plane : dict
        Dictionary containing data on the exit plane for the actual cascade operating condition.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    model_options : dict
        Options for the model used in the critical condition evaluation.
    reference_values : dict
        Reference values used in the calculation, including the reference mass flow rate.

    Returns
    -------
    dict
        Dictionary containing the residuals of choking model:
        
        - `m*`: Mass flow rate residual.
        - `Y*`: Loss error residual.
        - `L*`: Residual of the lagrange function.
        - `choking`: Choking residual.
    dict
        Dictionary containing state information on the critical conditions at inlet and throat plane. 

    """

    # Load reference values
    mass_flow_ref = reference_values["mass_flow_ref"]

    # Load model options
    loss_model = model_options["loss_model"]
    blockage = model_options["blockage_model"]
    deviation_model = model_options["deviation_model"]

    # Define critical state dictionary to store information
    critical_state = {}

    # Define array for inputs in compute critical values
    x_crit = np.array(
        [
            choking_input["v_crit_in"],
            choking_input["w_crit_throat"],
            choking_input["s_crit_throat"],
        ]
    )

    # Evaluate the current cascade at critical conditions
    f0 = compute_critical_values(
        x_crit,
        inlet_plane,
        fluid,
        geometry,
        angular_speed,
        critical_state,
        model_options,
        reference_values,
    )

    # Evaluate the Jacobian of the evaluate_critical_cascade function
    J = compute_critical_jacobian(
        x_crit,
        inlet_plane,
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
    )

    # Calculate the Lagrange multipliers explicitly
    determinant = a11 * a22 - a12 * a21
    l1_det = a22 * b1 - a12 * b2
    l2_det = a11 * b2 - a21 * b1

    # Evaluate the last equation
    df, dg1, dg2 = J[0, 2 - 1], J[1, 2 - 1], J[2, 2 - 1]
    grad = (determinant * df + l1_det * dg1 + l2_det * dg2) / mass_flow_ref

    critical_mach = critical_state["throat_plane"]["Ma_rel"]
    critical_mass_flow = critical_state["throat_plane"]["mass_flow"]
    if exit_plane["Ma_rel"] <= critical_mach:
        beta_model = np.sign(exit_plane["beta"]) * dm.get_subsonic_deviation(
            exit_plane["Ma_rel"], critical_mach, geometry, deviation_model
        )
    else:
        beta_model = np.sign(exit_plane["beta"]) * math.arccosd(
            critical_mass_flow / exit_plane["d"] / exit_plane["w"] / geometry["A_out"]
        )
    choking_residual = math.cosd(beta_model) - math.cosd(exit_plane["beta"])

    # Restructure critical state dictionary
    inlet_plane = {
        f"critical_{key}_in": val for key, val in critical_state["inlet_plane"].items()
    }
    throat_plane = {
        f"critical_{key}_throat": val
        for key, val in critical_state["throat_plane"].items()
    }
    critical_state = {**inlet_plane, **throat_plane}

    # Return last 3 equations of the Lagrangian gradient (df/dx2+l1*dg1/dx2+l2*dg2/dx2 and g1, g2)
    g = f0[1:]  # The two constraints
    residual_values = np.concatenate((g, np.array([grad, choking_residual])))
    residual_keys = ["m*", "Y*", "L*", "choking"]
    residuals_critical = dict(zip(residual_keys, residual_values))

    return residuals_critical, critical_state


def compute_critical_values(
    x_crit,
    inlet_plane,
    fluid,
    geometry,
    angular_speed,
    critical_state,
    model_options,
    reference_values,
):
    """
    Compute cascade performance at the critical conditions

    This function evaluates the performance of a cascade at its critical operating point defined by:

        1. Critical inlet absolute velocity,
        2. Critical throat relative velocity,
        3. Critical throat entropy.

    Using these variables, the function calculates the critical mass flow rate and residuals of the mass balance and the loss model equations.

    Parameters
    ----------
    x_crit : numpy.ndarray
        Array containing scaled critical variables `[v_in*, w_throat*, s_throat*]`.
    inlet_plane : dict
        Dictionary containing data on the inlet plane for the actual cascade operating condition.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    critical_state : dict
        Dictionary to store the critical state information.
    model_options : dict
        Options for the model used in the critical condition evaluation.
    reference_values : dict
        Reference values used in the calculations, including mass flow reference and other parameters.

    Returns
    -------
    numpy.ndarray
        An array containing the computed mass flow at the throat plane at critical state and the residuals
        for mass conservation and loss coefficient error.

    """

    # Define model options
    loss_model = model_options["loss_model"]

    # Load reference values
    mass_flow_ref = reference_values["mass_flow_ref"]
    v0 = reference_values["v0"]
    s_range = reference_values["s_range"]
    s_min = reference_values["s_min"]

    # Load input for critical cascade
    s_in = inlet_plane["s"]
    h0_in = inlet_plane["h0"]
    alpha_in = inlet_plane["alpha"]
    v_in, w_throat, s_throat = (
        x_crit[0] * v0,
        x_crit[1] * v0,
        x_crit[2] * s_range + s_min,
    )

    # Evaluate inlet plane
    critical_inlet_input = {
        "v": v_in,
        "s": s_in,
        "h0": h0_in,
        "alpha": alpha_in,
    }
    critical_inlet_plane = fm.evaluate_cascade_inlet(
        critical_inlet_input, fluid, geometry, angular_speed
    )

    # Evaluate throat plane
    critical_throat_input = {
        "w": w_throat,
        "s": s_throat,
        "beta": np.sign(geometry["gauging_angle"])
        * math.arccosd(geometry["A_throat"] / geometry["A_out"]),
        "rothalpy": critical_inlet_plane["rothalpy"],
    }

    critical_throat_plane, loss_dict = fm.evaluate_cascade_throat(
        critical_throat_input,
        fluid,
        geometry,
        critical_inlet_plane,
        angular_speed,
        model_options["blockage_model"],
        loss_model,
    )

    # Add residuals
    residuals = np.array(
        [
            (critical_inlet_plane["mass_flow"] - critical_throat_plane["mass_flow"])
            / mass_flow_ref,
            critical_throat_plane["loss_error"],
        ]
    )

    # Update critical state dictionary
    critical_state["inlet_plane"] = critical_inlet_plane
    critical_state["throat_plane"] = critical_throat_plane

    output = np.insert(residuals, 0, critical_throat_plane["mass_flow"])

    return output


def compute_critical_jacobian(
    x,
    inlet_plane,
    fluid,
    geometry,
    angular_speed,
    critical_state,
    model_options,
    reference_values,
    f0,
):
    """
    Compute the Jacobian matrix of the compute_critical_values function using finite differences.

    This function approximates the Jacobian of a combined function that includes the mass flow rate value,
    mass balance residual, and loss model evaluation residual at the critical point. It uses forward finite
    difference to approximate the partial derivatives of the Jacobian matrix.

    Parameters
    ----------
    x : numpy.ndarray
        Array of input variables for the `compute_critical_values` function.
    inlet_plane : dict
        Dictionary containing data on the inlet plane for the actual cascade operating condition.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    critical_state : dict
        Dictionary to store the critical state information.
    model_options : dict
        Options for the model used in the critical condition evaluation.
    reference_values : dict
        Reference values used in the calculations, including mass flow reference and other parameters.
    f0 : numpy.ndarray
        The function value at x, used for finite difference approximation.

    Returns
    -------
    numpy.ndarray
        The approximated Jacobian matrix of the compute_critical_values function.

    """

    # Define finite difference relative step size
    eps = model_options["rel_step_fd"] * x

    # Approximate problem Jacobian by finite differences
    jacobian = approx_derivative(
        compute_critical_values,
        x,
        method="2-point",
        f0=f0,
        abs_step=eps,
        args=(
            inlet_plane,
            fluid,
            geometry,
            angular_speed,
            critical_state,
            model_options,
            reference_values,
        ),
    )

    return jacobian


def critical_isentropic_throat(
    choking_input,
    inlet_plane,
    exit_plane,
    fluid,
    geometry,
    angular_speed,
    model_options,
    reference_values,
):
    r"""

    Calculate condition for choking and evaluate wheter or not the cascade is choked, based on the `critical_isentropic_throat` choking model.

    This choking model evaluates the cascade throat and checks if the throat mach number exceed unity. The throat is isentropic from inlet to throat.
    The exit flow angle is determined by ensuring that the mach at throat mach number equals the exit for subconic condition, and unity for supersonic conditions.

    Parameters
    ----------
    choking_input : dict
        Dictionary containing scaled input parameters required for selected choking model. Required items are:

        - `w_crit_throat` : throat velocity.
    inlet_plane : dict
        Dictionary containing data on the inlet plane for the actual cascade operating condition.
    exit_plane : dict
        Dictionary containing data on the exit plane for the actual cascade operating condition.
    fluid : object
        A fluid object with methods for thermodynamic property calculations.
    geometry : dict
        Geometric parameters of the cascade.
    angular_speed : float
        Angular speed of the cascade.
    model_options : dict
        Options for the model used to evaluate choking.
    reference_values : dict
        Reference values used in the calculation, including the reference mass flow rate.

    Returns
    -------
    dict
        Dictionary containing the residuals of choking model:

        - `m*`: Mass flow rate residual.
        - `choking`: Choking residual.
    dict
        Dictionary satisfying the required elements for critical state dict.

    """

    # Rename variables
    loss_model = model_options["loss_model"]
    blockage = model_options["blockage_model"]
    deviation_model = model_options["deviation_model"]
    A_throat = geometry["A_throat"]
    A_out = geometry["A_out"]
    v0 = reference_values["v0"]
    angle_range = reference_values["angle_range"]
    angle_min = reference_values["angle_min"]

    # Evaluate throat
    cascade_throat_input = {
        "w": choking_input["w_crit_throat"] * v0,
        "s": inlet_plane["s"],
        "beta": np.sign(exit_plane["beta"])
        * math.arccosd(geometry["A_throat"] / geometry["A_out"]),
        "rothalpy": inlet_plane["rothalpy"],
    }
    throat_plane, loss_dict = fm.evaluate_cascade_throat(
        cascade_throat_input,
        fluid,
        geometry,
        inlet_plane,
        angular_speed,
        blockage,
        loss_model,
    )

    # Evaluate critical mach
    choking_residual = throat_plane["Ma_rel"] - min(exit_plane["Ma_rel"], 1)

    # Evaluate resiudals
    residual_values = np.array(
        [
            (inlet_plane["mass_flow"] - throat_plane["mass_flow"])
            / reference_values["mass_flow_ref"],
            choking_residual,
        ]
    )
    residual_keys = ["m*", "choking"]
    residuals_critical = dict(zip(residual_keys, residual_values))

    # Define output values
    throat_plane = {f"{key}_throat": val for key, val in throat_plane.items()}

    return residuals_critical, throat_plane
