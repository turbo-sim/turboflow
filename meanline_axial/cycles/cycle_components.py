import numpy as np

from scipy.integrate import solve_ivp

# from .. import utilities
from .. import fluid_properties as props

def heat_exchanger(
    fluid_hot,
    h_in_hot,
    h_out_hot,
    p_in_hot,
    p_out_hot,
    fluid_cold,
    h_in_cold,
    h_out_cold,
    p_in_cold,
    p_out_cold,
    num_steps=50,
    counter_current=True,
):
    # Evaluate properties on the hot side
    hot_side = heat_transfer_process(
        fluid=fluid_hot,
        h_1=h_in_hot,
        h_2=h_out_hot,
        p_1=p_in_hot,
        p_2=p_out_hot,
        num_steps=num_steps,
    )

    # Evaluate properties in the cold side
    cold_side = heat_transfer_process(
        fluid=fluid_cold,
        h_1=h_in_cold,
        h_2=h_out_cold,
        p_1=p_in_cold,
        p_2=p_out_cold,
        num_steps=num_steps,
    )

    # Sort values for temperature difference calculation
    if counter_current:
        for key, value in hot_side["states"].items():
            hot_side["states"][key] = np.flip(value)

    # Compute temperature difference
    dT = hot_side["states"]["T"] - cold_side["states"]["T"]

    # Compute mass flow ratio
    # Use 1.0 for cases with no heat exchange (i.e., no recuperator)
    dh_hot = hot_side["state_in"].h - hot_side["state_out"].h
    dh_cold = cold_side["state_out"].h - cold_side["state_in"].h
    mass_ratio = 1.00 if (dh_cold == 0 or dh_hot == 0) else dh_cold / dh_hot

    # Create result dictionary
    result = {
        "type": "heat_exchanger",
        "hot_side": hot_side,
        "cold_side": cold_side,
        "q_hot_side": dh_hot,
        "q_cold_side": dh_cold,
        "temperature_difference": dT,
        "mass_flow_ratio": mass_ratio,
    }

    return result


def heat_transfer_process(fluid, h_1, p_1, h_2, p_2, num_steps=25):
    # Generate linearly spaced arrays for pressure and enthalpy
    p_array = np.linspace(p_1, p_2, num_steps)
    h_array = np.linspace(h_1, h_2, num_steps)

    # Initialize lists to store states for hot and cold sides
    states = []

    # Calculate states for hot side
    for p, h in zip(p_array, h_array):
        states.append(fluid.set_state(props.HmassP_INPUTS, h, p))

    # Store inlet and outlet states
    state_in = states[0]
    state_out = states[-1]

    # Create result dictionary
    result = {
        "states": props.states_to_dict(states),
        "fluid_name": fluid.name,
        "state_in": state_in,
        "state_out": state_out,
        "mass_flow": np.nan,
        "color": "black",
    }

    return result


def compression_process(
    fluid, h_in, p_in, p_out, efficiency, efficiency_type="isentropic", num_steps=10
):
    """
    Calculate properties along a compression process defined by a isentropic or polytropic efficiency

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties
    h_in : float
        Enthalpy at the start of the compression process.
    p_in : float
        Pressure at the start of the compression process.
    p_out : float
        Pressure at the end of the compression process.
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used in the process ('isentropic' or 'polytropic'). Default is 'isentropic'.
    num_steps : int, optional
        The number of steps for the polytropic process calculation. Default is 50.

    Returns
    -------
    tuple
        Tuple containing (state_out, states) where states is a list with all intermediate states

    Raises
    ------
    ValueError
        If an invalid 'efficiency_type' is provided.

    """

    # Compute inlet state
    state_in = fluid.set_state(props.HmassP_INPUTS, h_in, p_in)
    state_out_is = fluid.set_state(props.PSmass_INPUTS, p_out, state_in.s)

    # Evaluate compression process
    if efficiency_type == "isentropic":
        # Compute outlet state according to the definition of isentropic efficiency
        h_out = state_in.h + (state_out_is.h - state_in.h) / efficiency
        state_out = fluid.set_state(props.HmassP_INPUTS, h_out, p_out)
        states = [state_in, state_out]

    elif efficiency_type == "polytropic":
        # Differential equation defining the polytropic compression
        def odefun(p, h):
            state = fluid.set_state(props.HmassP_INPUTS, h, p)
            dhdp = 1.0 / (efficiency * state.rho)
            return dhdp, state

        # Solve polytropic compression differential equation
        sol = solve_ivp(
            lambda p, h: odefun(p, h)[0],
            [state_in.p, p_out],
            [state_in.h],
            t_eval=np.linspace(state_in.p, p_out, num_steps),
            method="RK45",
        )

        # Evaluate fluid properties at intermediate states
        states = postprocess_ode(sol.t, sol.y, odefun)
        state_in, state_out = states[0], states[-1]

    else:
        raise ValueError("Invalid efficiency_type. Use 'isentropic' or 'polytropic'.")

    # Compute work
    isentropic_work = state_out_is.h - state_in.h
    specific_work = state_out.h - state_in.h

    # Compute degree of superheating
    state_in = props.calculate_superheating(state_in, fluid)
    state_out = props.calculate_superheating(state_out, fluid)

    # Compute degree of subcooling
    state_in = props.calculate_subcooling(state_in, fluid)
    state_out = props.calculate_subcooling(state_out, fluid)

    # Create result dictionary
    result = {
        "type": "compressor",
        "fluid_name": fluid.name,
        "states": props.states_to_dict(states),
        "state_in": state_in,
        "state_out": state_out,
        "efficiency": efficiency,
        "efficiency_type": efficiency_type,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
        "pressure_ratio": state_out.p/state_in.p,
        "mass_flow": np.nan,
        "color": "black",
    }

    return result


def expansion_process(
    fluid, h_in, p_in, p_out, efficiency, efficiency_type="isentropic", num_steps=50
):
    """
    Calculate properties along a compression process defined by a isentropic or polytropic efficiency

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties
    h_in : float
        Enthalpy at the start of the compression process.
    p_in : float
        Pressure at the start of the compression process.
    p_out : float
        Pressure at the end of the compression process.
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used in the process ('isentropic' or 'polytropic'). Default is 'isentropic'.
    num_steps : int, optional
        The number of steps for the polytropic process calculation. Default is 50.

    Returns
    -------
    tuple
        Tuple containing (state_out, states) where states is a list with all intermediate states

    Raises
    ------
    ValueError
        If an invalid 'efficiency_type' is provided.

    """

    # Compute inlet state
    state_in = fluid.set_state(props.HmassP_INPUTS, h_in, p_in)
    state_out_is = fluid.set_state(props.PSmass_INPUTS, p_out, state_in.s)
    if efficiency_type == "isentropic":
        # Compute outlet state according to the definition of isentropic efficiency
        h_out = state_in.h - efficiency * (state_in.h - state_out_is.h)
        state_out = fluid.set_state(props.HmassP_INPUTS, h_out, p_out)
        states = [state_in, state_out]

    elif efficiency_type == "polytropic":
        # Differential equation defining the polytropic expansion
        def odefun(p, h):
            state = fluid.set_state(props.HmassP_INPUTS, h, p)
            dhdp = efficiency / state.rho
            return dhdp, state

        # Solve polytropic expansion differential equation
        sol = solve_ivp(
            lambda p, h: odefun(p, h)[0],
            [state_in.p, p_out],
            [state_in.h],
            t_eval=np.linspace(state_in.p, p_out, num_steps),
            method="RK45",
        )

        # Evaluate fluid properties at intermediate states
        states = postprocess_ode(sol.t, sol.y, odefun)
        state_in, state_out = states[0], states[-1]

    else:
        raise ValueError("Invalid efficiency_type. Use 'isentropic' or 'polytropic'.")

    # Compute work
    isentropic_work = state_in.h - state_out_is.h
    specific_work = state_in.h - state_out.h

    # Compute degree of superheating
    state_in = props.calculate_superheating(state_in, fluid)
    state_out = props.calculate_superheating(state_out, fluid)

    # Compute degree of subcooling
    state_in = props.calculate_subcooling(state_in, fluid)
    state_out = props.calculate_subcooling(state_out, fluid)

    # Create result dictionary
    result = {
        "type": "turbine",
        "fluid_name": fluid.name,
        "states": props.states_to_dict(states),
        "state_in": state_in,
        "state_out": state_out,
        "efficiency": efficiency,
        "efficiency_type": efficiency_type,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
        "pressure_ratio": state_in.p/state_out.p,
        "mass_flow": np.nan,
        "color": "black",
    }

    return result


def isenthalpic_valve(state_in, p_out, fluid, N=50):
    p_array = np.linspace(state_in.p, p_out, N)
    h_array = state_in.h * p_array
    states = []
    for p, h in zip(p_array, h_array):
        states.append(fluid.set_state(props.HmassP_INPUTS, h, p))
    return states


def postprocess_ode(t, y, ode_handle):
    # Collect additional properties for each integration step
    ode_out = []
    for t_i, y_i in zip(t, y.T):
        _, state = ode_handle(t_i, y_i)
        ode_out.append(state)
    return ode_out
