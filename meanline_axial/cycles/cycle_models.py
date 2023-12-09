
from .. import utilities
from .. import fluid_properties as props

from .cycle_components import (compression_process,
                               expansion_process,
                               heat_exchanger)

COLORS_MATLAB = utilities.COLORS_MATLAB

def evaluate_recuperated_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):
    # Initialize fluid objects
    working_fluid = props.Fluid(**parameters["working_fluid"])
    heating_fluid = props.Fluid(**parameters["heating_fluid"])
    cooling_fluid = props.Fluid(**parameters["cooling_fluid"])

    # FIXME Improve behavior of critical point and quality calculation for pseudomixtures

    # Rename pressure drops
    dp_heater_h = parameters["heater"]["pressure_drop_hot_side"]
    dp_heater_c = parameters["heater"]["pressure_drop_cold_side"]
    dp_recup_h = parameters["recuperator"]["pressure_drop_hot_side"]
    dp_recup_c = parameters["recuperator"]["pressure_drop_hot_side"]
    dp_cooler_h = parameters["cooler"]["pressure_drop_hot_side"]
    dp_cooler_c = parameters["cooler"]["pressure_drop_hot_side"]

    # Evaluate compressor
    compressor = compression_process(
        working_fluid,
        variables["compressor_inlet_enthalpy"],
        variables["compressor_inlet_pressure"],
        variables["turbine_inlet_pressure"] / (1.0 - dp_heater_c) / (1.0 - dp_recup_c),
        parameters["compressor"]["efficiency"],
        efficiency_type=parameters["compressor"]["efficiency_type"],
    )

    # Evaluate turbine
    turbine = expansion_process(
        working_fluid,
        variables["turbine_inlet_enthalpy"],
        variables["turbine_inlet_pressure"],
        variables["compressor_inlet_pressure"]
        / (1.0 - dp_recup_h)
        / (1.0 - dp_cooler_h),
        parameters["turbine"]["efficiency"],
        efficiency_type=parameters["turbine"]["efficiency_type"],
    )

    # Evaluate recuperator
    eps = variables["recuperator_effectiveness"]
    T_in_hot = turbine["state_out"].T
    p_in_cold = compressor["state_out"].p
    h_in_cold = compressor["state_out"].h
    p_out_cold = p_in_cold * (1.0 - dp_recup_c)
    h_out_cold_ideal = working_fluid.set_state(props.PT_INPUTS, p_out_cold, T_in_hot).h
    h_out_cold_ideal = working_fluid.set_state(props.PT_INPUTS, p_out_cold, T_in_hot).h
    h_out_cold_actual = h_in_cold + eps * (h_out_cold_ideal - h_in_cold)
    p_in_hot = turbine["state_out"].p
    h_in_hot = turbine["state_out"].h
    p_out_hot = p_in_hot * (1.0 - dp_recup_h)
    h_out_hot = h_in_hot - (h_out_cold_actual - h_in_cold)
    recuperator = heat_exchanger(
        working_fluid,
        h_in_hot,
        h_out_hot,
        p_in_hot,
        p_out_hot,
        working_fluid,
        h_in_cold,
        h_out_cold_actual,
        p_in_cold,
        p_out_cold,
        counter_current=True,
        num_steps=parameters["recuperator"]["num_elements"],
    )

    # Evaluate heater
    h_in_cold = recuperator["cold_side"]["state_out"].h
    p_in_cold = recuperator["cold_side"]["state_out"].p
    h_out_cold = turbine["state_in"].h
    p_out_cold = turbine["state_in"].p
    T_in_hot = parameters["heat_source"]["inlet_temperature"]
    p_in_hot = parameters["heat_source"]["inlet_pressure"]
    h_in_hot = heating_fluid.set_state(props.PT_INPUTS, p_in_hot, T_in_hot).h
    T_out_hot = variables["heat_source_exit_temperature"]
    p_out_hot = p_in_hot * (1 - dp_heater_h)
    h_out_hot = heating_fluid.set_state(props.PT_INPUTS, p_out_hot, T_out_hot).h
    heater = heat_exchanger(
        heating_fluid,
        h_in_hot,
        h_out_hot,
        p_in_hot,
        p_out_hot,
        working_fluid,
        h_in_cold,
        h_out_cold,
        p_in_cold,
        p_out_cold,
        counter_current=True,
        num_steps=parameters["heater"]["num_elements"],
    )

    # Evaluate cooler
    T_in_cold = parameters["heat_sink"]["inlet_temperature"]
    p_in_cold = parameters["heat_sink"]["inlet_pressure"]
    h_in_cold = cooling_fluid.set_state(props.PT_INPUTS, p_in_cold, T_in_cold).h
    p_out_cold = p_in_cold * (1 - dp_cooler_c)
    T_out_cold = variables["heat_sink_exit_temperature"]
    h_out_cold = cooling_fluid.set_state(props.PT_INPUTS, p_out_cold, T_out_cold).h
    h_in_hot = recuperator["hot_side"]["state_out"].h
    p_in_hot = recuperator["hot_side"]["state_out"].p
    h_out_hot = compressor["state_in"].h
    p_out_hot = compressor["state_in"].p
    cooler = heat_exchanger(
        working_fluid,
        h_in_hot,
        h_out_hot,
        p_in_hot,
        p_out_hot,
        cooling_fluid,
        h_in_cold,
        h_out_cold,
        p_in_cold,
        p_out_cold,
        counter_current=True,
        num_steps=parameters["cooler"]["num_elements"],
    )

    # Compute performance metrics
    W_net = parameters["net_power"]
    m_w = W_net / (turbine["specific_work"] - compressor["specific_work"])
    m_h = m_w * heater["mass_flow_ratio"]
    m_c = m_w / cooler["mass_flow_ratio"]
    Q_in = m_h * (heater["hot_side"]["state_in"].h - heater["hot_side"]["state_out"].h)
    Q_out = m_w * (cooler["hot_side"]["state_in"].h - cooler["hot_side"]["state_out"].h)
    W_out = m_w * turbine["specific_work"]
    W_in = m_w * compressor["specific_work"]
    h_hot_min = heating_fluid.set_state(
        props.PT_INPUTS,
        parameters["heat_source"]["exit_pressure"],
        parameters["heat_source"]["exit_temperature_min"],
    ).h
    Q_in_max = m_h * (heater["hot_side"]["state_in"].h - h_hot_min)
    cycle_efficiency = W_net / Q_in
    system_efficiency = W_net / Q_in_max
    backwork_ratio = W_in / W_out

    # Add the mass flow to the components
    heater["hot_side"]["mass_flow"] = m_h
    heater["cold_side"]["mass_flow"] = m_w
    recuperator["hot_side"]["mass_flow"] = m_w
    recuperator["cold_side"]["mass_flow"] = m_w
    cooler["hot_side"]["mass_flow"] = m_w
    cooler["cold_side"]["mass_flow"] = m_c
    turbine["mass_flow"] = m_w
    compressor["mass_flow"] = m_w

    # Set colors for plotting
    heater["hot_side"]["color"] = COLORS_MATLAB[6]
    heater["cold_side"]["color"] = COLORS_MATLAB[1]
    recuperator["hot_side"]["color"] = COLORS_MATLAB[1]
    recuperator["cold_side"]["color"] = COLORS_MATLAB[1]
    cooler["hot_side"]["color"] = COLORS_MATLAB[1]
    cooler["cold_side"]["color"] = COLORS_MATLAB[0]
    turbine["color"] = COLORS_MATLAB[1]
    compressor["color"] = COLORS_MATLAB[1]

    # Define dictionary with 1st Law analysis
    energy_analysis = {
        "heater_heat_flow": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "recuperator_heat_flow": Q_out,
        "cooler_heat_flow": Q_out,
        "turbine_power": W_out,
        "compressor_power": W_in,
        "net_power": W_net,
        "mass_flow_heating_fluid": m_h,
        "mass_flow_working_fluid": m_w,
        "mass_flow_cooling_fluid": m_c,
        "cycle_efficiency": cycle_efficiency,
        "system_efficiency": system_efficiency,
        "backwork_ratio": backwork_ratio,
    }

    # # Summary of processes
    # processes = {
    #     "turbine": turbine,
    #     "compressor": compressor,
    #     "recuperator_hot_side": recuperator["hot_side"],
    #     "recuperator_cold_side": recuperator["cold_side"],
    #     "heater_hot_side": heater["hot_side"],
    #     "heater_cold_side": heater["cold_side"],
    #     "cooler_hot_side": cooler["hot_side"],
    #     "cooler_cool_side": cooler["cold_side"],
    # }

    # Summary of processes
    components = {
        "turbine": turbine,
        "compressor": compressor,
        "recuperator": recuperator,
        "heater": heater,
        "heater": heater,
        "cooler": cooler,
    }

    output = {
        "components": components,
        # "processes": processes,
        "energy_analysis": energy_analysis,
    }

    # Evaluate objective function and constraints
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq = utilities.evaluate_constraints(output, constraints)

    # Cycle performance summary
    output = {
        **output,
        "working_fluid": working_fluid,
        "heating_fluid": heating_fluid,
        "cooling_fluid": cooling_fluid,
        "components": components,
        "objective_function": f,
        "equality_constraints": c_eq,
        "inequality_constraints": c_ineq,
    }

    return output


