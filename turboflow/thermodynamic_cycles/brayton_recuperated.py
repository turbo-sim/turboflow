import copy
from .. import utilities
from .. import fluid_properties as props

from .components import compression_process, expansion_process, heat_exchanger

COLORS_MATLAB = utilities.COLORS_MATLAB


def evaluate_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):
    # Create copies to not change the originals
    variables = copy.deepcopy(variables)
    parameters = copy.deepcopy(parameters)

    # Initialize fluid objects
    working_fluid = props.Fluid(
        **parameters.pop("working_fluid"), identifier="working_fluid"
    )
    heating_fluid = props.Fluid(
        **parameters.pop("heating_fluid"), identifier="heating_fluid"
    )
    cooling_fluid = props.Fluid(
        **parameters.pop("cooling_fluid"), identifier="cooling_fluid"
    )

    # Extract heat source/sink parameters and give short names
    T_source_out_max = parameters["heat_source"].pop("exit_temperature_max")
    T_source_out_min = parameters["heat_source"].pop("exit_temperature_min")
    T_sink_out_max = parameters["heat_sink"].pop("exit_temperature_max")
    T_sink_out_min = parameters["heat_sink"].pop("exit_temperature_min")
    p_source_out = parameters["heat_source"].pop("exit_pressure")
    p_sink_out = parameters["heat_sink"].pop("exit_pressure")

    # Compute coldest state at the heat source exit
    source_out_min = heating_fluid.set_state(
        props.PT_INPUTS, p_source_out, T_source_out_min
    )

    # Extract pressure drops and give short names
    dp_heater_h = parameters["heater"].pop("pressure_drop_hot_side")
    dp_heater_c = parameters["heater"].pop("pressure_drop_cold_side")
    dp_recup_h = parameters["recuperator"].pop("pressure_drop_hot_side")
    dp_recup_c = parameters["recuperator"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler"].pop("pressure_drop_cold_side")

    # Extract design variables from dictionary (make sure all are used)
    turbine_inlet_p = variables.pop("turbine_inlet_pressure")
    turbine_inlet_h = variables.pop("turbine_inlet_enthalpy")
    compressor_inlet_p = variables.pop("compressor_inlet_pressure")
    compressor_inlet_h = variables.pop("compressor_inlet_enthalpy")
    recuperator_effectiveness = variables.pop("recuperator_effectiveness")
    heat_source_temperature_out = variables.pop("heat_source_exit_temperature")
    heat_sink_temperature_out = variables.pop("heat_sink_exit_temperature")

    # Evaluate  compressor
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_c)
    compressor_outlet_p = turbine_inlet_p / dp
    compressor_eff = parameters["compressor"].pop("efficiency")
    compressor_eff_type = parameters["compressor"].pop("efficiency_type")
    compressor = compression_process(
        working_fluid,
        compressor_inlet_h,
        compressor_inlet_p,
        compressor_outlet_p,
        compressor_eff,
        compressor_eff_type,
    )

    # Evaluate turbine
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_h)
    turbine_outlet_p = compressor_inlet_p / dp
    turbine_efficiency = parameters["turbine"].pop("efficiency")
    turbine_efficiency_type = parameters["turbine"].pop("efficiency_type")
    turbine = expansion_process(
        working_fluid,
        turbine_inlet_h,
        turbine_inlet_p,
        turbine_outlet_p,
        turbine_efficiency,
        turbine_efficiency_type,
    )

    # Evaluate recuperator
    eps = recuperator_effectiveness
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
    num_elements = parameters["recuperator"].pop("num_elements")
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
        num_steps=num_elements,
    )

    # Evaluate heater
    h_in_cold = recuperator["cold_side"]["state_out"].h
    p_in_cold = recuperator["cold_side"]["state_out"].p
    h_out_cold = turbine["state_in"].h
    p_out_cold = turbine["state_in"].p
    T_in_hot = parameters["heat_source"].pop("inlet_temperature")
    p_in_hot = parameters["heat_source"].pop("inlet_pressure")
    h_in_hot = heating_fluid.set_state(props.PT_INPUTS, p_in_hot, T_in_hot).h
    T_out_hot = heat_source_temperature_out
    p_out_hot = p_in_hot * (1 - dp_heater_h)
    h_out_hot = heating_fluid.set_state(props.PT_INPUTS, p_out_hot, T_out_hot).h
    num_elements = parameters["heater"].pop("num_elements")
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
        num_steps=num_elements,
    )

    # Evaluate heat source pump
    h_in = heater["hot_side"]["state_out"].h
    p_in = heater["hot_side"]["state_out"].p
    p_out = p_source_out
    efficiency = parameters["heat_source_pump"].pop("efficiency")
    efficiency_type = parameters["heat_source_pump"].pop("efficiency_type")
    heat_source_pump = compression_process(
        heating_fluid,
        h_in,
        p_in,
        p_out,
        efficiency,
        efficiency_type,
    )

    # Evaluate heat sink pump
    T_in = parameters["heat_sink"].pop("inlet_temperature")
    p_in = parameters["heat_sink"].pop("inlet_pressure")
    h_in = cooling_fluid.set_state(props.PT_INPUTS, p_in, T_in).h
    p_out = p_sink_out / (1 - dp_cooler_c)
    efficiency = parameters["heat_sink_pump"].pop("efficiency")
    efficiency_type = parameters["heat_sink_pump"].pop("efficiency_type")
    heat_sink_pump = compression_process(
        cooling_fluid,
        h_in,
        p_in,
        p_out,
        efficiency,
        efficiency_type,
    )

    # Evaluate cooler
    p_in_cold = heat_sink_pump["state_out"].p
    h_in_cold = heat_sink_pump["state_out"].h
    p_out_cold = p_in_cold * (1 - dp_cooler_c)
    T_out_cold = heat_sink_temperature_out
    h_out_cold = cooling_fluid.set_state(props.PT_INPUTS, p_out_cold, T_out_cold).h
    h_in_hot = recuperator["hot_side"]["state_out"].h
    p_in_hot = recuperator["hot_side"]["state_out"].p
    h_out_hot = compressor["state_in"].h
    p_out_hot = compressor["state_in"].p
    num_elements = parameters["cooler"].pop("num_elements")
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
        num_steps=num_elements,
    )

    # Compute mass flow rates
    W_net = parameters.pop("net_power")
    turbine_work = turbine["specific_work"]
    compression_work = compressor["specific_work"]
    m_total = W_net / (turbine_work - compression_work)
    m_source = m_total * heater["mass_flow_ratio"]
    m_sink = m_total / cooler["mass_flow_ratio"]

    # Add the mass flow to the components
    heater["hot_side"]["mass_flow"] = m_source
    heater["cold_side"]["mass_flow"] = m_total
    recuperator["hot_side"]["mass_flow"] = m_total
    recuperator["cold_side"]["mass_flow"] = m_total
    cooler["hot_side"]["mass_flow"] = m_total
    cooler["cold_side"]["mass_flow"] = m_sink
    turbine["mass_flow"] = m_total
    compressor["mass_flow"] = m_total
    heat_source_pump["mass_flow"] = m_source
    heat_sink_pump["mass_flow"] = m_sink

    # Summary of components
    components = {
        "turbine": turbine,
        "compressor": compressor,
        "recuperator": recuperator,
        "heater": heater,
        "cooler": cooler,
        "heat_source_pump": heat_source_pump,
        "heat_sink_pump": heat_sink_pump,
    }

    # Compute energy balances
    for name, component in components.items():
        if component["type"] == "heat_exchanger":
            hot = component["hot_side"]
            cold = component["cold_side"]
            Q1 = hot["mass_flow"] * (hot["state_in"].h - hot["state_out"].h)
            Q2 = cold["mass_flow"] * (cold["state_out"].h - cold["state_in"].h)
            component["heat_flow_rate"] = Q1
            component["heat_flow_rate_"] = Q2
            component["heat_balance"] = Q1 - Q2
            component["power"] = 0.0

        if component["type"] in ["compressor", "turbine"]:
            component["power"] = component["mass_flow"] * component["specific_work"]
            component["heat_flow_rate"] = 0.00

    # First-law analysis
    Q_in = heater["heat_flow_rate"]
    Q_out = cooler["heat_flow_rate"]
    W_out = turbine["power"]
    W_comp = compressor["power"]
    W_aux = heat_source_pump["power"] + heat_sink_pump["power"]
    W_in = W_comp + W_aux
    Q_in_max = m_source * (heater["hot_side"]["state_in"].h - source_out_min.h)
    cycle_efficiency = (W_out - W_in) / Q_in
    system_efficiency = (W_out - W_in) / Q_in_max
    backwork_ratio = W_comp / W_out
    energy_balance = (Q_in + W_comp) - (W_out + Q_out)  # Ignore pumps

    # Define dictionary with 1st Law analysis
    energy_analysis = {
        "heater_heat_flow": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "recuperator_heat_flow": recuperator["heat_flow_rate"],
        "cooler_heat_flow": Q_out,
        "turbine_power": W_out,
        "compressor_power": W_comp,
        "heat_source_pump_power": heat_source_pump["power"],
        "heat_sink_pump_power": heat_sink_pump["power"],
        "net_cycle_power": W_net,
        "net_system_power": W_out - W_in,
        "mass_flow_heating_fluid": m_source,
        "mass_flow_working_fluid": m_total,
        "mass_flow_cooling_fluid": m_sink,
        "cycle_efficiency": cycle_efficiency,
        "system_efficiency": system_efficiency,
        "backwork_ratio": backwork_ratio,
        "energy_balance": energy_balance,
    }

    # Evaluate objective function and constraints
    output = {"components": components, "energy_analysis": energy_analysis}
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq = utilities.evaluate_constraints(output, constraints)
    # f = None
    # c_eq = None
    # c_ineq = None

    # Set colors for plotting
    heater["hot_side"]["color"] = COLORS_MATLAB[6]
    heater["cold_side"]["color"] = COLORS_MATLAB[1]
    recuperator["hot_side"]["color"] = COLORS_MATLAB[1]
    recuperator["cold_side"]["color"] = COLORS_MATLAB[1]
    cooler["hot_side"]["color"] = COLORS_MATLAB[1]
    cooler["cold_side"]["color"] = COLORS_MATLAB[0]
    turbine["color"] = COLORS_MATLAB[1]
    compressor["color"] = COLORS_MATLAB[1]

    # Check if any fixed parameter or design variable was not used
    utilities.check_for_unused_keys(parameters, "parameters", raise_error=True)
    utilities.check_for_unused_keys(variables, "variables", raise_error=True)

    # # Summary of components
    # components = {
    #     "turbine": turbine,
    #     "compressor": compressor,
    #     # "recuperator": recuperator,
    #     # "heater": heater,
    #     # "cooler": cooler,
    #     # "heat_source_pump": heat_source_pump,
    #     # "heat_sink_pump": heat_sink_pump,
    # }

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
