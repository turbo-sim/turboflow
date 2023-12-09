
import copy
from .. import utilities
from .. import fluid_properties as props

from .cycle_components import (compression_process,
                               expansion_process,
                               heat_exchanger)

COLORS_MATLAB = utilities.COLORS_MATLAB


def evaluate_split_compression_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):
    
    # Create copies to not change the originals
    variables = copy.deepcopy(variables)
    parameters = copy.deepcopy(parameters)

    # Initialize fluid objects
    working_fluid = props.Fluid(**parameters.pop("working_fluid"))
    heating_fluid = props.Fluid(**parameters.pop("heating_fluid"))
    cooling_fluid = props.Fluid(**parameters.pop("cooling_fluid"))

    # Rename pressure drops
    dp_heater_h = parameters["heater"].pop("pressure_drop_hot_side")
    dp_heater_c = parameters["heater"].pop("pressure_drop_cold_side")
    dp_recup_lowT_h = parameters["recuperator_lowT"].pop("pressure_drop_hot_side")
    dp_recup_lowT_c = parameters["recuperator_lowT"].pop("pressure_drop_cold_side")
    dp_recup_highT_h = parameters["recuperator_highT"].pop("pressure_drop_hot_side")
    dp_recup_highT_c = parameters["recuperator_highT"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler"].pop("pressure_drop_cold_side")

    # Rename design variables
    turbine_inlet_p = variables.pop("turbine_inlet_pressure")
    turbine_inlet_h = variables.pop("turbine_inlet_enthalpy")
    main_compressor_inlet_p = variables.pop("main_compressor_inlet_pressure")
    main_compressor_inlet_h = variables.pop("main_compressor_inlet_enthalpy")
    split_compressor_inlet_h = variables.pop("split_compressor_inlet_enthalpy")
    recup_intermediate_h = variables.pop("recuperator_intermediate_enthalpy")
    mass_split_fraction = variables.pop("mass_split_fraction")
    heat_source_temperature_out = variables.pop("heat_source_exit_temperature")
    heat_sink_temperature_out = variables.pop("heat_sink_exit_temperature")
    
    # Evaluate main compressor
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_highT_c) * (1.0 - dp_recup_lowT_c)
    main_compressor_outlet_p = turbine_inlet_p / dp
    main_compressor_efficiency = parameters["main_compressor"].pop("efficiency")
    main_compressor_efficiency_type = parameters["main_compressor"].pop("efficiency_type")
    main_compressor = compression_process(
        working_fluid,
        main_compressor_inlet_h,
        main_compressor_inlet_p,
        main_compressor_outlet_p,
        main_compressor_efficiency,
        efficiency_type=main_compressor_efficiency_type,
    )

    # Evaluate re-compressor
    split_compressor_inlet_p = main_compressor_inlet_p / (1 - dp_cooler_h)
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_highT_c)
    split_compressor_outlet_p = turbine_inlet_p / dp
    split_compressor_efficiency = parameters["split_compressor"].pop("efficiency")
    split_compressor_efficiency_type = parameters["split_compressor"].pop("efficiency_type")
    split_compressor = compression_process(
        working_fluid,
        split_compressor_inlet_h,
        split_compressor_inlet_p,
        split_compressor_outlet_p,
        split_compressor_efficiency,
        split_compressor_efficiency_type,
    )

    # Evaluate turbine
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_lowT_h) * (1.0 - dp_recup_highT_h)
    turbine_outlet_p = main_compressor_inlet_p / dp
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
    
    # Evaluate low temperature recuperator
    h_in_hot = recup_intermediate_h
    p_in_hot = turbine_outlet_p * (1 - dp_recup_highT_h)
    h_out_hot = split_compressor_inlet_h
    p_out_hot = split_compressor_inlet_p
    h_in_cold = main_compressor["state_out"].h
    p_in_cold = main_compressor["state_out"].p
    h_out_cold = h_in_cold + (1 / (1 - mass_split_fraction)) * (h_in_hot - h_out_hot)
    p_out_cold = split_compressor_outlet_p
    num_elements = parameters["recuperator_lowT"].pop("num_elements")
    recuperator_lowT = heat_exchanger(
        working_fluid,
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

    # Evaluate high temperature recuperator
    h_in_cold = mass_split_fraction*split_compressor["state_out"].h + (1-mass_split_fraction) * recuperator_lowT["cold_side"]["state_out"].h
    h_out_cold = (turbine["state_out"].h - recup_intermediate_h) + h_in_cold
    p_in_cold = split_compressor["state_out"].p
    p_out_cold = turbine_inlet_p / (1 - dp_heater_c)
    h_in_hot = turbine["state_out"].h
    h_out_hot = recup_intermediate_h
    p_in_hot = turbine_outlet_p
    p_out_hot = p_in_hot * (1 - dp_recup_highT_h)
    num_elements = parameters["recuperator_highT"].pop("num_elements")
    recuperator_highT = heat_exchanger(
        working_fluid,
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

    # Evaluate heater
    h_in_cold = recuperator_highT["cold_side"]["state_out"].h
    p_in_cold = recuperator_highT["cold_side"]["state_out"].p
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

    # Evaluate cooler
    T_in_cold = parameters["heat_sink"].pop("inlet_temperature")
    p_in_cold = parameters["heat_sink"].pop("inlet_pressure")
    h_in_cold = cooling_fluid.set_state(props.PT_INPUTS, p_in_cold, T_in_cold).h
    p_out_cold = p_in_cold * (1 - dp_cooler_c)
    T_out_cold = heat_sink_temperature_out
    h_out_cold = cooling_fluid.set_state(props.PT_INPUTS, p_out_cold, T_out_cold).h
    h_in_hot = recuperator_lowT["hot_side"]["state_out"].h
    p_in_hot = recuperator_lowT["hot_side"]["state_out"].p
    h_out_hot = main_compressor["state_in"].h
    p_out_hot = main_compressor["state_in"].p
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

    # Compute performance metrics
    W_net = parameters.pop("net_power")
    turbine_work = turbine["specific_work"]
    main_compression_work = main_compressor["specific_work"]*(1-mass_split_fraction)
    split_compression_work = split_compressor["specific_work"]*(mass_split_fraction)
    m_total = W_net / (turbine_work - main_compression_work - split_compression_work)
    m_main = m_total * (1 - mass_split_fraction)
    m_split = m_total * mass_split_fraction
    m_heat = m_total * heater["mass_flow_ratio"]
    m_cold = m_main / cooler["mass_flow_ratio"]
    # W_net = parameters.pop("net_power")
    # m_w = W_net / (turbine["specific_work"] - compressor["specific_work"])

    Q_in = m_heat * (heater["hot_side"]["state_in"].h - heater["hot_side"]["state_out"].h)
    Q_out = -m_cold * (cooler["cold_side"]["state_in"].h - cooler["cold_side"]["state_out"].h)
    W_out = m_total * turbine["specific_work"]
    W_in = m_main * main_compressor["specific_work"] + m_split * split_compressor["specific_work"]
    # h_hot_min = heating_fluid.set_state(
    #     props.PT_INPUTS,
    #     parameters["heat_source"].pop("exit_pressure"),
    #     parameters["heat_source"].pop("exit_temperature_min"),
    # ).h
    # Q_in_max = m_h * (heater["hot_side"]["state_in"].h - h_hot_min)
    cycle_efficiency = W_net / Q_in
    # system_efficiency = W_net / Q_in_max
    backwork_ratio = W_in / W_out

    # Add the mass flow to the components
    heater["hot_side"]["mass_flow"] = m_heat
    heater["cold_side"]["mass_flow"] = m_total
    recuperator_lowT["hot_side"]["mass_flow"] = m_total
    recuperator_lowT["cold_side"]["mass_flow"] = m_main
    recuperator_highT["hot_side"]["mass_flow"] = m_total
    recuperator_highT["cold_side"]["mass_flow"] = m_total
    cooler["hot_side"]["mass_flow"] = m_main
    cooler["cold_side"]["mass_flow"] = m_cold
    turbine["mass_flow"] = m_total
    main_compressor["mass_flow"] = m_main
    split_compressor["mass_flow"] = m_split


    # Set colors for plotting
    heater["hot_side"]["color"] = COLORS_MATLAB[6]
    heater["cold_side"]["color"] = COLORS_MATLAB[1]
    recuperator_lowT["hot_side"]["color"] = COLORS_MATLAB[1]
    recuperator_lowT["cold_side"]["color"] = COLORS_MATLAB[1]
    recuperator_highT["hot_side"]["color"] = COLORS_MATLAB[1]
    recuperator_highT["cold_side"]["color"] = COLORS_MATLAB[1]
    cooler["hot_side"]["color"] = COLORS_MATLAB[1]
    cooler["cold_side"]["color"] = COLORS_MATLAB[0]
    turbine["color"] = COLORS_MATLAB[1]
    main_compressor["color"] = COLORS_MATLAB[1]
    split_compressor["color"] = COLORS_MATLAB[1]

    # Define dictionary with 1st Law analysis
    energy_analysis = {
        "heater_heat_flow": Q_in,
        # "heater_heat_flow_max": Q_in_max,
        "recuperator_heat_flow": Q_out,
        "cooler_heat_flow": Q_out,
        "turbine_power": W_out,
        "compressor_power": W_in,
        "net_power": W_net,
        "mass_flow_heating_fluid": m_heat,
        "mass_flow_working_fluid": m_total,
        "mass_flow_cooling_fluid": m_cold,
        "mass_flow_main": m_main,
        "mass_flow_split": m_split,
        "cycle_efficiency": cycle_efficiency,
        # "system_efficiency": system_efficiency,
        "backwork_ratio": backwork_ratio,
    }

    # Summary of processes
    components = {
        "turbine": turbine,
        "split_compressor": split_compressor,
        "main_compressor": main_compressor,
        "recuperator_lowT": recuperator_lowT,
        "recuperator_highT": recuperator_highT,
        "heater": heater,
        "cooler": cooler,
    }


    # for name, component in components.items():
    #     if component["type"] == "heat_exchanger":
    #         # print(name)
    #         Q1 = component["hot_side"]["mass_flow"]*(component["hot_side"]["state_in"].h - component["hot_side"]["state_out"].h)
    #         Q2 = component["cold_side"]["mass_flow"]*(component["cold_side"]["state_in"].h - component["cold_side"]["state_out"].h)
    #         print(name, Q1, Q2)


    output = {
        "components": components,
        "energy_analysis": energy_analysis,
    }

    # utilities.print_dict(energy_analysis)

    # # Evaluate objective function and constraints
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq = utilities.evaluate_constraints(output, constraints)
    # f = None
    # c_eq = None
    # c_ineq = None

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

    # # # At the end of the function, check for unused keys
    # utilities.check_for_unused_keys(parameters)
    # utilities.check_for_unused_keys(variables)


    # raise Exception("End of evaluation :)")

    # output = None
    return output

    # var = {"main_compressor_out": main_compressor["state_out"].p, 
    #        "recuperator_lowT_in": recuperator_lowT["cold_side"]["state_in"].p,
    #       "recuperator_lowT_out": recuperator_lowT["cold_side"]["state_out"].p,
    #       "recompressor_out": recompressor["state_out"].p,
    #       "recuperator_highT_in": recuperator_highT["cold_side"]["state_in"].p,
    #       "recuperator_highT_out": recuperator_highT["cold_side"]["state_out"].p,
    #       "turbine_in": turbine["state_in"].p}
        
    # var = {"turbine_out": turbine["state_out"].p, 
    #        "recuperator_highT_in": recuperator_highT["hot_side"]["state_in"].p,
    #        "recuperator_highT_out": recuperator_highT["hot_side"]["state_out"].p,
    #        "recuperator_lowT_in": recuperator_lowT["hot_side"]["state_in"].p,
    #        "recuperator_lowT_out": recuperator_lowT["hot_side"]["state_out"].p,
    #        "recompressor_in": recompressor["state_in"].p,
    #        "main_compressor_in": main_compressor["state_in"].p}
    



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

    # Rename pressure drops
    dp_heater_h = parameters["heater"]["pressure_drop_hot_side"]
    dp_heater_c = parameters["heater"]["pressure_drop_cold_side"]
    dp_recup_h = parameters["recuperator"]["pressure_drop_hot_side"]
    dp_recup_c = parameters["recuperator"]["pressure_drop_cold_side"]
    dp_cooler_h = parameters["cooler"]["pressure_drop_hot_side"]
    dp_cooler_c = parameters["cooler"]["pressure_drop_cold_side"]

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

    # Summary of processes
    components = {
        "turbine": turbine,
        "compressor": compressor,
        "recuperator": recuperator,
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


