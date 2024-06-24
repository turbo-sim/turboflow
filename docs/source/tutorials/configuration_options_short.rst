.. _configuration_options_short:

Configuration Options 
===================================

This section describes the different configuration options available in TurboFlow.

.. contents::
    :local:
    :depth: 1

.. dropdown:: Loss Models
    :animate: fade-in-slide-down

    These are the valid loss model options:

    - `benner`: see :ref:`loss_model_bsm2006` for model details.
    - `kacker_okapuu`: see :ref:`loss_model_ko1982` for model details.
    - `moustapha`: see :ref:`loss_model_mkt1990` for model details.
    - `isentropic`: Ensures that the loss coefficient is zero.
    - `custom`: Implies a constant loss coefficent, specified through `custom_value` input in configuration file.

.. dropdown:: Deviation Models
    :animate: fade-in-slide-down

    These are the valid deviation model options:

    - `aungier`: see :ref:`aungier` for model details.
    - `ainley_mathieson`:  see :ref:`ainley_mathieson` for model details.
    - `zero_deviation`: see :ref:`zero_deviation` for model details.

.. dropdown:: Choking Models
    :animate: fade-in-slide-down

    These are the valid choking model options:

    - `evaluate_cascade_critical`: see :ref:`evaluate_cascade_critical` for model details.
    - `evaluate_cascade_throat`: see :ref:`evaluate_throat` for model details.
    - `evaluate_cascade_isentropic_throat`: see :ref:`isentropic_throat` for model details.


.. dropdown:: Objective Functions
    :animate: fade-in-slide-down

    The objective function is defined by specifying a variable, type and scale. The objective function is fetched from the `results` dictionary that stores all calculated variables, and 
    the variable name must be on the form `key.column`, where `key` is a key in `results`, and `column` is the column header in the DataFrame contained in `results[key]`. With `overall` as key, you can access the most 
    common objective functions such as:
    
    - `efficiency_ts`
    - `efficiency_tt`
    - `power`

    The scale is simply a nonzero float or integer, and is used to scale the objective function. This gives the user control to tune the behaviour of the optimization algorithm.
    The type must be either `maximize` or `minimize`. If the type is `maxmimize`, the sign of the scale changes to ensure that the objective function is in fact maximized. 

.. dropdown:: Constraints
    :animate: fade-in-slide-down

    Each constraint is defined by specifying variable name, value, type and normalize. Similar as for objective function, the constraints are fetched from the `results` dictionary that stores all calculated variables, and 
    the variable name must be on the form `key.column`, where `key` is a key in `results`, and `column` is the column header in the DataFrame contained in `results[key]`. Here are some example of valid keys:

    - `overall` to access variables such as mass flow rate (`overall.mass_flow_rate`)
    - `geometry` to access geometrical variables such as flaring angle (`geometry.flaring_angle`).
    - `plane` to access variables such as relative mach number (`plane.Ma_rel`)
    - `additional_constraints` to access additional variables such as interspace area ratio (`additional_constraints.interspace_area_ratio`)

    Type specifies if the constraint is an equaility or inequality constraint, and must be either `<`, `>` or `=`. 
    Value specifies the value that the variables should be equal to, less than or greater then, and normalize specifies if the constraint should be normalized with the value or not. 

.. dropdown:: Radius Types
    :animate: fade-in-slide-down

        These are the valid radius types:

    - `constant_hub`: Generates turbine with constant hub radius.
    - `constant_mean`: Generates turbine with constant mean radius.
    - `constant_tip`: Generates turbine with constant tip radius.
