.. _configuration_options_short:

Configuration Options 
===================================

This section describes the different configuration options available for various submodels and simulations in the project.

.. contents::
    :local:
    :depth: 1

.. dropdown:: Loss Models
    :animate: fade-in-slide-down

    These inputs are valid for `simulation_options.loss_model.model`:

    - `benner`: see :ref:`loss_model_bsm2006` for model details.
    - `kacker_okapuu`: see :ref:`loss_model_ko1982` for model details.
    - `moustapha`: see :ref:`loss_model_mkt1990` for model details.
    - `isentropic`: Ensures that the loss coefficient is zero.
    - `custom`: Implies a constant loss coefficent, specified through `custom_value` input in configuration file.

.. dropdown:: Deviation Models
    :animate: fade-in-slide-down

    These inputs are valid for `simulation_options.deviation_model`:

    - `aungier`: see :ref:`aungier` for model details.
    - `ainley_mathieson`:  see :ref:`ainley_mathieson` for model details.
    - `zero_deviation`: see :ref:`zero_deviation` for model details.


.. dropdown:: Choking Models
    :animate: fade-in-slide-down

    These inputs are valid for `simulation_options.deviation_model`:

    - `evaluate_cascade_critical`: see :ref:`evaluate_cascade_critical` for model details.
    - `evaluate_cascade_throat`: see :ref:`evaluate_throat` for model details.
    - `evaluate_cascade_isentropic_throat`: see :ref:`isentropic_throat` for model details.


.. dropdown:: Objective Functions
    :animate: fade-in-slide-down

    These inputs are valid for `simulation_options.deviation_model`:

    - `efficieny_ts`: Total-to-static efficiency.
    - `none`: Returns 0 as objective function. Will simply attempt to satisfy constraints

.. dropdown:: Constraints
    :animate: fade-in-slide-down

    These inputs are valid for `design_optimization.constraints`. Remember to add
    type (`<`, `>` or `=`), `lower_bound` and `upper_bound`:

    - `mass_flow_rate`: Mass flow rate.
    - `interstage_flaring`: Inlet area to ext area ratio inbetween cascades.

.. dropdown:: Radius Types
    :animate: fade-in-slide-down

    These inputs are valid for `design_optimization.radius_type`:

    - `constant_hub`: Generates turbine with constant hub radius.
    - `constant_mean`: Generates turbine with constant mean radius.
    - `constant_tip`: Generates turbine with constant tip radius.
