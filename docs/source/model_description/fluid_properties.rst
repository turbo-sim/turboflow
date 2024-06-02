.. _fluid_properties:

Fluid properties
===================

The model adopt CoolProp to calculate thermophysical properties of the given fluid, specifically,
enthalpy-entropy calls:

.. math::

    [\rho, \mu, p, T, a] = f(h, s).

To calculate the total property, the total enthalpy is adopted:

.. math::

    [\rho_0, \mu_0, p_0, T_0, a_0] = f(h_0, s).

Similarly, the total relative enthalpy is adopted to calculate the total relative property:

.. math::

    [\rho_{0, \mathrm{rel}}, \mu_{0, \mathrm{rel}}, p_{0, \mathrm{rel}}, T_{0, \mathrm{rel}}, a_{0, \mathrm{rel}}] = f(h_{0, \mathrm{rel}}, s).