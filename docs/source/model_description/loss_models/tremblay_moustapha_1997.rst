.. _loss_model_mkt1990:

Moustapha-Kacker-Tremblay (1990)
=======================================

This section describes the loss model proposed by :cite:`tremblay_off-design_1990` to compute aerodynamic losses in axial 
turbines.

Overview of the method
----------------------

The loss system by :cite:`tremblay_off-design_1990`, builds on the previous loss models proposed 
by :cite:`ainley_method_1951, kacker_mean_1982`.

The general form of the loss model is given by:

.. math::
    
    \mathrm{Y_{tot}} = \mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{s}} + \mathrm{Y_{cl}} + \mathrm{Y_{inc}}

The expressions used to compute each term as a function of the cascade geometry and flow variables 
are presented in the next sections. 

Profile loss coefficient
-------------------------

The profile loss coefficient (:math:`\mathrm{Y_{p}}`) is calculated as described in :ref:`profile_loss_KO`.

Secondary loss coefficient
--------------------------

:cite:`tremblay_off-design_1990` introduced a correction factor  (:math:`\mathrm{Y_{corr}}`) the secondary loss coefficient proposed by :cite:`kacker_mean_1982`,
to correct for incidence. The corrrection factor depends on an incidence parameter calculated as:

.. math::

    \chi = \frac{\beta_\mathrm{in} - \theta_\mathrm{in}}{180 - (\theta_\mathrm{in} + \theta_\mathrm{out})}\left(\frac{\cos(\theta_\mathrm{in})}{\cos(\theta_\mathrm{out})}\right)^{-1.5}\left(\frac{\mathrm{d_{le}}}{c}\right)^{-0.3}

where:

    - :math:`\chi` is the secondary flow incidence parameter.
    - :math:`c` is the chord length.
    - :math:`\mathrm{d_{le}}` is the leading edge diameter.
    - :math:`\theta_\mathrm{in}` is the inlet metal angle.
    - :math:`\theta_\mathrm{out}` is the exit metal angle.
    - :math:`\beta_\mathrm{in}` is the inlet relative flow angle.

From this parameter, the correction factor coefficient can be determined:

.. math::

    Y_\mathrm{corr} = \begin{cases}
    e^{0.9\chi} + 13\chi^2 + 400\chi^4 & \text{if } \chi \geq 0 \\
    e^{0.9\chi} & \text{if } \chi < 0
    \end{cases}

The total loss coefficient is then calculated as:

.. math::

    \mathrm{Y_{s}} = \mathrm{Y_{corr}}\mathrm{Y_{s, KO}}

where :math:`\mathrm{Y_{s, KO}}` is the secondary loss coefficient calculated as described in :ref:`secondary_loss_KO`.

Tip clearance loss coefficient
------------------------------

The tip clearance loss coefficient (:math:`\mathrm{Y_{cl}}`) is calculated as described in :ref:`tip_clearance_KO`.

Trailing edge loss coefficient
------------------------------

The trailing edge loss coefficient (:math:`\mathrm{Y_{te}}`) is calculated as described in :ref:`trailing_edge_KO`.

Incidence loss coefficient
------------------------------

The incidence loss coefficient (\mathrm{Y_{inc}}) is based on the kinetic energy loss coefficient. The calculation
involves to determine the increment in profile loss coefficient due to incidence.
The increment in profile losses due to incidence is based on the incidence parameter :math:`\chi`.
The formula used to compute :math:`\chi` is given by:

.. math::

    \chi = \left(\frac{\mathrm{d_{le}}}{s}\right)^{-1.6}\left(\frac{\cos{\theta_\mathrm{in}}}{\cos{\theta_\mathrm{out}}}\right)^{-2}(\beta_\mathrm{in} - \beta_\mathrm{des})

where:

    - :math:`\mathrm{d_{le}}` is the leading edge diameter.
    - :math:`s` is the pitch.
    - :math:`\theta_\mathrm{in}` and :math:`\theta_\mathrm{out}` is the blade metal angle at the inlet and outlet respectively.
    - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{des}` is the inlet relative flow angle at given and design conditions respectively.


Depending on the value of :math:`\chi`, two equations are used for computing the increment in profile losses:

.. math::

    \Delta\phi^2_p = \begin{cases}
                    -5.1734e^{-6}\chi + 7.6902e^{-9}\chi^2 & \text{if } -800 \leq \chi \leq 0 \\
                    0.778e^{-5}\chi + 0.56e^{-7}\chi^2 + 0.4e^{-10}\chi^3 + 2.054e^{-19}\chi^6 & \text{if } 0 \leq \chi \leq 800
                    \end{cases}

Subsequently, the increment in profile loss coefficient, in terms of kinetic energy loss, is converted to the total pressue loss coefficient definition:

.. math::

    \mathrm{Y} = \frac{\left(1-\frac{\gamma -1}{2}\mathrm{Ma_{out}}^2(\frac{1}{(1-\Delta\phi^2_p)}-1)\right)^\frac{-\gamma}{\gamma - 1}-1}{1-\left(1 + \frac{\gamma - 1}{2}\mathrm{Ma_{out}}^2\right)^\frac{-\gamma}{\gamma - 1}}

where:

    - :math:`\gamma` is the specific heat ratio.
    - :math:`\mathrm{Ma_{out}}` is the cascade exit relative mach number.
    - :math:`\Delta\phi^2_p` is the kinetic energy loss coefficient increment due to incidence.
