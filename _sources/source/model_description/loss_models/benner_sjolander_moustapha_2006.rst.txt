.. _loss_model_bsm2006:

Benner-Sjolander-Moustapha (2006)
=======================================

This section describes the loss model proposed by :cite:`benner_influence_1997, benner_empirical_2006, benner_empirical_2006-1` to compute aerodynamic losses in axial 
turbines.

Overview of the method
----------------------

The loss system follows from a series of publications :cite:`benner_influence_1997, benner_empirical_2006, benner_empirical_2006-1`
and builds on the previous loss models proposed by :cite:`ainley_method_1951, kacker_mean_1982, tremblay_off-design_1990`.

The general form of the loss model is given by:

.. math::

        \mathrm{Y_{tot}} = (\mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{inc}})(1-Z/H) + \mathrm{Y_{s}} + \mathrm{Y_{cl}}

The expressions used to compute each term as a function of the cascade geometry and flow variables 
are presented in the next sections. 

Profile loss coefficient
-------------------------

The profile loss coefficient (:math:`\mathrm{Y_{p}}`) is calculated as described in :ref:`profile_loss_KO`.

Secondary loss coefficient
--------------------------

The correlation for the secondary loss coefficient (:math:`\mathrm{Y_{s}}`) takes two different forms depending on the aspect ratio:

If aspect ratio :math:`\leq 2.0`

.. math::

    \mathrm{Y_{s}} = \frac{0.038 + 0.41 \tanh(1.20\delta^*/H)}{\sqrt{\cos(\xi)}CR(H/c)^{0.55}\frac{\cos(\beta_\mathrm{out})}{\cos(\xi)}^{0.55}}

wheras if aspect ratio :math:`>2.0`

.. math::

    \mathrm{Y_{s}} = \frac{0.052 + 0.56 \tanh(1.20\delta^*/H)}{\sqrt{\cos(\xi)}CR(H/c)\frac{\cos(\beta_\mathrm{out})}{\cos(\xi)}^{0.55}}

where:

    - :math:`CR = \frac{\cos(\beta_\mathrm{in})}{\cos(\beta_\mathrm{out})}` is the convergence ratio.
    - :math:`H` is the mean height.
    - :math:`c` is the chord.
    - :math:`\delta^*` is the inlet endwall boundary layer displacement thickness.
    - :math:`\xi` is the stagger angle.
    - :math:`\beta_\mathrm{out}` is the exit relative flow angle.

Tip clearance loss coefficient
------------------------------

The tip clearance loss coefficient (:math:`\mathrm{Y_{cl}}`) is calculated as described in :ref:`tip_clearance_KO`.

Trailing edge loss coefficient
------------------------------

The trailing edge loss coefficient (:math:`\mathrm{Y_{te}}`) is calculated as described in :ref:`trailing_edge_KO`.

Incidence loss coefficient
------------------------------

The incidence loss coefficient (:math:`\mathrm{Y_{inc}}`) is based on the kinetic energy loss coefficient. The calculation
involves to determine the increment in profile loss coefficient due to incidence.
The increment in profile losses due to incidence is based on the incidence parameter :math:`\chi`.
The formula used to compute :math:`\chi` is given by:

.. math::

    \chi = \frac{\mathrm{d_{le}}}{s}^{-0.05}\mathrm{We_{le}}^{-0.2}\frac{\cos\theta_\mathrm{in}}{\cos\theta_\mathrm{out}}^{-1.4}(\beta_\mathrm{in} - \beta_\mathrm{des})

where:

    - :math:`\mathrm{d_{le}}` is the leading edge diameter.
    - :math:`s` is the pitch.
    - :math:`\mathrm{We_{le}}` is the leading edge wedge angle.
    - :math:`\theta_\mathrm{in}` and :math:`\theta_\mathrm{out}` is the blade metal angle at the inlet and outlet respectively.
    - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{des}` is the inlet relative flow angle at given and design conditions respectively.

Depending on the value of :math:`\chi`, two equations are used for computing the increment in profile losses:

1. For :math:`\chi \geq 0`:

.. math::

    \Delta \phi_{p}^2 = \sum_{i=1}^{8} a_i \, \chi^i

with coefficients:

.. math::

    a_1 = -6.149 \times 10^{-5}  \\
    a_2 = +1.327 \times 10^{-3}  \\
    a_3 = -2.506 \times 10^{-4}  \\
    a_4 = -1.542 \times 10^{-4}  \\
    a_5 = +9.017 \times 10^{-5}  \\
    a_6 = +1.106 \times 10^{-5}  \\
    a_7 = -5.318 \times 10^{-6}  \\
    a_8 = +3.711 \times 10^{-7}

2. For :math:`\chi < 0`:

.. math::

    \Delta \phi_{p}^2 = \sum_{i=1}^{2} b_i \, \chi^i

with coefficients:

.. math::

    b_1 = -8.720e-4 \times 10^{-4}  \\
    b_2 = +1.358e-4 \times 10^{-4}

Subsequently, the increment in profile loss coefficient, in terms of kinetic energy loss, is converted to the total pressue loss coefficient definition:

.. math::

    \mathrm{Y} = \frac{\left(1-\frac{\gamma -1}{2}\mathrm{Ma_{out}}^2(\frac{1}{(1-\Delta\phi^2_p)}-1)\right)^\frac{-\gamma}{\gamma - 1}-1}{1-\left(1 + \frac{\gamma - 1}{2}\mathrm{Ma_{out}}^2\right)^\frac{-\gamma}{\gamma - 1}}

where:

    - :math:`\gamma` is the specific heat ratio.
    - :math:`\mathrm{Ma_{out}}` is the cascade exit relative mach number.
    - :math:`\Delta\phi^2_p` is the kinetic energy loss coefficient increment due to incidence.

    

Penetration depth
------------------

The penetration depth (:math:`Z`) refer to the spanwize penetration of the vortices originating from the cascade inlet endwall boundary layer. The penetration depth relative to hte mean 
blade height is calcluated by:

.. math::

    \frac{\mathrm{Z_{te}}}{H} = \frac{0.10F_t^{0.79}}{\sqrt{CR}\left(\frac{H}{c}\right)^{0.55}} + 32.70\frac{\delta^*}{H}^2

where:

    - :math:`CR = \frac{\cos(\beta_\mathrm{in})}{\cos(\beta_\mathrm{out})}` is the convergence ratio.
    - :math:`H` is the mean height.
    - :math:`c` is the chord.
    - :math:`\delta^*` is the inlet endwall boundary layer displacement thickness.
    - :math:`\beta_\mathrm{out}` is the exit relative flow angle.
    - :math:`F_t` is the tangential loading coefficient.

The tangiential loading coefficient is calculated by:

.. math::

        F_t = 2\frac{s}{c_\mathrm{ax}}\cos^2(\beta_m)(\tan(\beta_\mathrm{in} - \beta_\mathrm{out}))

where:

    - :math:`s` is the pitch.
    - :math:`c_\mathrm{ax}` is the axial chord.
    - :math:`\beta_m = \tan^{-1}(0.5(\tan(\beta_\mathrm{in}) + \tan(\beta_\mathrm{out})))` is the mean vector angle.
    - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{out}` is the inlet and outlet relative flow angle.