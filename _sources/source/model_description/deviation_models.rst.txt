.. _deviation_models:

Deviation models
======================

Accurately predicting exit flow angles is essential for turbine performance analysis because they have a significant
influence on the velocity triangles and work output from the turbine. In general, the relative flow angle at the exit of a
cascade is not exactly equal to the metal angle because the blades cannot guide the flow perfectly. This discrepancy is
known as the deviation angle, which is formally defined as the difference between a reference geometric angle (:math:`\beta_g`) and the
relative flow angle (:math:`\beta`) at the exit of the cascade:

.. math::

    \delta = \beta_g - \beta 

The follwoing approaches to model deviation are presented here:

    - :ref:`zero_deviation`
    - :ref:`ainley_mathieson`
    - :ref:`aungier`

.. _zero_deviation:

Zero deviation
------------------------------

The simplest way to model deviation is to assume zero deviation:

.. math::

    \beta = \beta_g

.. _ainley_mathieson:     

Ainley and Mathieson
----------------------

The deviation model presented by Ainley and Mathieson (:cite:`ainley_method_1951`) assumes constant deviation at low speed (mach :math:`<` 0.5), and a linear 
deviation between mach = 0.5 and mach = 1.0 (critical point). This result in sharp edges at the points where mach = 0.5 and mach = 1.0. In addition, the model
assumes that the critical state occur when mach = 1.0 at the throat, which can lead to unphysical behaviour of the simulated performance (:cite:`anderson_method_2022`).
To correct for this behaviour, the model is adjusted to be interpolated between mach = 0.5 and the critical mach number.  

Note that his model defines the gauing angle with respect to axial direction:

.. math::
    \beta_g = \cos^{-1}(A_\mathrm{throat} / A_\mathrm{out})

- For :math:`\mathrm{Ma_exit} < 0.50` (low-speed), the deviation is a function of the gauging angle:

.. math::
    \delta_0 = \beta_g - (35.0 + \frac{80.0-35.0}{79.0-40.0}\cdot (\beta_g-40.0))

- For :math:`0.50 \leq \mathrm{Ma_exit} < \mathrm{Ma_crit}` (medium-speed), the deviation is calculated by a linear
    interpolation between low and critical Mach numbers:

.. math::
    \delta = \delta_0\cdot \left(1+\frac{0.5-\mathrm{Ma_exit}}{\mathrm{Ma_crit}-0.5}\right)

- For :math:`\mathrm{Ma_exit} \geq \mathrm{Ma_crit}` (supersonic), zero deviation is assumed:

.. math::
    \delta = 0.00

The flow angle (:math:`\beta`) is then computed based on the deviation and the gauging angle:

.. math::
    \beta = \beta_g - \delta

.. _aungier:

Aungier
------------------------------

The deviation model presented by Aungier (:cite:`aungier_turbine_2006`) assumes constant deviation at low speed (Mach :math:`<` 0.5), 
and a fifth order polynomial between mach = 0.5 and mach = 1.0 (critical point). This ensures a smooth evolution of the flow angle with the mach number. 
However, also this model assumes that the critical state occur when mach = 1.0 at the throat. To correct this behaviour, the model is adjusted to be interpolated
between mach = 0.5 and the critical mach number.

Note that his model defines the gauging angle with respect to tangential axis:

.. math::

    \beta_g = 90 - \cos^{-1}\left(\frac{A_\mathrm{throat}}{A_\mathrm{out}}\right)

- For :math:`Ma_\mathrm{exit} < 0.50`, the deviation is a function of the gauging angle:

.. math::

    \delta_0 = \sin^{-1}\left(\frac{A_\mathrm{throat}}{A_\mathrm{out}} \left(1+\left(1-\frac{A_\mathrm{throat}}{A_\mathrm{out}}\right)\cdot\left(\frac{\beta_g}{90}\right)^2\right)\right)

- For :math:`0.50 \leq Ma_\mathrm{exit} < Ma_\mathrm{crit}`, the deviation is calculated by a fifth order interpolation between low and critical Mach numbers:

.. math::
    \begin{align*}
    X &= \frac{2\cdot Ma_\mathrm{exit}-1}{2\cdot Ma_\mathrm{crit}-1} \\
    \delta &= \delta_0 \cdot (1-10X^3+15X^4-6X^5)
    \end{align*}

- For :math:`Ma_\mathrm{exit} \geq Ma_\mathrm{crit}`, zero deviation is assumed:

.. math:: 
    \delta = 0.00

The flow angle (:math:`\beta`) is then computed based on the deviation and the gauging angle:

.. math::
    \beta = 90 - \beta_g - \delta



