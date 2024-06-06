.. _loss_model_ko1982:

Kacker-Okapuu (1982)
=======================================


This section describes the loss model proposed by :cite:`kacker_mean_1982` to compute aerodynamic losses in axial 
turbines.

Overview of the method
----------------------

The :cite:`kacker_mean_1982` loss system is a refinement of the correlations proposed by
:cite:`ainley_method_1951`, :cite:`ainley_examination_1951`, and by :cite:`dunham_improvements_1970`. 
The general form of the loss model is given by:

.. math::

    Y = f_{\mathrm{Re}} \,  f_{\mathrm{Ma}} \, Y_{\mathrm{p}} + Y_{\mathrm{s}} + Y_{\mathrm{cl}} + Y_{\mathrm{te}}

The expressions used to compute each term as a function of the cascade geometry and flow variables 
are presented in the next sections. Some of the signs from the original correlations were modified 
to comply with the angle conventions used in this work. These modifications are explicitly mentioned in the text.



Reynolds number correction factor
---------------------------------

The term :math:`f_{\mathrm{Re}}` accounts for the effects of the Reynolds number and it is computed
according to the following equation:

.. math::

   f_{\mathrm{Re}} =
   \begin{cases}
   (\frac{\mathrm{Re}}{2 \cdot 10^5} )^{-0.40} & {\text{for }} \mathrm{Re} < 2 \cdot 10^5\\
   1 & {\text{for }} 2 \cdot 10^5 < \mathrm{Re} < 1 \cdot 10^6\\
   ( \frac{\mathrm{Re}}{1 \cdot 10^6})^{-0.20}  & {\text{for }} \mathrm{Re} > 1 \cdot 10^6 
   \end{cases}

where the Reynolds number is given in terms of the chord length
and the density, viscosity, and relative velocity at the outlet of the cascade:

.. math::

   \mathrm{Re} = \frac{\rho_{\mathrm{out}} \, w_{\mathrm{out}} \, c}{\mu_{\mathrm{out}}}

Mach number correction factor
-----------------------------

The term :math:`f_{\mathrm{Ma}}` accounts for losses associated with supersonic flows at the trailing
edge of the blades and it is computed by:

.. math::

   f_{\mathrm{Ma}} =
   \begin{cases}
   1 & {\text{for }}  \mathrm{Ma_{\,out}^{\, rel}} \leq 1 \\
   1+60\cdot(\mathrm{Ma_{\,out}^{\, rel}}-1)^2 & {\text{for }} \mathrm{Ma_{\,out}^{\, rel}} > 1 
   \end{cases}

The Mach number is defined by the relative velocity and the speed
of sound at the outlet of the cascade:

.. math::

   \mathrm{Ma_{\,out}^{\, rel}} =  w_{\mathrm{out}}/a_{\mathrm{out}}


.. _profile_loss_KO:

Profile loss coefficient
-------------------------

The profile loss coefficient :math:`Y_{\mathrm{p}}` is computed according to:

.. math::

   Y_{\mathrm{p}} = 0.914 \cdot \left( \frac{2}{3} \cdot  Y_{\mathrm{p}}' \cdot K_{\mathrm{p}} + Y_{\mathrm{shock}} \right).

The term :math:`Y_{\mathrm{p}}'` is given by:

.. math::

   Y_{\mathrm{p}}' = \left[ Y_{\mathrm{p, \, reaction}} - \left( \frac{\theta_{\mathrm{in}}}{\beta_{\mathrm{out}}} \right)  \left| \frac{\theta_{\mathrm{in}}}{\beta_{\mathrm{out}}} \right| \cdot (Y_{\mathrm{p, \, impulse}}- Y_{\mathrm{p, \, reaction}}) \right] \cdot \left(\frac{t_{\mathrm{max}}/c}{0.20}\right)^{-\frac{\theta_{\mathrm{in}}}{\beta_{\mathrm{out}}}}

where the terms, :math:`Y_{\mathrm{p, \, reaction}}` and :math:`Y_{\mathrm{p, \, impulse}}` are be obtained
from graphical data by :cite:`aungier_turbine_2006`. 
The subscript *reaction* refers to blades with zero inlet metal angle (i.e., axial entry) and the
subscript *impulse* refers to blades that have an inlet metal angle with the same magnitude but 
opposite sign as the exit relative flow angle.

The second term of the right-hand side is a correction factor that accounts
for the effects of the maximum blade thickness. In this equation, the sign of :math:`\beta_{\mathrm{out}}` was changed with 
respect to the original work of Kacker--Okappu to comply with the angle convention used in this work.

The factor :math:`K_{p}` accounts for compressible flow effects when the Mach 
number within the cascade is subsonic and approaches unity. These effects tend to accelerate the flow, 
make the boundary layers thinner, and decrease the profile losses. :math:`K_{p}` is a function on the
inlet and outlet relative Mach numbers and it is computed from the following set of equations:

.. math::

   K_{\mathrm{p}} = 1-K_{2} \cdot \left(1-K_{1} \right)

.. math::

   K_{1} =
   \begin{cases}
   1 & \text{for } \mathrm{Ma_{\,out}^{\, rel}} < 0.20 \\
   1-1.25 \cdot(\mathrm{Ma_{\,out}^{\, rel}}-0.20) & \text{for } 0.20 <\mathrm{Ma_{\,out}^{\, rel}} < 1.00 \\
   0 & \text{for } \mathrm{Ma_{\,out}^{\, rel}} > 1.00 \\
   \end{cases}

.. math::

   K_{2} = \left( \frac{\mathrm{Ma_{\,in}^{\, rel}}}{\mathrm{Ma_{\,out}^{\, rel}}} \right)^2

The term :math:`Y_{\mathrm{shock}}` accounts for the relatively weak shock waves
that may occur at the leading edge of the cascade due to the acceleration of the flow. After some algebra, 
the equations proposed in the Kacker--Okapuu method can be condensed to: 

.. math::

   Y_{\mathrm{shock}}  = 0.75 \cdot \left(f_{\mathrm{hub}} \cdot\mathrm{Ma_{\,in}^{\, rel}} -0.40 \right)^{1.75} \cdot \left( \frac{r_{\mathrm{hub}}}{r_{\mathrm{tip}}} \right)_{\mathrm{in}} \cdot \left( \frac{p_{\mathrm{0rel,in}}-p_{\mathrm{in}}}{p_{\mathrm{0rel,out}}-p_{\mathrm{out}}} \right)

where :math:`f_{\mathrm{hub}}` is given graphically in :cite:`kacker_mean_1982` and it is a function of the hub-to-tip ratio only.

.. _secondary_loss_KO:

Secondary loss coefficient
--------------------------

The secondary loss coefficient :math:`Y_{\mathrm{s}}` is computed according to:

.. math::

   Y_{\mathrm{s}} = 1.2 \cdot K_{\mathrm{s}} \cdot \left[0.0334 \cdot f_{\mathrm{AR}} \cdot Z \cdot \left( \frac{\cos(\beta_{\mathrm{out}})}{\cos(\theta_{\mathrm{in}})} \right) \right]

The factor 1.2 is included to correct the secondary loss for blades with zero trailing edge thickness. 
Trailing edge losses are accounted independently.

The factor :math:`K_{\mathrm{s}}` accounts for compressible flow effects present when the Mach number
approaches unity. These effects tend to accelerate the flow, make the end wall boundary layers thinner,
and decrease the secondary losses. :math:`K_{\mathrm{s}}` is computed from:

.. math::

   K_{\mathrm{s}} = 1-K_{3} \cdot \left(1-K_{\mathrm{p}} \right)

where :math:`K_{\mathrm{p}}` is given similarly as described in :ref:`profile_loss_KO` and :math:`K_{3}` is given as 
a function of the axial aspect ratio :math:`H/b` only:

.. math::

   K_{3} = \left(\frac{1}{H/b}\right)^2

:math:`f_{\mathrm{AR}}` accounts for the blade aspect ratio :math:`H/c` and it is given by:

.. math::

   f_{\mathrm{AR}} =
   \begin{cases}
   \frac{1-0.25\cdot \sqrt{2-H/c}}{H/c} & \text{for } H/c < 2\\
   \frac{1}{H/c} & \text{for } H/c > 2 
   \end{cases}

The Ainley-Mathieson loading parameter :math:`Z` is given by the following set of equations:

.. math::

   Z = \left(\frac{C_{\mathrm{L}}}{s/c}\right)^2 \, \frac{\cos(\beta_{\mathrm{out}})^2}{\cos(\beta_{\mathrm{m}})^3}

.. math::

   \left(\frac{C_{\mathrm{L}}}{s/c}\right) = 2 \cos(\beta_{\mathrm{m}}) \, \left[\tan(\beta_{\mathrm{in}}) - \tan(\beta_{\mathrm{out}})\right]

.. math::

   \tan(\beta_{\mathrm{m}}) = \frac{1}{2}\left[\tan(\beta_{\mathrm{in}}) + \tan(\beta_{\mathrm{out}})\right]

where the sign of :math:`\beta_{\mathrm{out}}` was changed with respect
to the original work to comply with the angle convention used in this work.

.. _tip_clearance_KO:

Tip clearance loss coefficient
------------------------------

The tip clearance loss coefficient :math:`Y_{\mathrm{cl}}` is computed according to:

.. math::

   Y_{\mathrm{cl}} = B \cdot Z \cdot \left(\frac{c}{H}\right) \cdot \left( \frac{t_{\mathrm{cl}}}{H}\right)^{0.78}

In this equation, :math:`Z` is given similarly as decribed in :ref:`secondary_loss_KO`. 
The Kacker-Okapuu loss system proposes :math:`B=0.37` for rotor blades with shrouded tips,
and :math:`B=0.00` for stator blades. In addition, Kacker and Okapuu warn that using :math:`B=0.47`, 
as suggested by :cite:`dunham_improvements_1970`, over-predicts the loss for rotor blades with plain tips.

.. _trailing_edge_KO:

Trailing edge loss coefficient
------------------------------

The trailing edge loss coefficient :math:`Y_{\mathrm{te}}` is computed according to:

.. math::

   Y_{\mathrm{te}} \approx \zeta = \frac{1}{\phi^2}-1 = \frac{1}{1-\Delta \phi^2}-1

Where the pressure loss coefficient :math:`Y` was approximated by the enthalpy loss coefficient
:math:`\zeta` and then related to the kinetic energy loss coefficients :math:`\phi^2` and
:math:`\Delta \phi^2`. See the work by :cite:`dahlquist_investigation_2008` for details about the definitions of the
different loss coefficients and the relations among them. The parameter :math:`\Delta \phi^2`
is computed by interpolation of impulse and reaction blades according to:

.. math::

   \Delta \phi^2 = \Delta \phi_{\mathrm{reaction}}^2 - \left( \frac{\theta_{\mathrm{in}}}{\beta_{\mathrm{out}}} \right)  \left| \frac{\theta_{\mathrm{in}}}{\beta_{\mathrm{out}}} \right| \cdot ( \Delta \phi_{\mathrm{impulse}}^2 -\Delta \phi_{\mathrm{reaction}}^2)

The sign of :math:`\beta_{\mathrm{out}}` was changed with respect to 
the original work of Kacker--Okappu to comply with the angle convention used in this work.
:math:`\Delta \phi_{\mathrm{reaction}}^2` and :math:`\Delta \phi_{\mathrm{impulse}}^2` are the kinetic
energy loss coefficients of reaction and impulse blades and they are a function of the trailing edge 
thickness to opening ratio :math:`t_{\mathrm{te}}/o` only. The functional relation was published in 
graphical form.

Final remarks
-------------

The Kacker--Okapuu loss model was developed to estimate the performance of 
competent turbine designs and its predictions will not be accurate if the input 
parameters are outside the range of the experimental data used to develop the 
correlations.

This situation is often encountered before the optimization algorithm converges 
since, in general, it is not possible to satisfy constraints for each iterate of 
a nonlinear programming problem.

For this reason, some of the variables used within the Kacker--Okapuu loss model 
were bounded to avoid numerical problems that might prevent the convergence to 
a feasible solution. For instance, some variables were forced to be non-negative 
because the correlations were not developed to cover such cases.

These modifications do not affect the final results of the optimization and they 
are not reported in this paper although they are documented in detail within the 
code.