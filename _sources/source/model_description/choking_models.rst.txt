.. _choking_models:

Choking models
======================

The model can predict choked flow in three different ways. The prinicples for the different methods are explained in the following. 

Isentropic throat model
------------------------------

This model predict choked flow by evaluating the throat of the cascade, and set the mach number to be equal to the exit plane at subsonic conditions, and equal to one at supersonic conditions:  

.. math::

    \mathrm{Ma_{throat}} = \begin{cases}
                            \mathrm{Ma_{exit}} & \text{if } \mathrm{Ma_{exit}}<1 \\
                            1  & \text{if } \mathrm{Ma_{exit}} \geq 1
                            \end{cases}

When the flow through a cascade is isentropic, the flow chokes when the mach reaches unity at the throat:

.. math::

    \left(1-\text{Ma}^2\right)\frac{dv}{v} = -\frac{dA}{A} + \frac{1}{2}(1+G) \,\text{Ma}^2   \left[C_f \left(\frac{P_\text{w}}{A} \right) \,\text{d}x \right]

where:

    - :math:`\mathrm{Ma}` is the mach number.
    -  :math:`v` is the velocity.
    - :math:`A` is the flow area.
    - :math:`G` is the Grüneiser parameter.
    - :math:`P_\mathrm{w}` is the wetted peremiter. 
    - :math:`C_f` is the skin friction. 
    - :math:`x` is the cascade flow length. 

When the term to the right is zero (isentropic), the mach number is unity for an accelerating flow at the most narrow section (:math:`dA = 0`). In order for this model to be consitent, the throat entropy must be set equal to 
the entropy at the inlet of the plane. However, when assuming an isentropic throat, the critical mass flow rate will be overestimated. 


.. _evaluate_cascade_critical:

Calculate critical state model
------------------------------

This model predict choked flow by calculating the maximum mass flow rate at the throat section of the cascade, and thereby the critical state at the throat, while consdering losses at the throat. 
When the exit plane mach number is less than the critical, the flow angle is settled by a deviation model, while at supercritical flow, the flow angle is calculated from the critical mass flow rate:

.. math::

    \beta_\text{out} =
    \begin{cases}
        \beta_\text{deviation model} & \text{if } \text{Ma}_\text{out}<\text{Ma}^* \\
        \arccos{\left(\frac{\dot{m}^*}{\rho_\text{out} \, w_\text{out} \, A_\text{out}}\right)} & \text{if } \text{Ma}_\text{out} \ge \text{Ma}^*.
    \end{cases}

where 

    - :math:`\beta` is the relative flow angle.
    - :math:`\mathrm{Ma}` is the relative mach number. 
    - :math:`\dot{m}` is the mass flow rate. 
    - :math:`\rho` is the fluid density.
    - :math:`w` is the relative velocity.
    - :math:`A` is the flow area.
    - Subscript :math:`\mathrm{out}` refer to the exit plane.
    - Superscript :math:`^*` refer to the critical state. 

Evaluate cascade throat model
------------------------------

This model predict choked flow by evaluating the throat of the cascade, and set the flow angle at the exit plane according to the deviation model at subsonic conditions, while at supersonic conditions the
the throat mach number equals the critical mach number:

.. math::

    \begin{align}
        & \beta_\text{out} = \beta_\text{deviation model} \text{    } && \text{if } \text{Ma}_\text{out}<\text{Ma}^* \\
        & \mathrm{Ma_{throat}} = \text{Ma}^* && \text{if } \text{Ma}_\text{out} \ge \text{Ma}^*
    \end{align}


The critical mach number is calculated from a correlation for the critical mach number as a function of the loss coefficient. The correlation is developed using an algebraic nozzle model to calulate the critical mach
number for different values of the loss coefficient, and use linear regression. 

.. math::



