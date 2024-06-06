.. _equation_formulation:

Equation formulation and solution
=======================================

In this section, an overview of the model description and how the equation are assembled 
and solved are presented. The model description sections presents a set of equations to evaluate
velocity triangles, fluid properties, deviation, losses and to evaluate choking. But how are 
these equation used throughout the model? For this model formulation, the fluid properties and velocity triangles are evaluated at both the inlet and 
exit plane of each cascade in the turbine. The throat section is either evaluated only at critical
conditions or also evlauated similarly as the exit plane, dependent on the choking model (see :ref:`choking_models` for more details).
The losses are evaluated at the exit plane for each cascade, in addition to the throat (also here at either 
critical conditions or evaluated simliarly as the exit plane). Similar applies to the deviation, which is also evaluated at the exit
and may be applied to the throat, depending on the choking model. 

To evalaute the equations, a set of inputs are required, which also vary somewhat for different choking models. 
For the `evaluate_critical_cascade` the following set of input is required for each cascade (except for :math:`v_{\mathrm{in}}`):

.. math::

     x = [
    v_{\mathrm{in}} ,\,
    w_{\mathrm{out}} ,\,
    s_{\mathrm{out}} ,\,
    \beta_{\mathrm{out}} ,\,
    v^*_{\mathrm{in}} ,\,
    w^*_{\mathrm{throat}} ,\,
    s^*_{\mathrm{throat}}
    ]

where:

    - :math:`v` and :math:`w` are the absolute and relative velocity.
    - :math:`\beta` is the relative flow angle.
    - :math:`s` is entropy. 
    - Superscript :math:`*` denotes chhoked state property.
    - Subscript dentoes the respective plane.

When this set if input is used to evaluate the model equations, the output may be inconsistent. The mass flow rates for different
planes and cascades may differ, the loss coefficient error may not be zero, the relative flow angles may be different from the 
deviation models and the calculated exit pressure may not match the one specified by the boundary conditions. This raises 
a set of residual equations that is required to be solved, in order for the model to be consistent. For example, for the `evaluate_critical_cascade`
this set of equations will look like this:

.. math::

    R(x) = [
    \Delta p_\mathrm{out} ,\,
    \Delta \dot{m}_\mathrm{out} ,\,
    \Delta Y_\mathrm{out} ,\,
    \Delta \beta_\mathrm{out} ,\,
    \Delta \mathcal{L}_\mathrm{throat}^* ,\,
    \Delta \dot{m}_\mathrm{throat}^* ,\,
    \Delta Y_\mathrm{throat}^*
    ] = 0,

where: 

    - :math:`\Delta p_\mathrm{out}` refer to the exit pressure error. 
    - :math:`\Delta \dot{m}` refer to differences in mass flow rates.
    - :math:`\Delta Y` refer to loss coefficient errors.
    - :math:`\Delta \beta` refer to discrepancy between flow input flow angle and deviation model. 
    - :math:`\Delta \mathcal{L}_\mathrm{throat}^*` refer to error in component of the gradient of the Lagrange function (see :ref:`evaluate_cascade_critical`).

In this way, the model present an equation-oriented problem formulation, where the equations can be solved using gradient-based solvers for both performance analysis and design optimization. 

