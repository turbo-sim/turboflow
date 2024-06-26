
.. _loss_coefficients:

Loss coefficients
=========================

A loss can be interpreted as any  mechanism that leads to entropy generation and reduces/increases the power output/consumption of 
the turbomahcinery, such as viscous friction in boundary layers, wake mixing, or 
shock waves (:cite:`denton_loss_1993`). In meanline analysis, the losses are characterized in terms of a loss coefficient.
This parameter has different possible definitions, which are provided here for reference. 

The **stagnation pressure loss coefficient** is defined as the reduction in stagnation pressure from inlet to outlet relative to a reference dynamic pressure. For
compressor blades, the inlet conditions are used for reference, while the exit conditions are used for turbine blades:

.. math::

   \mathrm{Y}=
   \begin{cases}
      \frac{p_{0, \mathrm{in}}-p_{0, \mathrm{out}}}{p_{0, \mathrm{in}} - p_\mathrm{in}}, & \text{for compressor blades} \\
      \frac{p_{0, \mathrm{in}}-p_{0, \mathrm{out}}}{p_{0, \mathrm{out}} - p_\mathrm{out}}, & \text{for turbine blades}.
   \end{cases}

Subscript `0` refer to the stagnation property in the relative frame of reference. 

The **kinetic energy loss coefficient** is the ratio of the enthalpy increase due to irreversibility to the isentropic total-to-static enthalpy change.
Also for this loss coefficient, the inlet conditions ar used for reference for compressore blades, and exit conditions for turbine blades:

.. math::

   \Delta \phi^2  =  
   \begin{cases}
      \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{h_{0,\mathrm{in}}-h_{\mathrm{in},s}} = \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{\frac{1}{2}v_{\mathrm{in},s}^2} =1 - \left(\frac{v_{\mathrm{out}}}{v_{\mathrm{in},s}}\right)^2 =  1- \phi^2, & \text{for compressor blades} \\
      \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{h_{0,\mathrm{out}}-h_{\mathrm{out},s}} = \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{\frac{1}{2}v_{\mathrm{out},s}^2} =1 - \left(\frac{v_{\mathrm{out}}}{v_{\mathrm{out},s}}\right)^2 =  1- \phi^2, & \text{for turbine blades}.
   \end{cases}

Here, :math:`\phi^2` is the ratio of actual to ideal kinetic energy at the cascade's exit, commonly interpreted as the efficiency of the cascade.

The **enthalpy loss coefficient** is analogously defined, but it utilizes the actual total-to-static enthalpy change in the denominator:

.. math::

   \zeta=
   \begin{cases}
      \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{h_{0,\mathrm{in}}-h_{\mathrm{in}}} = \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{\frac{1}{2}v_{\mathrm{in}}^2} = \left(\frac{v_{\mathrm{out},s}}{v_{\mathrm{in}}  }\right)^2 - 1 = \frac{1}{\phi^2}-1, & \text{for compressor blades} \\
      \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{h_{0,\mathrm{out}}-h_{\mathrm{out}}} = \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{\frac{1}{2}v_{\mathrm{out}}^2} = \left(\frac{v_{\mathrm{out},s}}{v_{\mathrm{out}}  }\right)^2 - 1 = \frac{1}{\phi^2}-1, & \text{for turbine blades}. 
   \end{cases}

The **entropy loss coefficient** is the product of exit temperature and the entropy increase across the cascade, divided by the kinetic energy at the cascade's exit:

.. math::

   \varsigma  = 
   \begin{cases}
      \frac{T_\mathrm{out}(s_{\mathrm{out}}-s_{\mathrm{in}})}{\frac{1}{2}v_{\mathrm{in}}^2}, & \text{for compressor blades} \\
      \frac{T_\mathrm{out}(s_{\mathrm{out}}-s_{\mathrm{in}})}{\frac{1}{2}v_{\mathrm{out}}^2}, & \text{for turbine blades}.
   \end{cases}

When the proposed model is evaluated, the loss coefficient 
computed from its definition, as given above, and the loss coefficient 
computed using the loss model, as described in :ref:`loss_models`, may not 
have the same value. Therefore, the loss coefficient error is given by

.. math::

   Y_{\mathrm{error}} =  Y_{\mathrm{definition}}  -  
   Y_{\mathrm{loss\,model}}

The evaluation of the model involves measures to ensure that the loss coefficient error is zero. For performance analysis, this is ensured by 
thr root-solver algorithm, while for design optimization this is implemented as an equality constraint for the same purpose. 

.. warning::
   Only the stagnation pressure loss coefficient is implemented in the current version of the code. 