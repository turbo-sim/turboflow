
.. _loss_coefficients:

Loss coefficients
=========================

Loss model
----------

During the preliminary design phase, it is common to rely on empirical 
correlations to estimate the losses within the turbine. These sets of 
correlations are known as loss models. A loss can be interpreted as any 
mechanism that leads to entropy generation and reduces the power output of 
the turbine, such as viscous friction in boundary layers, wake mixing, or 
shock waves. The work of :cite:`denton_loss_1993`  presents a detailed description of 
loss mechanisms in turbomachinery.



Loss coefficient definitions


The losses are characterized in terms of the loss coefficient
This parameter has different possible definitions




Perhaps, the most popular loss model for axial turbines is the one proposed 
by :cite:`ainley_method_1951,ainley_examination_1951` and its subsequent refinements by 
:cite:`dunham_improvements_1970` 
and :cite:`kacker_mean_1982`. The Kacker--Okapuu loss model has been further refined to 
account for off-design performance by :cite:`moustapha_improved_1990` and by 
:cite:`benner_influence_1997`. 
One remarkable aspect of the Ainley--Mathieson family of loss methods is that 
it has been updated with new experimental data several times since the first 
version of the method was published. This was not the case for other loss 
prediction methods such as the ones proposed by :cite:`balje_axial_1968`, :cite:`craig_performance_1970`, 
:cite:`traupel_thermische_2001`, or :cite:`aungier_turbine_2006`.

In this work, the :cite:`kacker_mean_1982` loss model was adopted because of its 
popularity and maturity. The improvements of this loss model to account for 
off-design performance were not considered because the proposed methodology is 
aimed towards the optimization of the turbine performance at its design point. 
The Kacker--Okapuu loss model is described in detail in Appendix: KO-loss-model. 
The original formulation of the loss model has been adapted to the nomenclature 
and sign conventions used in this work for the convenience of the reader.

As described by :cite:`denton_loss_1993` and by :cite:`dahlquist_investigation_2008`, there are several 
possible definitions for the loss coefficient. In this work, the stagnation 
pressure loss coefficient was used because the Kacker--Okapuu loss model was 
developed based on this definition. This loss coefficient is meaningful for 
stator or rotor cascades with a constant mean radius and it is defined as the 
ratio of relative stagnation pressure drop across the cascade to relative 
dynamic pressure at the outlet of the cascade:


.. math::
   
   Y =  \frac{p_{\mathrm{0rel,in}}-p_{\mathrm{0rel,out}}}
   {p_{\mathrm{0rel,out}}-p_{\mathrm{out}}}

When the proposed axial turbine model is evaluated, the loss coefficient 
computed from its definition, as given above, and the loss coefficient 
computed using the loss model, as described in Appendix: KO-loss-model, may not 
have the same value. In the optimization problem section, the turbine design is 
formulated as an optimization problem that relies on equality constraints to 
ensure that the value of both loss coefficients are consistent for each 
cascade. Therefore, the loss coefficient error is given by

.. math::

   Y_{\mathrm{error}} =  Y_{\mathrm{definition}}  -  
   Y_{\mathrm{loss\,model}}





.. There are several definitions of the loss coefficient commonly used to characterize the irreversibility within blade cascades. The meanline model described in this paper can employ any of these definitions, which are provided here for reference. Throughout these definitions, the subscript "0" signifies the stagnation state for stator cascades and the relative stagnation state for rotor cascades.

.. The **stagnation pressure loss coefficient** is defined as the reduction in stagnation pressure from inlet to outlet relative to the dynamic pressure at the cascade's exit:
.. $$Y=\frac{p_{0, \mathrm{in}}-p_{0, \mathrm{out}}}{p_{0, \mathrm{out}} - p_\mathrm{out}}$$

.. The **kinetic energy loss coefficient** is the ratio of the enthalpy increase due to irreversibility to the isentropic total-to-static enthalpy change:
.. $$\Delta \phi^2  =  \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{h_{0,\mathrm{out}}-h_{\mathrm{out},s}} = \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{\frac{1}{2}v_{\mathrm{out},s}^2} =1 - \left(\frac{v_{\mathrm{out}}}{v_{\mathrm{out},s}}\right)^2 =  1- \phi^2$$
.. Here, $\phi^2$ is the ratio of actual to ideal kinetic energy at the cascade's exit, commonly interpreted as the efficiency of the cascade.

.. The **enthalpy loss coefficient** is analogously defined, but it utilizes the actual total-to-static enthalpy change in the denominator:
.. $$ \zeta=\frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{h_{0,\mathrm{out}}-h_{\mathrm{out}}} = \frac{h_{\mathrm{out}}-h_{\mathrm{out},s}}{\frac{1}{2}v_{\mathrm{out}}^2} = \left(\frac{v_{\mathrm{out},s}}{v_{\mathrm{out}}  }\right)^2 - 1 = \frac{1}{\phi^2}-1$$

.. The **entropy loss coefficient** is the product of exit temperature and the entropy increase across the cascade, divided by the kinetic energy at the cascade's exit:
.. $$\varsigma  = \frac{T_\mathrm{out}(s_{\mathrm{out}}-s_{\mathrm{in}})}{\frac{1}{2}v_{\mathrm{out}}^2} $$

.. These four loss coefficient definitions are interconnected as they are different way to quantify the irreversibility within a cascade. The kinetic energy and enthalpy loss coefficient definitions can be directly converted as follows:
.. $$\zeta = \frac{\Delta \phi^2}{1-\Delta \phi^2} \Longleftrightarrow \Delta \phi^2 = \frac{\zeta}{1+\zeta}$$
.. However, to transition between either of these two definitions and the entropy loss coefficient or the stagnation pressure loss coefficient it is necessary to know the inlet thermodynamic state and the pressure ratio across the cascade [citation].
