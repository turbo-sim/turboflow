
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
shock waves. The work of [Denton1993]_ presents a detailed description of 
loss mechanisms in turbomachinery.

Perhaps, the most popular loss model for axial turbines is the one proposed 
by [Ainley1951a, Ainley1951b]_ and its subsequent refinements by [Dunham1970]_ 
and [Kacker1982]_. The Kacker--Okapuu loss model has been further refined to 
account for off-design performance by [Moustapha1990]_ and by [Benner1997]_. 
One remarkable aspect of the Ainley--Mathieson family of loss methods is that 
it has been updated with new experimental data several times since the first 
version of the method was published. This was not the case for other loss 
prediction methods such as the ones proposed by [Balje1968a]_, [Craig1971]_, 
[Traupel1982]_, or [Aungier2006]_.

In this work, the [Kacker1982]_ loss model was adopted because of its 
popularity and maturity. The improvements of this loss model to account for 
off-design performance were not considered because the proposed methodology is 
aimed towards the optimization of the turbine performance at its design point. 
The Kacker--Okapuu loss model is described in detail in Appendix: KO-loss-model. 
The original formulation of the loss model has been adapted to the nomenclature 
and sign conventions used in this work for the convenience of the reader.

As described by [Denton1993]_ and by [Dahlquist2008]_, there are several 
possible definitions for the loss coefficient. In this work, the stagnation 
pressure loss coefficient was used because the Kacker--Okapuu loss model was 
developed based on this definition. This loss coefficient is meaningful for 
stator or rotor cascades with a constant mean radius and it is defined as the 
ratio of relative stagnation pressure drop across the cascade to relative 
dynamic pressure at the outlet of the cascade,

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

