
.. _loss_models:

Loss models
===============

For meanline modelling, it is common to rely on empirical 
correlations to estimate the losses within the turbomachinery. These sets of 
correlations are known as loss models. This section presents the implemented loss models
for each turbomachinery configuration. 

Axial turbines
---------------

Perhaps, the most popular loss model for axial turbines is the one proposed 
by :cite:`ainley_method_1951,ainley_examination_1951` and its subsequent refinements by 
:cite:`dunham_improvements_1970` 
and :cite:`kacker_mean_1982`. The Kacker--Okapuu loss model has been further refined to 
account for off-design performance by :cite:`moustapha_improved_1990`. The model was further improved in subsequent studies, 
culminating in enhanced loss correlations for profile losses at off-design incidence :cite:`benner_influence_1997`
and endwall losses at design incidence :cite:`benner_influence_2004`. Furthermore, the authors introduced a 
new formulation for the total loss coefficient using the concept of penetration depth of secondary flows, 
aiming to enhance the prediction of endwall losses based on the underlying flow physics :cite:`benner_empirical_2006, benner_empirical_2006`.

The loss models implemented in the meanline code is further described in each separate section:

.. toctree::
   :maxdepth: 1

   loss_models/kacker_okapuu_1982
   loss_models/benner_sjolander_moustapha_2006
   loss_models/tremblay_moustapha_1997

