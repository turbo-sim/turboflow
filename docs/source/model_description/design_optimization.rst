.. _design_optimization:

Design optimization
======================

Description of how to optimize the turbine geometry

- Modeling equations same as for performance analysis
- Independent variables extended with a subset of dimensionless geometry parameters
- Design variables are all dimensionless and well-scaled
- Constraints added to ensure feasible design
- Bounds for the design variables
- Objective function defined by the user (total-to-static) efficiency
- Optimization with gradient based algorithms
- Extension to multi-point optimization problems

For design optimization, the same model is adopted, ensuring constency with performance analysis mode. However, a set of geometrical parameters are available 
as design variables, together with the set :math:`x`. The residual equations are implemented as equality constraints, while an objective function is specified 
(such as total-to-static efficiency) to obtain the optimal geometry of the turbine. 


.. _design_optimization_single_point:

Single-point optimization
------------------------------

To be completed.


.. _design_optimization_single_point:


Multi-point optimization
----------------------------------

To be completed.

.. _design_optimization_multi_point:
