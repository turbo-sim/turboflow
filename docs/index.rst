

Welcome to Meanline Axial's documentation
===========================================

``meanline_axial`` is a Python package for mean-line modelling and simulation of axial turbines. It provides a systematic approach for performance analysis and design optimization of axial turbines:


- **Performance Analysis Mode**:

  - Evaluate single operating points
  - Produce performance maps

- **Design Optimization Mode**:

  - Single point optimization
  - Multi-point optimization

- **Problem formulation and solution**:

  - Equation-oriented problem formulation for performance analysis and design optimization
  - Consistency between both calculation modes is guaranteed by design
  - Efficient solution with gradient-based root-finding and optimization solvers
  - Multi-start strategies or derivative-free optimizers for global optimization

- **Fluid Property Analysis**:

  - Use CoolProp to determine real gas fluid properties.

- **Design Flexibility**:

  - Supports modelling with any number of turbine stages.
  - Specify turbine geometry using main geometric parameters.

- **Choking Calculations**:

  - General computational strategy to evaluate cascade choking.
  - Formulation autonomously identifies choked cascades for a set of operating conditions

- **Loss Models**:

  - Kacker Okapuu model.
  - Benner model.


Contents:
==========
.. toctree::
   :maxdepth: 2

   source/installation
   source/tutorials
   source/model_description
   source/model_validation
   source/nomenclature
   source/glossary
   source/bibliography
   source/configuration
   source/api/meanline_axial





   
