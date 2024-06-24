
=========================================================
TurboFlow: Mean-Line Modelling of Axial Turbines
=========================================================

.. .. image:: https://example.com/logo.png
..    :alt: TurboFlow Logo
..    :align: center
..    :scale: 50 %

**Version:** 0.1.10  
**Author:** Lasse Borg Anderson & Roberto Agromayor  
**License:** MIT License

Overview
========

TurboFlow is a Python package for mean-line modelling of axial turbines. It aims to offer flexible and reliable simulations for both performance prediction and design optimization, 
and should present a valuable resource for engineers and researchers working in the field of turbomachinery.

Features
========

- **Performance Prediction:** Accurately predict the performance of axial turbines based on various input parameters.
- **Design Optimization:** Optimize preliminary turbine design to achieve optimal performance metrics.
- **Equation-oriented problem formulation:** Equation-oriented problem formulation for performance analysis and design optimization.
- **Model consitency:** The model is consistent for both performance prediction and design optimization.
- **Efficient solution:** The model adopts gradient-based root-finding and optimization solver.
- **Real gas fluid property analysis:** CoolProp is used to determine thermohpysical properties.
- **Flexible model:** The model offers options for submodels for loss, deviation and choking calculations.
- **General geometry:** Geometrical variables are defined to cover a wide range of axial turbine configurations, including multistage configurations.  
- **Easy-to-use:** Intuitive and easy setup of input parameters for rapid development and analysis. 
- **Extensive Documentation:** Comprehensive guides and examples to help you get started quickly.


Contents:
==========
.. toctree::
   :maxdepth: 1

   source/installation
   source/tutorials
   source/model_description
   source/developer_guidlines
   source/nomenclature
   source/glossary
   source/bibliography
   source/api_reference

.. note::
  The package is for now limited to axial-turbines, but is intended to cover other types of turbomachinery in future versions. 





   
