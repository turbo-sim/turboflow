
.. _introduction:

Introduction
=============


``meanline_axial`` is a Python package for mean-line modelling and simulation of axial turbines. It provides a systematic approach for performance analysis and design optimization of axial turbines:

.. _overview:

Overview
------------
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


.. _installation:

Installation
-----------------
.. note:: 
   
   This is a temporary development solution. In the future the package will be installed via pip/conda.


1. **Install Conda**:

   Before proceeding, ensure you have `Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_ installed on your system.

2. **Create a virtual environment with dependencies**:

   Run the following command in `Bash <https://gitforwindows.org/>`_ to set up a new virtual environment with all the required dependencies:

   .. code-block:: bash

      conda env create --file environment.yaml

   This command will create a virtual environment named ``meanline_env`` and will install all the packages specified in the ``environment.yaml`` file.

3. **Activate the virtual environment**:

   .. code-block:: bash

      conda activate meanline_env

4. **Installing additional packages (optional)**:

   Packages can be installed using:

   .. code-block:: bash

      conda install <name of the package>

   Alternatively, you can add package names directly to the ``environment.yaml`` file and then update the environment:

   .. code-block:: bash

      conda env update --file environment.yaml --prune

5. **Setting up for local development**:

   To ensure you can import the package for local development, add the package directory to the ``PYTHONPATH`` variable. A convenient script named `install_local.py` is provided for this.

   Run:

   .. code-block:: bash

      python install_local.py

   This will append the current working directory (which should be the root of this repository) to your ``PYTHONPATH``.
   
   

.. _getting_started:

Getting started
-----------------

To be completed. Add demo showing how to use the code for a simple example, including:

- Performance analysis
- Design optimization

