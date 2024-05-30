
.. _installation:

Installation
===============
.. note:: 
   
   This is a temporary development solution. In the future the package will be installed via pip.


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
   
   


