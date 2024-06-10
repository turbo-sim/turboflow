

.. _installation:

Installation
============

User Installation Guide
-----------------------

This guide will walk you through the process of installing `Turboflow` via `pip`. To isolate the Turboflow installation and avoid conflicts with other Python packages, it is recommended to create a dedicated Conda virtual environment.

1. Ensure conda is installed:

   - Check if conda is installed in your terminal:

   .. code-block:: bash

      conda list

   - If installed packages do not appear, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

2. Open a terminal or command prompt and create a new virtual environment named ``turboflow_env`` with Python 3.11:

   .. code-block:: bash

      conda create --name turboflow_env python=3.11

3. Activate the newly created virtual environment:

   .. code-block:: bash

      conda activate turboflow_env

4. Install Turboflow using pip within the activated virtual environment:

   .. code-block:: bash

      pip install turboflow

5. Verify the installation by running the following command in your terminal:

   .. code-block:: bash

      python -c "import turboflow; turboflow.print_package_info()"

   If the installation was successful, you should see the Turboflow banner and package information displayed in the console output.

Congratulations! You have now successfully installed Turboflow in its own Conda virtual environment using pip. You're ready to start using Turboflow in your Python projects.

Installing Additional Solvers
-----------------------------

By default, ``turboflow`` can use the optimization solvers available in the ``scipy`` package. However, a wider range of solvers are available through the ``pygmo`` wrapper, including `IPOPT <https://coin-or.github.io/Ipopt/>`_ and `SNOPT <https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html>`_. Follow these steps to install additional solvers:

1. Activate your Turboflow Conda environment:

   .. code-block:: bash

      conda activate turboflow_env

2. Install the ``pygmo`` package via Conda (currently not available via pip):

   .. code-block:: bash

      conda install -c conda-forge pygmo

3. To access the ``SNOPT`` solver through the pygmo wrapper, you need to install the `pygmo_plugins_nonfree <https://ccom.ucsd.edu/~optimizers/solvers/snopt/>`_ package. Additionally, you need a local installation of the solver and a valid license. Follow these steps to set it up:

   - Install the `pygmo_plugins_nonfree package <https://esa.github.io/pagmo_plugins_nonfree/>`_ via Conda:

     .. code-block:: bash

        conda install -c conda-forge pygmo_plugins_nonfree

   - Download the SNOPT solver from the `official source <https://ccom.ucsd.edu/~optimizers/solvers/snopt/>`_ and obtain a valid license.
   - Extract the downloaded files to a directory of your choice.
   - Open your ``.bashrc`` file and add the following environment variables:

     .. code-block:: bash

        # Set SNOPT directory
        export SNOPT_DIR="/path/to/snopt/directory"
        export PATH="$PATH:$SNOPT_DIR"
        export SNOPT_LIB="$SNOPT_DIR/snopt7.dll"
        export SNOPT_LICENSE="$SNOPT_DIR/snopt7.lic"

     Replace ``/path/to/snopt/directory`` with the actual path to your ``SNOPT`` directory.
     These environment variables allow SNOPT to locate the license file and Turboflow to find the ``SNOPT`` binary.

   - Save the changes to your ``.bashrc`` file.
   - Restart your terminal or run ``source ~/.bashrc`` to apply the changes to the environment variables.
   - When installing SNOPT on Windows, your operating system may lack some "Dynamic Link Libraries" (DLLs) required by SNOPT. If this is the case, you can use the `Dependencies <https://lucasg.github.io/Dependencies/>`_ tool to identify which DLLs are missing. To resolve this issue:

     1. Download the `Dependencies <https://lucasg.github.io/Dependencies/>`_ tool.
     2. Run the tool and open the ``snopt7.dll`` library file.
     3. The tool will display a list of missing DLLs, if any.
     4. Download the missing DLLs from a reliable source.
     5. Place the downloaded DLLs in the same directory as the ``snopt7.dll`` library.

By following these steps, you can install additional solvers like IPOPT and SNOPT through the pygmo wrapper, expanding the range of optimization algorithms available for solving performance analysis and optimization problems in turboflow.
