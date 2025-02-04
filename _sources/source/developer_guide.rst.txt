.. _developer_guide:

Developer guide
=======================

Thank you for considering contributing to this project! Here are some guidelines to help you get started:

Developer Installation Guide
----------------------------

This installation guide is intended for developers who wish to contribute to or modify the Turboflow source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

1. **Fork the repository:**

   - Navigate to the `project's GitHub page <https://github.com/turbo-sim/TurboFlow>`_.
   - Click the "Fork" button in the upper right corner of the repository page to create a copy of the repository under your own GitHub account.


2. **Clone the forked repository:**

   - Open your terminal.
   - Run the following command, replacing `<your-username>` with your GitHub username:

   .. code-block:: bash

      git clone https://github.com/<your-username>/<repository-name>.git

   - Navigate into the cloned repository:

   .. code-block:: bash

      cd <repository-name>

3. **Create a dedicated Conda virtual environment for TurboFlow development**:

   - Check that conda is installed:

   .. code-block:: bash

      conda list

   - If not conda is installed, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.
   - Create dedicated virtual environment for turboflow package:

   .. code-block:: bash

      conda env create --name turboflow_env python=3.11

4. **Activate the newly created virtual environment**:

   .. code-block:: bash

      conda activate turboflow_env

5. **Install Poetry to manage dependencies**:

   .. code-block:: bash

      conda install poetry 

   Poetry is a powerful dependency manager that offers separation of user and developer dependencies, ensuring that only the necessary packages are installed based on the user's intent. Additionally, it simplifies the process of adding, updating, and removing dependencies, making it easier to maintain the project's requirements.

6. **Use Poetry to install the required dependencies for TurboFlow development**:

   .. code-block:: bash

      poetry install --with=dev

7. **Verify the installation by running the following command**:

   .. code-block:: bash

      python -c "import turboflow; turboflow.print_package_info()"

   If the installation was successful, you should see the Turboflow banner and package information displayed in the console output.


Pull request guidelines
-------------------------

Please follow these steps to submit a pull request.

1. **Create a branch in your forked repository**:

   - Open your terminal in the projects root.
   - Create branch:

   .. code-block:: bash

      git checkout -b <feature-name>

2. **Make your changes**:

   - Implement your feature or bugfix.


3. **Commit your changes**:

   .. code-block:: bash 

      git commit -m "Description of changes"

4. **Push to your fork**: 

   .. code-block:: bash

      git push origin feature-name

5. **Open a pull request**: 

   - Go to your fork on GitHub and click the "New pull request" button.

Testing guidlines
-------------------

When implementing new features or adding new submodels, you should aim to create unit tests that verifies the functionality of what you are adding. 
This testing framework is designed to streamline the process of testing and validating the project using a centralized script to manage tests. 
The framework includes functions for running tests and generating regression data, and are contained in the `tests` folder:

.. code-block:: text

    tests/
    ├── config_files/       # Configuration files for tests
    │   ├── performance_analysis_ainley_mathieson.yaml
    │   ├── design_optimization.yaml
    │   └── ...
    │
    ├── regression_data_linux/ # Regression data for tests in linux based os
    │   ├── performance_analysis_ainley_mathieson_linux.xlsx
    │   ├── design_optimization_linux.xlsx
    │   └── ...
    ├── regression_data_windows/ # Regression data for tests in windows
    │   ├── performance_analysis_ainley_mathieson_windows.xlsx
    │   ├── design_optimization_windows.xlsx
    │   └── ...
    │
    ├── __init__.py                    # Make 'tests' a package
    |── tests_manager.py               # Centralized test management
    ├── generate_regression_data.py    # Script to generate regression data
    ├── test_performance_analysis.py   # Test function for performance analysis
    ├── test_design_optimization.py    # Test function for design optimization
    └── ....                         

The principle of the framwork is that `test_performance_analysis` and `test_design_optimization` runs performance analysis and design optimization for a subset of the configuration files 
in `config_files`. The outcome of these simulations are compared to the respective data in `regression_data_windows` and `regression_data_linux`, depending on the OS the code is executed from. 
The configuration files used for performance analysis and design optimization tests are determined in `tests_manager`, through the dictionary `TEST_CONFIGS`:

.. code-block:: python

   TEST_CONFIGS = {
        'performance_analysis': ["performance_analysis_ainley_mathieson.yaml"],
        'design_optimization': ["design_optimization.yaml"],
    }

Add new tests
^^^^^^^^^^^^^^

To add new tests, follow these steps:

1. **Add Configuration Files:**

   Place the new configuration file in the `config_files` folder.

2. **Update `config_manager.py`:**

   Add your new configuration file with its respective test function in `TEST_CONFIGS` in `tests_manager.py`:
   
   .. code-block:: python

      TEST_CONFIGS = {
         'performance_analysis': ["performance_analysis_ainley_mathieson.yaml",
                                 "performance_analysis_new_test.yaml"],
         'design_optimization': ["design_optimization.yaml",
                              "design_optimization_new_test.yaml"],
      }

3. **Generate Regression Data:**

   Generate regression data according to `these steps <generate_regression_data_>`_.

4. **Check that the tests pass**

   Check that the test pass by following `these steps <run_tests_>`_.

.. _generate_regression_data:

Generate regression data
^^^^^^^^^^^^^^^^^^^^^^^^^^
When adding new tests, or making code changes that affect the output of existing tests, new regression files must be generated. 
Follow these steps to add regression data.

1. **Update `REGRESSION_CONFIGS` in `tests_manager.py:**

   Include the configuration files for which you want to generate new regression data:

   .. code-block:: python

      REGRESSION_CONFIGS = {
      'performance_analysis': ["performance_analysis_new_test.yaml"],
      'design_optimization': ["design_optimization_new_test.yaml"],
      }

2. **Generate regression data:**

   Run the following command in the root directory of the project:
   
   .. code-block:: bash 

      python tests/generate_regression_data.py


The new regression data should now be within its respective `regression_data` folder. 
The new regression data must be added to both `regression_data_linux` and `regression_data_windows`. For the OS you do not have access to, 
you can use github actions workflow to generate regression data for other OS. See `Generate Regression Data` workflow in 
`gitub repository <https://github.com/turbo-sim/TurboFlow/actions/workflows/generate_regression_data_ubuntu.yaml>`_. The output of this workflow
is saved as a github artifact and must be downloaded and added to the respective regression data folder manually. 

Make sure that the new tests pass with the new regression files by following `these steps <run_tests_>`_.

.. _run_tests:

Run tests
^^^^^^^^^^

The tests are executed locally using pytest. To run the tests, open the terminal in the root directory of the project and run the following command:

.. code-block:: bash

   python pytest

This command will run all scripts starting with `test_` in the `tests` folder, with the corresponding configuration files specified in `test_manager.py`.
A summary of the tests will be printed, and show if the tests passed or not. 


Reporting issue
----------------

If you find a bug or have a feature request, please open an issue in the Github project page and follow the provided templates.

CI/CD Pipeline
--------------

Turboflow uses GitHub Actions to automate its Continuous Integration and Continuous Deployment (CI/CD) processes.

Automated Testing
^^^^^^^^^^^^^^^^^

The ``ci.yml`` action is triggered whenever a commit is pushed to the repository. This action runs the test suite on both Windows and Linux environments, ensuring the code's compatibility and correctness across different platforms.

Package Publishing
^^^^^^^^^^^^^^^^^^

Turboflow utilizes the ``bumpversion`` package to manage versioning and release control. To increment the version number, use the following command:

.. code-block:: bash

   bumpversion patch  # or minor, major

After bumping the version, push the changes to the remote repository along with tags to signify the new version:

.. code-block:: bash

   git push origin --tags

If the tests pass successfully, the package is automatically published to the Python Package Index (PyPI), making it readily available for users to install and use.

Documentation Deployment
^^^^^^^^^^^^^^^^^^^^^^^^

Turboflow automates the deployment of documentation using the ``deploy_docs`` action. This action builds the Sphinx documentation of the project and publishes the HTML files to GitHub Pages each time that a new commit is pushed to the remote repository. By automating this process, Turboflow ensures that the project's documentation remains up-to-date and easily accessible to users and contributors.