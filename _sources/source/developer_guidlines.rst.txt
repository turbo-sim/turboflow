.. _developer_guide:

Developer guidlines
=======================

Thank you for considering contributing to this project! Here are some guidelines to help you get started:

Developer Installation Guide
----------------------------

This guide is intended for developers who wish to contribute to or modify the Turboflow source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

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

3. **Create a dedicated Conda virtual environment for Turboflow development**:

   - Check that conda is installed:

   .. code-block:: bash

      conda list

   - If not conda is installed, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.
   - Create dedicated vitrual environment for turboflow package:

   .. code-block:: bash

      conda create --name turboflow_env python=3.11

4. **Activate the newly created virtual environment**:

   .. code-block:: bash

      conda activate turboflow_env

5. **Install Poetry to manage dependencies**:

   .. code-block:: bash

      conda install poetry

   Poetry is a powerful dependency manager that offers separation of user and developer dependencies, ensuring that only the necessary packages are installed based on the user's intent. Additionally, it simplifies the process of adding, updating, and removing dependencies, making it easier to maintain the project's requirements.

6. **Use Poetry to install the required dependencies for Turboflow development**:

   .. code-block:: bash

      poetry install

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
TurboFlow adopt `pytest` testing framework, and the folder `tests` in the root repository host the test files. Here you can find the test files:

   - `test_performance_analysis`
   - `test_performance_analysis`

Both these function fetch configuration files from `config_files` folder inside to either conduct perfromance analysis or design optimization.
The output of these tests are compared with regression data contained in `regression_data_linux` or `regression_data_windows`, depending on the OS where the
code runs on. 

With this testing structure, follow these steps to add tests:

1. Add configuration file of the test you want to add to `config_files`.  
2. Add configuration file of the test in `CONFIG_FILES` in either or both `test_performance_analysis` or `test_performance_analysis`.
3. Generate regression data and add files to both `regression_data_linux` and `regression_data_linux`.

To add regression data, use the `generate_regression_data` script. Here you can define the `CONFIG_FILES_PERFORMANCE_ANALYSIS` and 
`CONFIG_FILES_DESIGN_OPTIMIZATION` which are list of configuration files for which you want to generate regression data. 
To add regression data for a linux based OS, you can run the `generate_regression_data_ubuntu` workflow from Github Actions. 
This stores the regression data files as artifacts, which can be downloaded and inserted in the regression data folder manually. 

Otherwise, a independent tests can be added in the `tests` folder. 

To run the test suite, simply run the following command from the root of the repository:

.. code-block:: bash

   python pytest

Reporting issue
----------------

If you find a bug or have a feature request, please open an issue and follow the provided templates.


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