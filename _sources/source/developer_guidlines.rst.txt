.. _developer_guide:

Developer guidlines
=======================

Thank you for considering contributing to this project! Here are some guidelines to help you get started:

Developer Installation Guide
----------------------------

This guide is intended for developers who wish to contribute to or modify the Turboflow source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

1. Clone the Turboflow remote repository:

   .. code-block:: bash

      git clone https://github.com/turbo-sim/TurboFlow.git

2. Create a dedicated Conda virtual environment for Turboflow development:

   .. code-block:: bash

      conda create --name turboflow_env python=3.11

3. Activate the newly created virtual environment:

   .. code-block:: bash

      conda activate turboflow_env

4. Install Poetry to manage dependencies:

   .. code-block:: bash

      conda install poetry

   Poetry is a powerful dependency manager that offers separation of user and developer dependencies, ensuring that only the necessary packages are installed based on the user's intent. Additionally, it simplifies the process of adding, updating, and removing dependencies, making it easier to maintain the project's requirements.

5. Use Poetry to install the required dependencies for Turboflow development:

   .. code-block:: bash

      poetry install

6. Verify the installation by running the following command:

   .. code-block:: bash

      python -c "import turboflow; turboflow.print_package_details()"

   If the installation was successful, you should see the Turboflow banner and package information displayed in the console output.

Pull request guidelines
-------------------------

Please follow these steps to submit a pull request.

1. **Fork the repository**: Click the "Fork" button on the top right of this page.
2. **Clone your fork**: `git clone https://github.com/turbo-sim/TurboFlow`
3. **Create a branch**: `git checkout -b feature-name`
4. **Make your changes**: Implement your feature or bugfix.
5. **Commit your changes**: `git commit -m "Description of changes"`
6. **Push to your fork**: `git push origin feature-name`
7. **Open a pull request**: Go to your fork on GitHub and click the "New pull request" button.

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