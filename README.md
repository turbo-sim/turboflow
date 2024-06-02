
# TurboFlow: Axial Turbine Mean-line Modelling

[![PyPI](https://img.shields.io/pypi/v/turboflow.svg)](https://pypi.org/project/turboflow/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://turbo-sim.github.io/TurboFlow/)

**TurboFlow** is Python tool for meanline analysis and optimization of turbomachinery.


## Core features

- **Performance Analysis**: 
  - Evaluate single operating points
  - Produce performance maps
- **Design Optimization**: 
  - Single point optimization
  - Multi-point optimization
- **Problem formulation and solution**
  - Equation-oriented problem formulation for performance analysis and design optimization
  - Consistency between both calculation modes by construction
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
- **Turbomachinery architectures**:
  - Axial turbines
  - Radial turbines (to be implemented)
  - Axial compressors (to be implemented)
  - Centrifugal compressors (to be implemented)


## User Installation Guide

This guide will walk you through the process of installing `Turboflow` via `pip`. To isolate the Turboflow installation and avoid conflicts with other Python packages, it is recommended to create a dedicated Conda virtual environment.

1. [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) if you don't have it already.

2. Open a terminal or command prompt and create a new virtual environment named `turboflow_env` with Python 3.11:
   ```
   conda create --name turboflow_env python=3.11
   ```

3. Activate the newly created virtual environment:
   ```
   conda activate turboflow_env
   ```

4. Install Turboflow using pip within the activated virtual environment:
   ```
   pip install turboflow
   ```

5. Verify the installation by running the following command in your terminal:
   ```
   python -c "import turboflow; turboflow.print_package_details()"
   ```

   If the installation was successful, you should see the Turboflow banner and package information displayed in the console output.

Congratulations! You have now successfully installed Turboflow in its own Conda virtual environment using pip. You're ready to start using Turboflow in your Python projects.


## Installing Additional Solvers

By default, `turboflow` can use the optimization solvers available in the `scipy` package. However, a wider range of solvers are available through the `pygmo` wrapper, including [`IPOPT`](https://coin-or.github.io/Ipopt/) and [`SNOPT`](https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html). Follow these steps to install additional solvers:

1. Activate your Turboflow Conda environment:
   ```
   conda activate turboflow_env
   ```

2. Install the `pygmo` package via Conda (currently not available via pip):
   ```
   conda install pygmo
   ```

3. To access the `SNOPT` solver through the pygmo wrapper, you need to install the [`pygmo_plugins_nonfree`](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) package. Additionally you need a local installation of the solver and a valid license. Follow these steps to set it up:
   - Install the [pygmo_plugins_nonfree package](https://esa.github.io/pagmo_plugins_nonfree/) via Conda:
      ```
      conda install pygmo_plugins_nonfree
      ```
   - Download the SNOPT solver from the [official source](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) and obtain a valid license.
   - Extract the downloaded files to a directory of your choice.
   - Open your `.bashrc` file and add the following environment variables:

     ```bash
     # Set SNOPT directory
     export SNOPT_DIR="/path/to/snopt/directory"
     export PATH="$PATH:$SNOPT_DIR"
     export SNOPT_LIB="$SNOPT_DIR/snopt7.dll"
     export SNOPT_LICENSE="$SNOPT_DIR/snopt7.lic"
     ```

     Replace `/path/to/snopt/directory` with the actual path to your `SNOPT` directory.
     These environment variables allow SNOPT to locate the license file and Turboflow to find the `SNOPT` binary.

   - Save the changes to your `.bashrc` file.
   - Restart your terminal or run `source ~/.bashrc` to apply the changes to the environment variables.   
   - When installing SNOPT on Windows, your operating system may lack some "Dynamic Link Libraries" (DLLs) required by SNOPT. If this is the case, you can use the [`Dependencies`](https://lucasg.github.io/Dependencies/) tool to identify which DLLs are missing. To resolve this issue:

     1. Download the [`Dependencies`](https://lucasg.github.io/Dependencies/) tool.
     2. Run the tool and open the `snopt7.dll` library file.
     3. The tool will display a list of missing DLLs, if any.
     4. Download the missing DLLs from a reliable source.
     5. Place the downloaded DLLs in the same directory as the `snopt7.dll` library.


By following these steps, you can install additional solvers like IPOPT and SNOPT through the pygmo wrapper, expanding the range of optimization algorithms available for solving performance analysis and optimization problems in turboflow.


## Developer Installation Guide

This guide is intended for developers who wish to contribute to or modify the Turboflow source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

1. Clone the Turboflow remote repository:
   ```
   git clone https://github.com/turbo-sim/TurboFlow.git
   ```

2. Create a dedicated Conda virtual environment for Turboflow development:
   ```
   conda create --name turboflow_env python=3.11
   ```

3. Activate the newly created virtual environment:
   ```
   conda activate turboflow_env
   ```

4. Install Poetry to manage dependencies:
   ```
   conda install poetry
   ```
   Poetry is a powerful dependency manager that offers separation of user and developer dependencies, ensuring that only the necessary packages are installed based on the user's intent. Additionally, it simplifies the process of adding, updating, and removing dependencies, making it easier to maintain the project's requirements.

5. Use Poetry to install the required dependencies for Turboflow development:
   ```
   poetry install
   ```

6. Verify the installation by running the following command:
   ```
   python -c "import turboflow; turboflow.print_package_details()"
   ```

   If the installation was successful, you should see the Turboflow banner and package information displayed in the console output.




## CI/CD Pipeline

Turboflow uses GitHub Actions to automate its Continuous Integration and Continuous Deployment (CI/CD) processes.

### Automated Testing

The `ci.yml` action is triggered whenever a commit is pushed to the repository. This action runs the test suite on both Windows and Linux environments, ensuring the code's compatibility and correctness across different platforms.

### Package Publishing

Turboflow utilizes the `bumpversion` package to manage versioning and release control. To increment the version number, use the following command:

```bash
bumpversion patch  # or minor, major
```

After bumping the version, push the changes to the remote repository along with tags to signify the new version:

```bash
git push origin --tags
```

If the tests pass successfully, the package is automatically published to the Python Package Index (PyPI), making it readily available for users to install and use.

### Documentation Deployment

Turboflow automates the deployment of documentation using the `deploy_docs` action. This action builds the Sphinx documentation of the project and publishes the HTML files to GitHub Pages each time that a new commit is pushed to the remote repository. By automating this process, Turboflow ensures that the project's documentation remains up-to-date and easily accessible to users and contributors.









## To-do list
- [x] Verify torque and efficiency deviation match
- [x] Check if the correlation for incidence losses from Moustapha 1990 is better
- [x] Check if baseline kacker-okapuu + moustapha 1990 is better
- [x] Create clean dataset including torque/efficiency/flow angle for each pressure ration/speed
- [x] Extend validation Kofskey1972 to exit flow angle
- [x] Add x=y comparison plots for the validation
- [ ] Verify the displacement thickness value for kofskey1974
- [ ] Try to extract the shock loss at the inlet fromm the profile loss to have a more clear split of the losses
- [x] Add better smoother for Mach number constraint
- [x] Add generic smoothing functions for min/max/abs/piecewise
- [ ] Replace all non-differentiable functions of the code (specially in the loss models)
- [x] Improve initial guess computation for supersonic case (call the function `initialize()`)
- [ ] Validate model with experimental data
- [x] Add automatic testing (pytest) to ensure that new code pushes do not introduce breaking changes
- [x] Make function to plot performance maps more general
- [x] Add functionality to export/import all the parameters corresponding to a calculated operating point
- [x] Make the plotting of performance maps more general and add functionality to save figures
- [x] Add environment.yaml file to install all dependencies at once
- [x] Add Sphinx documentation structure

- [ ] Implement design optimization
  - [ ] Single-point
  - [ ] Multi-point
- [ ] Add CI/CD pipeline to run tests and update documentation on new pushes to main branch
- [ ] Think of a nice catchy name for the project
  - MeanFlow?
  - MeanStream?
  - MeanTurbo
  - MeanTurboPy
  - TurboCompuCore
  - TurboCore
  - Meanpy
  - Others?
  - TurboFlow?


## To-do list 10.11.2023
- [x] Residuals and independent variables should be dictionaries
- [x] Improve initial guess functionality (Lasse)
  - [x] Move initial guess generation into the CascadesNonlinearSystemProblem() class
  - [x] Move the solution scaling into the CascadesNonlinearSystemProblem() class [how to re-use scaling for optimization?]
  - [x] Generation within CascadesNonlinearSystemProblem() class
  - [x] Initial guess should be a dictionary with keys
  - [x] Initial guess specification in YAML file (this will be used for the first operation point)
  - [x] Improve initial guess calculation and extend to multistage
  - [ ] 1D correlation for Ma_crit as a function of the loss coefficient of the cascade
- [x] Geometry processing (Roberto)
  - [x] Clean up code
  - [x] Improve docstrings
  - [x] Discuss and improve the throat area calculation (Roberto/Lasse)
- [ ] Update PySolverView to solve root finding problems with the optimization solver.
- [x] Add computation time in solver report
- [ ] Improve robustness with random generation of initial guesses
  - [ ] Specify the number of initial guesses that will be tried, e.g., 50
  - [ ] Specify the ranges of variables to be sampled from (enthalpy frac distribution, efficiency ts and tt)
  - [ ] Use a sampling techniques like latin hypercube, montecarlo, or orthogonal sampling'
  - [ ] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
- [x] Add regression tests
- [ ] 

## TODO 17.11.2023
- [x] Add validation of configuration file
- [x] Add configuration file options to documentation (automated process)
- [ ] Add logger object to the performance analysis and optimization functions
- [ ] Log errors and warnings during model evaluation (print after convergence history?)
- [ ] Improve geometry generation functionality to create full geometry from design optimization variables
- [ ] Improve printing/logging with imporve control over verbosity
- [ ] Solver improvements
  - [ ] Unify optimization problem so that it can be solver with root finding and optimization methods
  - [ ] Add pygmo solvers to the pysolverview interface (IPOPT/SNOPT)
- [x] Improve deviation modeling to cover cases when:
  - [x] Area changes from throat to exit
  - [x] Radius changes from throat to exit
  - [x] There is an additional blockage factor at the throat
- [ ] Add plotting functionality
  - [ ] Plot velocity triangles
  - [ ] Plot blade to plade plane
  - [ ] Plot meridional view
  - [ ] Plot h-s or T-s diagrams of the expansion
  - [ ] Nice validation plotting against experimental data
  - [ ] 



## TODO 06.12.2023
- [x] Use gauge and and not opening-to-pitch in deviation model functions
- [x] Make sure opening is only used for the critical mass flow rate calculation and blockage factor
- [x] Do not use ordinary compute_cascade_exit function to determine the flow conditions at the throat
- [ ] Have the option to calculate the state at the throat for the actual condition? Probably not
- [x] Remove option of choking condition
- [ ] Deviation model new structure:
 - [ ] low_speed_limit: aungier | ainley-methieson | zero-deviation
 - [ ] interpolation: aungier | ainley-methieson | borg
 - [ ] blending_parmaters:
	n: 
	m: 
	slope:
- [x] We improve slope so that it works when A_th=A_exit and in other cases (slope zero or one)
- [ ] Improve initialization / initial guess generation
- [x] Write down notes explaining the behavior of the deviation model when the area is the same and when it changes
- [ ] Write down notes explaining the equations for the blending
- [x] check the throat area calculation i nthe geometyr.py script and clean it
- [ ] add option to calculate throat loasses without trailing edge losses 

## TODO 15.12.2023
- [ ] Change loader and dumper to recognize numpy (custom tags)
- [x] Use pedantic for validation instead of custom function (JSON schema)
- [ ] Add logger with errors, info and warnings
- [ ] FASTApi - open API (swagger)


## TODO 16.01.2024
- [ ] Improve solver to not converge for low residual, and converge for satisfactory residuals
- [ ] Implement the blender such that it is only used for zero deviation
- [ ] Improve initial guess function


Guidelines/instructions for the deviation model

Subsonic deviation according to ainley-mathieson or aungier, or zero-deviation
critical mach number according to out calculation
critical flow angle equal to metal_angle_out

interpolation can be:
- ainley-methieson (linear)
- aungier (5th order polynomial for delta)
- third order polynomial with controller end-point slope (parameter to have slope 1 or slope 0)

slope should be zero for cases when Mach_crit_throat = Mach_crit_exit
slope should be close to one for cases when Mach_crit_throat > Mach_crit_exit

sigmoid blending between subsonic interpolated function and the supersonic solution (in the subsonic regime)
we have to expend the supersonic branch into the subsonic to calculate the flow angle to ensure that we have a smooth flow angle transition

