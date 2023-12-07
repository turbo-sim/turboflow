
# Meanline Axial: Axial Turbine Mean-line Modelling

**Meanline Axial** is a tool for mean-line modelling of axial turbines. It provides a systematic approach to analyze and optimize axial turbines based on specified requirements.

## Core Features
- **Performance Analysis Mode**: 
  - Evaluate single operating points
  - Produce performance maps
- **Design Optimization Mode**: 
  - Single point optimization
  - Multi-point optimization
- **Problem formulation and solution**
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


## Installation Instructions

1. **Install Conda**:

   Before proceeding, ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) installed on your system.

2. **Create a virtual environment with dependencies**:

   Run the following command in [Bash](https://gitforwindows.org/) to set up a new virtual environment with all the required dependencies:

   ```bash
   conda env create --file environment.yaml
   ```

   This command will create a virtual environment named `meanline_env` and will install all the packages specified in the `environment.yaml` file.

3. **Activate the virtual environment**:

   ```bash
   conda activate meanline_env
   ```

4. **Installing additional packages (optional)**:

   If you need any additional packages, they can be installed using:

   ```bash
   conda install <name of the package>
   ```

   Alternatively, you can add package names directly to the `environment.yaml` file and then update the environment:

   ```bash
   conda env update --file environment.yaml --prune
   ```

5. **Setting up for local development**

   To ensure that you can import the package for local development, you need to add the package directory to the `PYTHONPATH` variable. We have provided a convenient script named `install_local.py` to do this for you.

   Just run:

   ```bash
   python install_local.py
   ```

   This will append the current working directory (which should be the root of this repository) to your `PYTHONPATH` by adding a line to your `~/.bashrc` file.

   **Note:** This is a temporary development solution. In the future the package will be installed via pip/conda.


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
  - [ ] Discuss and improve the throat area calculation (Roberto/Lasse)
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
- [ ] Remove option of choking condition
- [ ] Deviation model new structure:
 - [ ] low_speed_limit: aungier | ainley-methieson | zero-deviation
 - [ ] interpolation: aungier | ainley-methieson | borg
 - [ ] blending_parmaters:
	n: 
	m: 
	slope:
- [ ] We improve slope so that it works when A_th=A_exit and in other cases (slope zero or one)
- [ ] Improve initialization / initial guess generation
- [x] Write down notes explaining the behavior of the deviation model when the area is the same and when it changes
- [ ] Write down notes explaining the equations for the blending
- [ ] check the throat area calculation i nthe geometyr.py script and clean it
- [ ] add option to calculate throat loasses without trailing edge losses 


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



## CI/CD

[Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)

