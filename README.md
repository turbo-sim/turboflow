
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
- [ ] Verify torque and efficiency deviation match
- [ ] Check if the correlation for incidence losses from Moustapha 1990 is better
- [ ] Check if baseline kacker-okapuu + moustapha 1990 is better
- [ ] Create clean dataset including torque/efficiency/flow angle for each pressure ration/speed
- [ ] Extend validation Kofskey1972 to exit flow angle
- [ ] Add x=y comparison plots for the validation
- [ ] Verify the displacement thickness value for kofskey1974
- [ ] Try to extract the shock loss at the inlet form the profile loss to have a more clear split of the losses
- [x] Add better smoother for Mach number constraint
- [ ] Add generic smoothing functions for min/max/abs/piecewise
- [ ] Replace all non-differentiable functions of the code (specially in the loss models)
- [ ] Improve initial guess computation for supersonic case (call the function `initialize()`)
- [ ] Validate model with experimental data
- [ ] Add automatic testing (pytest) to ensure that new code pushes do not introduce breaking changes
- [x] Make function to plot performance maps more general
- [ ] Add functionality to export/import all the parameters corresponding to a calculated operating point
- [x] Make the plotting of performance maps more general and add functionality to save figures
- [x] Add environment.yaml file to install all dependencies at once
- [x] Add Sphinx documentation structure
- [ ] Add pygmo solvers to the pysolverview interface
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



## CI/CD

[Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)

