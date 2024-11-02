

## Installation Instructions

1. **Install Conda**:

   Before proceeding, ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) installed on your system.

2. **Create a virtual environment with dependencies**:

   Run the following command in [Bash](https://gitforwindows.org/) to set up a new virtual environment with all the required dependencies:

   ```bash
   conda env create --file requirements.yaml
   ```

   This command will create a virtual environment named `hydrogen_env` and will install all the packages specified in the `requirements.yaml` file.

3. **Activate the virtual environment**:

   ```bash
   conda activate hydrogen_env
   ```

4. **Run the scripts**
    Execute the python files in the terminal or in your IDE (e.g., Pycharm or VScode):

   ```bash
   python demo_hydrogen_pipeline.py
   ```

5. **Installing additional packages (optional)**:

   If you need any additional packages, they can be installed using:

   ```bash
   conda install <name of the package>
   ```

   Alternatively, you can add package names directly to the `requirements.yaml` file and then update the environment:

   ```bash
   conda env update --file requirements.yaml --prune
   ```

## Modeling
This model computes the steady-state in a one-dimensional duct by integrating mass, momentum, and energy equations along the pipe's length. 

- Wall friction is modeled using the Darcy-Weisbach equation with the Haaland correlation to determine the friction factor.
- Wall heat transfer is modeled through an overall heat transfer coefficient estimated with the Reynolds analogy between momentum and heat transfer
- Fluid properties are calculated with the CoolProp fluid library. The fluid does not behave as an ideal gas (Z>1) because the pressure level is significantly higher than the critical pressure. 

The equations are integrated numerically from the inlet to the outlet of the duct using the Explicit Runge-Kutta method of order 5(4) available in the scipy.


## Validation
Validation of the computational model was performed using analytical solutions applicable to perfect gas scenarios:

- Isentropic flow within ducts of varying cross-sectional area.
- Fanno flow in ducts with constant cross-sectional area.
  
The results showed good agreement with the analytical formulas for both the critical area ratio and critical pipe length in the scenarios mentioned above.

In addition, it was verified that the mass, total enthalpy and entropy as conserved (up the numerical integration tolerance) when friction and heat transfer are zero.