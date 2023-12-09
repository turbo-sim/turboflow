# Work Progress

## Completed Tasks

### Modeling
- Main translation from the MATLAB code to Python.
- Implemented functions and classes to evaluate/optimize recuperated Brayton cycle.
- Implemented polytropic process for expansion compression and isentropic calculations for faster computation speed.
- Discretization and temperature difference calculation for heat exchangers.
- Reformulated the problem using components instead of states. This makes the code more flexible and intuitive. There is no need to keep track of many state points numbered from 1 to a big number.
- Implemented lazy initialization to avoid recomputing the saturation line.
- Generalized the vapor quality calculation for values outside of the two-phase region.
- Added calculation of superheating and subcooling that works inside and outside the two-phase region

### Optimization
- Formulated the objective function and constraints automatically via configuration.
- Introduced very flexible constraint definition by accessing variables in nested dictionaries.
- Added functionality to ensured the objective function are numeric and scalar.
- Added functionality to ensured the objective function are numeric.
- Incorporated constraint and objective normalization options.
- Improved the callback function to save convergence history.

### File IO
- Encapsulated all problem parameters in a YAML configuration file.
- Devised a method for normalizing constraints according to the configuration file.
- Implemented data export to an organized folder with unique datetime identifiers.
- Added functionality to export the configuration file.
- Added export functionality for cycle performance as a pandas dataframe.
- Developed a modest expression rendering engine using the `$` character in the YAML file.
  - Directly set bounds for the heat sink and heat source from fixed parameters.
  - Enhanced pressure/enthalpy bounds definition based on key states.


### Plotting
- Added functionality to plot the cycle at each optimization iteration (callback function).
- Implemented interactive plotting function that reads from the configuration file. This facilitates the generation of an initial guess interactively.
- Enabled code to continue upon closing the interactive plot or pressing enter. The function runs a subprocess in the background to ensure that the code recognizes the user typing enter.
- Enhanced phase diagram plotting and updating, including quality grid and response to property pair changes. Added visibility control for phase diagram items via the YAML file. The phase diagram settings are global for all the thermodynamic diagrams. This means that if the option to plot the saturation line is activated then it will be displayed in all the diagrams specified in the configuration file.
- Implemented default plotting options when specifics are not provided.
- Added functionality to create an animation of the optimization process. Exporting the optimization progress of a video does not work when the image has a very low aspect ratio (i.e., when it is too wide). I experienced that the export works when the figure has one row with two subplots, but it fails when it has one row with 3 subplots.
- Integrated a Q-T pinch point diagram that works for any cycle.


expression engine for constraint and obj funct


### Work in Progress
- Creating a generic cycle optimization problem class, as opposed to a sophisticated Brayton cycle problem class?
- Providing the configuration to the solver class? Should all the process from initialization to optimization be handled by a single class? It could be convenient because then all the configuration dictionary is passed as a single argument.
- Developing methods for the solver class, including plot interactive, start optimization, and export configuration.
- Further generalizing the vapor quality calculation for supercritical qualities.


# To-Do Checklist
- [ ] Check for any discrepancies with the MATLAB code.
- [ ] Implement exergy closure calculations.
- [ ] Improve the generalization of vapor quality calculations for supercritical conditions.
- [ ] Create a cycle optimization problem class.
- [ ] Develop `plot interactive` and `start optimization` methods for the solver class.
- [ ] Add `export configuration` functionality.
- [ ] Implement a method to create GIFs from optimization runs.
- [ ] Call the class in a loop to plot the cycle for different inlet temperatures or other parameters.
- [ ] Add a directory with example optimization cases.
  - [ ] Rankine cycle
    - [ ] Superheated cycle
    - [ ] Saturated cycle
    - [ ] Partial evaporation cycle
    - [ ] Trilateral cycle
  - [ ] Brayton cycle
    - [ ] Recuperated
    - [ ] Split compression
- [ ] Add configuration validation? All design variables and fixed parameters defined?
- [ ] Develop a strategy to restart from previously converged solutions.
- [ ] Conduct sCO2 studies using the code:
  - [ ] Influence of inlet temperature and pressure on efficiency.
  - [ ] Influence of turbomachinery modeling
    - [ ] Isentropic machines
    - [ ] Isentropic efficiency
    - [ ] Map correlation
    - [ ] Meanline model
  - [ ] Study different shaft configurations
    - [ ] Recuperated cycle with 1 or 2 shafts
    - [ ] Recompression cycle with 1, 2, or 3 shafts
