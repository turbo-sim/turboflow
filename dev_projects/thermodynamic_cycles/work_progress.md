# Work Progress


## Todo
- Clean the script to plot the Ts diagram with the Z-factor contours
- Clean the cycle optimization functionality
  - Move plotting outside of main class
  - Wrapper should just redefine some methods of the object? or create a new object with just the methods for optimization to prevent deepcoping problems?
  - Clean script about IPOPT superiority with Hessian
  - Make folder for the comparison of solvers for cycle optimization
  - Make some notes describing each case of cycle optimization
  - Merging of cycle optimization functionality with turbine optimization
  - Merging of fluid properties from barotropic model and meanline axial
  - Merging of barotropy with the meanline code?
  - clean pysolverview  demo scripts and functinality. make sure it is consistent with the meanline code? combine int he same package to share utilities?
  - Improve fluid property calculations:
    - Pseudocritical Line
    - Spinodal line
    - Superheating
    - Subcooling
    - Generalized quality


The cycle problem class only takes information about the problem formulation, not the overall config file
specifying also the settings for the optimization problem
As a result, the output YAML is not as meaninful as it could be

Should I have the plotting within the CycleProblem class? or move it somewhere else
Keep in mind that I wanbt to update plot iteratively when calling function plot_cycle_realtime,
but also update the plots with the latest data during optimization when a callback function is called

Now I export properly the YAML file. Check the naming and the folder structure to comly with the exporting of figures
Export initial configuration and final configuration separately

I have to create a wraper class for doing the optimization?

Would it be better to have a class for optimization where I also handle the solution of the optimization problem
In this class I could have the extra functionaliyt about plotting and exporting yaml file
Would this apporach be more suited to save confguration file at start and ensd of the optimiazation
Would this approach help me make parameter sweeps more, easily, or it does not really matter?

**TODO options to add as configuration**
Use plot callback or not, if yes, then make sure a function call to the plot_cycle function is done
Move the configuration function handling to the configuration management class
Move the plotting outside of the cycleProblem class
Update code to handle parameter sweeps gracefully (create ranges of variabtion by permutations of the parameters to be checked, and pass a single list of dctionaries with the updated variable names. Update the keys of the configu dictionary with the keys provided)

optimizer.problem.plot_cycle()  # TODO make sure that the optimization can run even if plot not initialized


## Done
- Ts diagram of the split compression cycle
- Do inkscape illustration of the split compression cycle
- Draft description of the split compression cycle (missing main modelling equations)
- 


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
- Cleaned up the recompression and recuperated cycle functions and moved them to their own files.


### Optimization
- Formulated the objective function and constraints automatically via configuration.
- Introduced very flexible constraint definition by accessing variables in nested dictionaries.
- Added functionality to ensured the objective function are numeric and scalar.
- Added functionality to ensured the objective function are numeric.
- Incorporated constraint and objective normalization options.
- Improved the callback function to save convergence history.
- Added option to specify the cycle topology in the configuration file and error message listing available options in case the cycle topology is invalid.
- Modified objective/constraint parsing. Now all variables being parsed during configuration reading have to be preceded by `$` character
- The rendering function was extended to handle variables that are Numpy arrays (e.g., the vector of temperature differences across a heat exchanger).
 
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
- The create/update plots functionality is handled with a class dictionary that keeps track of all the graphical objects and the axes to which they belong. This dictionary is re-initialized in case the figure window is closed and the user tries to plot the cycle again to have a smooth user experience.


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
- [ ] Add `export configuration` functionality. Very important, now I am only exporting excel data of the optimal solution
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
