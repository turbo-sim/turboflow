





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
  - [x] Single-point
  - [ ] Multi-point
- [x] Add CI/CD pipeline to run tests and update documentation on new pushes to main branch


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
- [ ] Increase robustness allowing CoolProp failures during convergence? but not in the final iteration
- [x] Add regression tests

## TODO 17.11.2023
- [x] Add validation of configuration file
- [x] Add configuration file options to documentation (automated process)
- [ ] Add logger object to the performance analysis and optimization functions
- [ ] Log errors and warnings during model evaluation (print after convergence history?)
- [ ] Improve geometry generation functionality to create full geometry from design optimization variables
- [ ] Improve printing/logging with imporve control over verbosity
- [ ] Solver improvements
  - [ ] Unify optimization problem so that it can be solver with root finding and optimization methods
  - [x] Add pygmo solvers to the pysolverview interface (IPOPT/SNOPT)
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




Add idal gas calculations for fast evaluations
Add automatic differentiation and jit for reliable convergence and speed
Add rho-T thermodynamic function calls for increased robustness and speed (true equation oriented)