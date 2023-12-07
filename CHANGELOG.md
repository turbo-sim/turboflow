# Changelog

> WARNING! There are convergence problems for supersonic cases when using the `mach_critical` and `mach_unity` options for the choking condition (Kofskey case). I am not sure since when do we have these problems because I had not used these options before. We have to see if we broke something when refactoring the code and make sure that the `mach_critical` and `mach_unity` options work again.\

## 07.12.2023
- Removed throat calculation for actual cascade
- Created a evalute_throat_plane function
    - Different way to calculate mass flow rate
    - Can you actual throat area and not annulus area
- Correct deviation models to not use the cosine rule
- Enable flexibility to have different A_throat and A_exit
    - Can specify throat location
    - Can specify blockage factor
    - Can specify opening 
- Implement blending function between subsonic and supersonic flow angle for a smooth transition
- Developed new deviation model that should approximate the slope of the supersonic flow angle function at critical point, and interpolated between low speed flow angle, and supersonic flow angle more or less smoothly


## 15.11.2023
- Added configuration file validation functionality
- Added validation schema to documentation (updates automatically)
- Updated documentation for configuration validation
- Updated the regression tests YAML file with the new configuration options

## 14.11.2023
- Refractor the "utilities" import in `cascade_series.py` and `performance_analsys.py`
- Add documentation and validation for the keys expected in the results dictionary
- Moved `compute_overall_performance()` calculations to a dedicated function
- Refractored and renamed `compute_stage_parameters()` to  `compute_stage_performance()`.
- Added a new generic function: `validate_keys()` that should be used for all validations of dictionary keys.
- Added the new `validate_keys()` generic function to `geometry.py`
- Simplified the argument list of `compute_efficiency_breakdown()`
- Renamed `cascade_series.py` to `flow_model.py` and replace the abbreviation "cs" by "flow"
## 13.11.2023
- Add additional problem evaluation after root finding completion (this ensures the solver.problem object has the converged solution)
- Add elapsed time to root finding solution report
- Add function `print_simulation_summary()` to print report after running performance analysis.

## 12.11.2023
- Regression testing
    - Added script to generate regression datasets from several configuration files.
    - Added test functions to verify that the code gives the same outcome as the regression files.
    - Added one regression case for each loss model.
- Add functionality to save geometry to 'results' dictionary.
- Added functionality to export geometry to Excel file.
- Converted the results["overall"] from dictionary into pandas dataframe.
- Added more critical variables to the "cascade" dataframe (d_crit, w_crit, p_crit, beta_crit)
- Added 'area' and 'radius' arguments to the `evaluate_exit()` function. The area and radius can be different for the exit and for the throat sections if the annulus shape is not constant. The position of the throat (and therefore its area and radius) are specified by the geometric parameter 'throat_location_fraction'.
- WARNING! The option above is giving convergence problems when 'throat_location_fraction'=/=1 (that is, when the throat geometry is not exactly equal to the exit geometry). We have to investigate and solve this problem.
- Fix error in `geometry.calculate_throat_radius()` function.
- Add option to not export results to Excel on `performance_analysis()` function
- Improve fnctionality in `plot_functions.py`
  - Clean up code
  - Add function to save in multiple formats
- Add `utilities.extract_timestamp(filename)` function to extract the time stamp from exported files
- Improve `plot_kofskey1972_1stage.py`
- Fix spelling mistake in `run_tests.py`

## 11.11.2023

### Performance Analysis
- Removed options to generate new initial guesses (functionality should be improved before implementation).

### Utilities
- Added `all_numeric()` and `all_non_negative()` functions to `math.py`.
- Added functionality to validate configuration files, including checks for mandatory and unexpected fields.
- Extended `postprocess_config()` function to automatically convert lists to numpy arrays.
- Enhanced `convert_numpy_to_python()` function to handle `None` values for YAML export.
- Moved `add_string_to_keys()` function from `cascade_series.py` into `utilities.py`.

### Geometry
- Added 'throat_location_fraction' variable for control over throat area's radius variation.
- Cleaned `check_axial_turbine_geometry()` function.

### Deviation Model
- Refactored `deviation_model.py` module to use function mapping for option selection.
- Renamed option "metal_angle" to "zero_deviation".

### Loss Models
- Refactored `loss_model.py` module to use function mapping for option selection.
- Removed `evaluate_loss_model()` function from `cascade_series.py`; functionality moved to `loss_model.py`.
- Removed reference endwall displacement thickness from reference values dict.
- Removed endwall displacement thickness calculations from cascade series.
- Added endwall displacement thickness calculations to Benner loss model.
- Added option to specify displacement thickness for Benner model in configuration file.
- Updated variable names to new geometry definition in all loss models.
- Added option to specify tuning factors for the loss model.
- Added function to validate loss dictionary returned from loss models.
- Added option to specify loss coefficient definition (currently only 'stagnation_pressure' implemented).

### Blockage
- Improved option for adding boundary layer blockage at the throat.
- Setting 'throat_blockage' to a number in the configuration file now specifies fraction blockage.
- Setting 'throat_blockage' to 'flat_plate_turbulent' in the configuration file applies blockage according to correlation.

### Cascade Series
- Refactored handling of choking equation in `cascade_series.py` to use `CHOKING_OPTIONS` literal and function mapping for option selection.
- Split plane keys in `cascade_series.py` into several categories for better reusability.
- Extended `calculate_efficiency_drop_fractions` to include all loss components.
- Ensured that the sum of the loss components + kinetic energy loss + `efficiency_ts` equals 1.00.
- Renamed `evaluate_lagrange_gradient()` to `get_critical_residuals()`.
- Extended overall calculations to include exit velocity and flow angle, among others.
- Added blade velocity to the 'plane' dataframes.