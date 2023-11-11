# Changelog

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