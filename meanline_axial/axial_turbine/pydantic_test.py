
from pydantic import BaseModel, conlist, confloat, conint, constr, ValidationError, Field
from typing import Optional, List, Union, Literal
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
import numpy as np
import numbers
import yaml

LOSS_MODELS = ["kacker_okapuu", "moustapha", "benner", "isentropic","custom"]
LOSS_COEFFICIENTS = ["stagnation_pressure"]
DEVIATION_MODELS = ["aungier", "ainley_mathieson", "zero_deviation"]
CHOKING_MODELS = ["evaluate_cascade_critical",
                  "evaluate_cascade_throat",
                  "evaluate_cascade_isentropic_throat"]
ROOT_SOLVERS = ["hybr", "lm"]
DERIVATIVE_METHODS = ["2-point", "3-point"]
OBJECTIVE_FUNCTIONS = ["none", "efficiency_ts"]
DESIGN_VARIABLES = ["specific_speed", 
                    "blade_jet_ratio",
                    "hub_tip_ratio_in",
                    "hub_tip_ratio_out",
                    "aspect_ratio",
                    "pitch_chord_ratio",
                    "trailing_edge_thickness_opening_ratio",
                    "leading_edge_angle",
                    "gauging_angle"
                    ]

CONSTRAINTS = ["mass_flow_rate", "interstage_flaring"]
LIBRARIES = ["scipy", "pygmo", "pygmo_nlopt"]
RADIUS_TYPE = ["constant_mean",
               "constant_hub",
               "constant_tip"]

def check_string(input : str, input_type, options) -> str:
    assert input in options, f"Invalid {input_type}: '{input}'. Available options: {options}"
    return input

class OperationPoints(BaseModel):
    fluid_name: str
    T0_in: Annotated[float, Field(gt=0)]
    p0_in: Annotated[float, Field(gt=0)]
    p_out: Annotated[float, Field(gt=0)]
    omega: Annotated[float, Field(gt=0)]
    alpha_in: float

class PerformanceMap(BaseModel):
    fluid_name: Union[str, List[str]]
    T0_in: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    p0_in: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    p_out: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    omega: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    alpha_in: Union[float, List[float]]

class TuningFactors(BaseModel):
    profile: float = 1.00
    incidence: float = 1.00
    secondary: float = 1.00
    trailing: float = 1.00
    clearance: float = 1.00

class LossModel(BaseModel):
    model : Annotated[str, AfterValidator(lambda input: check_string(input, "loss model", LOSS_MODELS))]
    loss_coefficient: Annotated[str, AfterValidator(lambda input: check_string(input, "loss coefficient",LOSS_COEFFICIENTS))]
    inlet_displacement_thickness_height_ratio: float = 0.011
    tuning_factors: TuningFactors = TuningFactors()

class SimulationOptions(BaseModel):
    deviation_model: Annotated[str, AfterValidator(lambda input: check_string(input, "deviation model", DEVIATION_MODELS))]
    blockage_model: Union[str, float] = 0.00
    choking_model: Annotated[str, AfterValidator(lambda input: check_string(input, "choking model", CHOKING_MODELS))]
    rel_step_fd: float = 1e-4
    loss_model: LossModel

class SolverOptions(BaseModel):
    method: Annotated[str, AfterValidator(lambda input: check_string(input, "root solver",ROOT_SOLVERS)), Field(default = 'hybr')]
    tolerance: float = 1e-8
    max_iterations: int = 100
    derivative_method: Annotated[str, AfterValidator(lambda input: check_string(input, "derivative method", DERIVATIVE_METHODS)), Field(default = '2-point')]
    derivative_abs_step: float = 1e-6
    print_convergence: bool = True
    plot_convergence: bool = False

class GeometryPerformanceAnalysis(BaseModel):
    cascade_type: List[Annotated[str, AfterValidator(lambda input: check_string(input, "cascade type", ["stator", "rotor"]))]]
    radius_hub_in: List[Annotated[float, Field(gt=0)]]
    radius_hub_out: List[Annotated[float, Field(gt=0)]]
    radius_tip_in: List[Annotated[float, Field(gt=0)]]
    radius_tip_out: List[Annotated[float, Field(gt=0)]]
    pitch: List[Annotated[float, Field(gt=0)]]
    chord: List[Annotated[float, Field(gt=0)]]
    stagger_angle: List[Annotated[float, Field(le=90), Field(gt=-90)]]
    opening: List[Annotated[float, Field(gt=0)]]
    leading_edge_angle: List[Annotated[float, Field(le=90), Field(gt=-90)]]
    leading_edge_wedge_angle: List[Annotated[float, Field(le=90), Field(gt=0)]]
    leading_edge_diameter: List[Annotated[float, Field(gt=0)]]
    trailing_edge_thickness: List[Annotated[float, Field(gt=0)]]
    maximum_thickness: List[Annotated[float, Field(gt=0)]]
    tip_clearance: List[Annotated[float, Field(ge=0)]]
    throat_location_fraction: List[Annotated[float, Field(le=1), Field(gt=0)]]

class Constraint(BaseModel):
    variable : Annotated[str, AfterValidator(lambda input: check_string(input, "constraint", CONSTRAINTS))]
    type : Literal["<", "=", ">"]
    value : float
    normalize : bool

class SolverOptionsOptimization(BaseModel):
    library : Annotated[str, AfterValidator(lambda input: check_string(input, "optimization library",LIBRARIES)), Field(default = 'scipy')]
    method : str = "slsqp" # TODO: Update when integrated into TurboFlow
    tolerance : float = 1e-5
    max_iterations : int = 100
    derivative_method :  Annotated[str, AfterValidator(lambda input: check_string(input, "derivative method",DERIVATIVE_METHODS)), Field(default = '2-point')]
    derivative_abs_step : float = None
    print_convergence : bool = True
    plot_convergence : bool = False
    update_on : Annotated[str, AfterValidator(lambda input: check_string(input, "update_on input",["gradient", "function"])), Field(default = 'gradient')]

class Optimization(BaseModel):
    objective_function : Annotated[str, AfterValidator(lambda input: check_string(input, "objective function",OBJECTIVE_FUNCTIONS)), Field(default = 'efficiency_ts')]
    design_variables : List[Annotated[str, AfterValidator(lambda input: check_string(input, "design variable", DESIGN_VARIABLES))]]
    constraints : List[Constraint] = None
    solver_options : SolverOptionsOptimization = SolverOptionsOptimization()
    radius_type : Annotated[str, AfterValidator(lambda input: check_string(input, "radius type", RADIUS_TYPE)), Field(default = 'constant_mean')]
    bounds : List[tuple] = None

class GeometryOptimization(BaseModel):

    hub_tip_ratio_in : List[Annotated[float, Field(le=1), Field(gt=0)]]
    hub_tip_ratio_out : List[Annotated[float, Field(le=1), Field(gt=0)]]
    aspect_ratio : List[Annotated[float, Field(gt=0)]]
    pitch_chord_ratio : List[Annotated[float, Field(gt=0)]]
    gauging_angle : List[Annotated[float, Field(le=90), Field(gt=-90)]]
    leading_edge_angle : List[Annotated[float, Field(le=90), Field(gt=-90)]]
    trailing_edge_thickness_opening_ratio : List[Annotated[float, Field(le=1), Field(gt=0)]]
    cascade_type : List[Annotated[str, AfterValidator(lambda input: check_string(input, "cascade type", ["stator", "rotor"]))]]
    leading_edge_diameter : List[Annotated[float, Field(gt=0)]]
    leading_edge_wedge_angle : List[Annotated[float, Field(le=90), Field(gt=0)]]
    tip_clearance : List[Annotated[float, Field(ge=0)]]
    throat_location_fraction : List[Annotated[float, Field(le=1), Field(gt=0)]]

class PerformanceAnalysisConfig(BaseModel):
    operation_points: OperationPoints
    performance_map: PerformanceMap = OperationPoints
    simulation_options: SimulationOptions 
    solver_options: SolverOptions = SolverOptions()
    geometry: GeometryPerformanceAnalysis

class DesignOptimizationConfig(BaseModel):
    operation_points: OperationPoints
    simulation_options: SimulationOptions
    optimization: Optimization
    geometry: Union[GeometryPerformanceAnalysis, GeometryOptimization] = Field(union_mode='left_to_right')

def convert_configuration_options(config):
    """
    Processes configuration data by evaluating string expressions as numerical values and converting lists to numpy arrays.

    This function iteratively goes through the configuration dictionary and converts string representations of numbers
    (e.g., "1+2", "2*np.pi") into actual numerical values using Python's `eval` function. It also ensures that all numerical
    values are represented as Numpy types for consistency across the application.

    Parameters
    ----------
    config : dict
        The configuration data loaded from a YAML file, typically containing a mix of strings, numbers, and lists.

    Returns
    -------
    dict
        The postprocessed configuration data where string expressions are evaluated as numbers, and all numerical values
        are cast to corresponding NumPy types.

    Raises
    ------
    ConfigurationError
        If a list contains elements of different types after conversion, indicating an inconsistency in the expected data types.
    """

    def convert_strings_to_numbers(data):
        """
        Recursively converts string expressions within the configuration data to numerical values.

        This function handles each element of the configuration: dictionaries are traversed recursively, lists are processed
        element-wise, and strings are evaluated as numerical expressions. Non-string and valid numerical expressions are
        returned as is. The conversion supports basic mathematical operations and is capable of recognizing Numpy functions
        and constants when used in the strings.

        Parameters
        ----------
        data : dict, list, str, or number
            A piece of the configuration data that may contain strings representing numerical expressions.

        Returns
        -------
        The converted data, with all string expressions evaluated as numbers.
        """
        if isinstance(data, dict):
            return {
                key: convert_strings_to_numbers(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [convert_strings_to_numbers(item) for item in data]
        elif isinstance(data, bool):
            return data
        elif isinstance(data, str):
            # Evaluate strings as numbers if possible
            try:
                data = eval(data)
                return convert_numbers_to_numpy(data)
            except (NameError, SyntaxError, TypeError):
                return data
        elif isinstance(data, numbers.Number):
            # Convert Python number types to corresponding NumPy number types
            return convert_numbers_to_numpy(data)
        else:
            return data

    def convert_numbers_to_numpy(data):
        """
        Casts Python native number types (int, float) to corresponding Numpy number types.

        This function ensures that all numeric values in the configuration are represented using Numpy types.
        It converts integers to `np.int64` and floats to `np.float64`.

        Parameters
        ----------
        data : int, float
            A numerical value that needs to be cast to a Numpy number type.

        Returns
        -------
        The same numerical value cast to the corresponding Numpy number type.

        """
        if isinstance(data, int):
            return np.int64(data)
        elif isinstance(data, float):
            return np.float64(data)
        else:
            return data

    def convert_to_arrays(data, parent_key=""):
        """
        Convert lists within the input data to Numpy arrays.

        Iterates through the input data recursively. If a list is encountered, the function checks if all elements are of the same type.
        If they are, the list is converted to a Numpy array. If the elements are of different types, a :obj:`ConfigurationError` is raised.

        Parameters
        ----------
        data : dict or list or any
            The input data which may contain lists that need to be converted to NumPy arrays. The data can be a dictionary (with recursive processing for each value), a list, or any other data type.
        parent_key : str, optional
            The key in the parent dictionary corresponding to `data`, used for error messaging. The default is an empty string.

        Returns
        -------
        dict or list or any
            The input data with lists converted to NumPy arrays. The type of return matches the type of `data`. Dictionaries and other types are returned unmodified.

        Raises
        ------
        ValueError
            If a list within `data` contains elements of different types. The error message includes the problematic list and the types of its elements.

        """
        if isinstance(data, dict):
            return {k: convert_to_arrays(v, parent_key=k) for k, v in data.items()}
        elif isinstance(data, list):
            if not data:  # Empty list
                return data
            first_type = type(data[0])
            if not all(isinstance(item, first_type) for item in data):
                element_types = [type(item) for item in data]
                raise ValueError(
                    f"Option '{parent_key}' contains elements of different types: {data}, "
                    f"types: {element_types}"
                )
            return np.array(data)
        else:
            return data

    # Convert the configuration options to Numpy arrays when relevant
    config = convert_strings_to_numbers(config)
    config = convert_to_arrays(config)

    return config

def is_convertible_to_dict(obj):
    """Check if an object can be converted to a dictionary by looking for the __dict__ attribute."""
    return hasattr(obj, '__dict__')

def object_to_dict(obj):
    """Recursively convert an object's attributes to a dictionary."""
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return np.array([object_to_dict(i) for i in obj])
    elif is_convertible_to_dict(obj):
        return {k: object_to_dict(v) for k, v in vars(obj).items()}
    else:
        return obj

# Function to load and validate the configuration file
def load_config(config_file_path: str, mode = "performance_analysis"):
    try:
        with open(config_file_path, 'r') as config_file:
            # Load config file
            config_data = yaml.safe_load(config_file)

            # Convert string representation of numbers to number and lists to arrays
            config_data = convert_configuration_options(config_data)

            # Perform confuguration check according to mode
            if mode == "performance_analysis":
                config = PerformanceAnalysisConfig(**config_data)
            elif mode == "optimization":
                config = DesignOptimizationConfig(**config_data)
            
            if config:
                # Check which set of geometry the input corresponds with
                if type(config.geometry) == GeometryOptimization: 
                    geometry_type = "optimization" 
                elif type(config.geometry) == GeometryPerformanceAnalysis: 
                    geometry_type = "performance_analysis"
                print("hei")
                # Convert configuration object to a nested dictionary
                config = object_to_dict(config)

                # Mark the geometry with wich set it corresponds to 
                config["geometry"]["geometry_type"] = geometry_type

                # Print configuration summary
                print("Configuration summary: \n")
            return config
        
    except FileNotFoundError:
        print("Configuration file not found.")
    except ValidationError as e:
        print("Validation error in configuration file:")
        print(e)
    except yaml.YAMLError as e:
        print("Error parsing YAML file:")
        print(e)

# Example usage
if __name__ == "__main__":
    config_file_path = "config.yaml"  # Path to your YAML configuration file
    config = load_config(config_file_path, mode = "optimization")
    if config:
        print("Configuration loaded successfully")

