
from pydantic import BaseModel, ValidationError, Field, model_validator, field_validator
from typing import Optional, List, Union, Literal, Dict
from typing_extensions import Self
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator, BeforeValidator
from ..axial_turbine import VARIABLES, LOSS_MODELS, LOSS_COEFFICIENTS, CHOKING_MODELS, DEVIATION_MODELS, OBJECTIVE_FUNCTIONS, CONSTRAINTS, RADIUS_TYPE
from ..pysolver_view import SOLVER_OPTIONS, VALID_LIBRARIES_AND_METHODS, DERIVATIVE_METHODS

DEFAULT_VARIABLES = {
    "v_in": {
        "value": 0.1,
        "lower_bound": 0.001,
        "upper_bound": 0.5
    },
    "w_out": {
        "value": [0.65, 0.65],
        "lower_bound": [0.1, 0.1],
        "upper_bound": [1.0, 1.0]
    },
    "s_out": {
        "value": [0.15, 0.15],
        "lower_bound": [0.0, 0.0],
        "upper_bound": [0.32, 0.32]
    },
    "beta_out": {
        "value": [0.83, 0.17],
        "lower_bound": [0.72, 0.06],
        "upper_bound": [0.94, 0.28]
    },
    "v*_in": {
        "value": [0.4, 0.4],
        "lower_bound": [0.1, 0.1],
        "upper_bound": [1.0, 1.0]
    },
    "beta*_throat": {
        "value": [0.83, 0.17],
        "lower_bound": [0.72, 0.06],
        "upper_bound": [0.94, 0.28]
    },
    "w*_throat": {
        "value": [0.65, 0.65],
        "lower_bound": [0.1, 0.1],
        "upper_bound": [1.0, 1.0]
    },
    "s*_throat": {
        "value": [0.15, 0.15],
        "lower_bound": [0.0, 0.0],
        "upper_bound": [0.32, 0.32]
    },
    "specific_speed": {
        "value": 1.2,
        "lower_bound": 0.01,
        "upper_bound": 10
    },
    "blade_jet_ratio": {
        "value": 0.5,
        "lower_bound": 0.1,
        "upper_bound": 0.9
    },
    "hub_tip_ratio_in": {
        "value": [0.6, 0.6],
        "lower_bound": [0.6, 0.6],
        "upper_bound": [0.9, 0.9]
    },
    "hub_tip_ratio_out": {
        "value": [0.6, 0.6],
        "lower_bound": [0.6, 0.6],
        "upper_bound": [0.9, 0.9]
    },
    "aspect_ratio": {
        "value": [1.5, 1.5],
        "lower_bound": [1.0, 1.0],
        "upper_bound": [2.0, 2.0]
    },
    "pitch_chord_ratio": {
        "value": [0.9, 0.9],
        "lower_bound": [0.75, 0.75],
        "upper_bound": [1.10, 1.10]
    },
    "trailing_edge_thickness_opening_ratio": {
        "value": [0.1, 0.1],
        "lower_bound": [0.05, 0.05],
        "upper_bound": [0.4, 0.4]
    },
    "leading_edge_angle": {
        "value": [0.41, 0.5],
        "lower_bound": [0.41, 0.08],
        "upper_bound": [0.92, 0.58]
    },
    "gauging_angle": {
        "value": [0.17, 0.94],
        "lower_bound": [0.06, 0.72],
        "upper_bound": [0.28, 0.94]
    }, 
    "throat_location_fraction": {
        "value": [1.0, 1.0],
        "lower_bound": None,
        "upper_bound": None
    },
    "leading_edge_diameter": {
        "value": [2*0.00600448, 2*0.00600448],
        "lower_bound": None,
        "upper_bound": None
    },
    "leading_edge_wedge_angle": {
        "value": [45.00, 45.00],
        "lower_bound": None,
        "upper_bound": None
    },
    "tip_clearance": {
        "value": [0.00, 5e-4],
        "lower_bound": None,
        "upper_bound": None
    },
    "cascade_type": {
        "value": ["stator", "rotor"],
        "lower_bound": None,
        "upper_bound": None
    }
}
"""
Default set of variables for design optimization.
"""

def check_string(input : str, input_type, options) -> str:
    """
    Check if a string is within a list of options.

    This function checks whether the provided string is present in the list of options. 
    It raises an AssertionError if the string is not found in the options list.

    Parameters
    ----------
    input : str
        The string to check.
    input_type : str
        The type of the input, used for error message formatting.
    options : list
        List of valid options.

    Returns
    -------
    str
        The input string if it is found in the options list.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    assert input in options, f"Invalid {input_type}: '{input}'. Available options: {options}"
    return input
    
class OperationPoints(BaseModel):
    """
    Model describing the operational points of an axial turbine.

    Attributes
    ----------
    fluid_name : str
        Name of the fluid.
    T0_in : float
        Inlet total temperature (must be greater than 0).
    p0_in : float
        Inlet total pressure (must be greater than 0).
    p_out : float
        Outlet pressure (must be greater than 0).
    omega : float
        Rotational speed (must be greater than 0).
    alpha_in : float
        Inlet angle.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """

    fluid_name: str
    T0_in: Annotated[float, Field(gt=0)]
    p0_in: Annotated[float, Field(gt=0)]
    p_out: Annotated[float, Field(gt=0)]
    omega: Annotated[float, Field(gt=0)]
    alpha_in: float

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class PerformanceMap(BaseModel):
    """
    Model describing a performance map for axial turbine analysis.

    Attributes
    ----------
    fluid_name : Union[str, List[str]]
        Name of the fluid or list of fluid names.
    T0_in : float, list
        Inlet total temperature or list of inlet total temperatures (each must be greater than 0).
    p0_in : float, list
        Inlet total pressure or list of inlet total pressures (each must be greater than 0).
    p_out : float, list
        Outlet pressure or list of outlet pressures (each must be greater than 0).
    omega : float, list
        Rotational speed or list of rotational speeds (each must be greater than 0).
    alpha_in : float, list
        Inlet angle or list of inlet angles.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    fluid_name: Union[str, List[str]]
    T0_in: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    p0_in: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    p_out: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    omega: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    alpha_in: Union[float, List[float]]

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class TuningFactors(BaseModel):

    """
    Model describing tuning factors of loss coefficients. 

    Attributes
    ----------
    profile : float, optional
        Tuning factor for profile loss coefficient. Default is 1.00.
    incidence : float, optional
        Tuning factor for incidence loss coefficient. Default is 1.00.
    secondary : float, optional
        Tuning factor for secondary loss coefficient. Default is 1.00.
    trailing : float, optional
        Tuning factor for trailing loss coefficient. Default is 1.00.
    clearance : float, optional
        Tuning factor for clearance loss coefficient. Default is 1.00.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """

    profile: float = 1.00
    incidence: float = 1.00
    secondary: float = 1.00
    trailing: float = 1.00
    clearance: float = 1.00

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class LossModel(BaseModel):
    """
    Model describing loss model setup and related parameters. 

    Attributes
    ----------
    model : str
        The type of loss model. Should be one of the predefined loss models.
    loss_coefficient : str
        The loss coefficient type. Should be one of the predefined loss coefficients.
    custom_value : float
        Loss coefficient value if model is 'custom'
    inlet_displacement_thickness_height_ratio : float, optional
        Ratio of inlet displacement thickness to height. Default is 0.011.
    tuning_factors : TuningFactors, optional
        Object containing tuning factors for loss coefficients. Default is a new instance of TuningFactors.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """

    model : Annotated[str, AfterValidator(lambda input: check_string(input, "loss model", LOSS_MODELS))]
    loss_coefficient: Annotated[str, AfterValidator(lambda input: check_string(input, "loss coefficient",LOSS_COEFFICIENTS))]
    custom_value : Annotated[float, Field(ge = 0)] = 0.1
    inlet_displacement_thickness_height_ratio: float = 0.011
    tuning_factors: TuningFactors = TuningFactors()

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class SimulationOptions(BaseModel):
    """
    Model describing simulation options.

    Attributes
    ----------
    deviation_model : str
        The type of deviation model. Should be one of the predefined deviation models.
    blockage_model : str, float, optional
        The blockage model. Can be a string or a float value. Default is 0.00.
    choking_model : str
        The choking model. Should be one of the predefined choking models.
    rel_step_fd : float, optional
        Relative step for finite difference calculations (only for evaluate_cascade_critical choking model). Default is 1e-4.
    loss_model : LossModel
        Object describing loss model setup and related parameters.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    deviation_model: Annotated[str, AfterValidator(lambda input: check_string(input, "deviation model", DEVIATION_MODELS))]
    blockage_model: Union[str, float] = 0.00
    choking_model: Annotated[str, AfterValidator(lambda input: check_string(input, "choking model", CHOKING_MODELS))]
    rel_step_fd: float = 1e-4
    loss_model: LossModel

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class SolverOptionsPerformanceAnalysis(BaseModel):
    """
    Model describing solver options for performance analysis.

    Attributes
    ----------
    method : str, optional
        The root solver method. Should be one of the predefined solver options. Default is 'hybr'.
    tolerance : float, optional
        Tolerance for convergence. Default is 1e-8.
    max_iterations : int, optional
        Maximum number of iterations. Default is 100.
    derivative_method : str, optional
        The method for finite difference approximation calculation. Should be one of the predefined derivative methods. Default is '2-point'.
    derivative_abs_step : float, optional
        Absolute step size for finite difference approximation calculation. Default is 1e-6.
    print_convergence : bool, optional
        Flag indicating whether to print convergence information. Default is True.
    plot_convergence : bool, optional
        Flag indicating whether to plot convergence. Default is False.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    method: Annotated[str, AfterValidator(lambda input: check_string(input, "root solver",SOLVER_OPTIONS)), Field(default = 'hybr')]
    tolerance: float = 1e-8
    max_iterations: int = 100
    derivative_method: Annotated[str, AfterValidator(lambda input: check_string(input, "derivative method", DERIVATIVE_METHODS)), Field(default = '2-point')]
    derivative_abs_step: float = 1e-6
    print_convergence: bool = True
    plot_convergence: bool = False

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class GeometryPerformanceAnalysis(BaseModel):
    """
    Model describing geometry parameters for performance analysis.

    Attributes
    ----------
    cascade_type : list[str]
        Types of cascades, either 'stator' or 'rotor'.
    radius_hub_in : list[float]
        Hub radius of the cascades inlet (each must be greater than 0).
    radius_hub_out : list[float]
        Hub radius of the cascades exit (each must be greater than 0). 
    radius_tip_in : list[float]
        Tip radius of the cascades inlet (each must be greater than 0).
    radius_tip_out : list[float]
        Tip radius of the cascades exit (each must be greater than 0).
    pitch : list[float]
        Pitch of the cascades (each must be greater than 0).
    chord : list[float]
        Chord of the cascades (each must be greater than 0).
    stagger_angle : list[float]
        Stagger angle of the cascades, in degrees (each must be between -90 and 90).
    opening : list[float]
        Throat opening of the cascades (each must be greater than 0).
    leading_edge_angle : list[float]
        Leading edge angle of the cascades, in degrees (each must be between -90 and 90).
    leading_edge_wedge_angle : list[float]
        Leading edge wedge angle of the cascades, in degrees (each must be between 0 and 90).
    leading_edge_diameter : list[float]
        Diameter of the leading edge of the cascades (each must be greater than 0).
    trailing_edge_thickness : list[float]
        Thickness of the trailing edge of the cascades (each must be greater than 0).
    maximum_thickness : list[float]
        Maximum thickness of the cascades (each must be greater than 0).
    tip_clearance : list[float]
        Tip clearance of the cascades (each must be greater than or equal 0).
    throat_location_fraction : list[float]
        Fractional location of the throat in the cascades (each must be between 0 and 1). 

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
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

    @model_validator(mode='after')
    def check_length(self) -> Self:
        attributes = vars(self).values()
        if not all(len(attr) == len(self.cascade_type) for attr in attributes):
            raise ValueError('Geometry input is not of same length')
        return self

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class SolverOptionsOptimization(BaseModel):
    """
    Model describing solver options for optimization.

    Checks that the given solver method corresponds with the given library. 

    Attributes
    ----------
    library : str, optional
        The optimization library to be used. Should be one of the predefined libraries. Default is 'scipy'.
    method : str, optional
        The optimization method to be used. Default is 'slsqp'.
    tolerance : float, optional
        Tolerance for convergence. Default is 1e-5.
    max_iterations : int, optional
        Maximum number of iterations. Default is 100.
    derivative_method : str, optional
        The method for finite difference calculation. Should be one of the predefined derivative methods. Default is '2-point'.
    derivative_abs_step : float, optional
        Absolute step size for finite difference calculation. Default is None.
    print_convergence : bool, optional
        Flag indicating whether to print convergence information. Default is True.
    plot_convergence : bool, optional
        Flag indicating whether to plot convergence. Default is False.
    update_on : str, optional
        The criteria for updating. Should be either 'gradient' or 'function'. Default is 'gradient'.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    library : Annotated[str, AfterValidator(lambda input: check_string(input, "optimization library",VALID_LIBRARIES_AND_METHODS.keys())), Field(default = 'scipy')]
    method : str = "slsqp"
    tolerance : float = 1e-5
    max_iterations : int = 100
    derivative_method :  Annotated[str, AfterValidator(lambda input: check_string(input, "derivative method",DERIVATIVE_METHODS)), Field(default = '2-point')]
    derivative_abs_step : float = None
    print_convergence : bool = True
    plot_convergence : bool = False
    update_on : Annotated[str, AfterValidator(lambda input: check_string(input, "update_on input",["gradient", "function"])), Field(default = 'gradient')]

    @model_validator(mode='after')
    def check_solver_method(self) -> Self:
        lib = self.library
        method = self.method
        if not method in VALID_LIBRARIES_AND_METHODS[lib]:
            raise ValueError(f'Method {method} is not available in {lib} library.')
        return self    

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class PerformanceAnalysis(BaseModel):
    """
    Model describing performance analysis parameters.

    Attributes
    ----------
    performance_map : PerformanceMap, optional
        Object containing performance map data.
    solver_options : SolverOptionsPerformanceAnalysis, optional
        Object containing solver options for performance analysis.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    performance_map: PerformanceMap = None
    solver_options: SolverOptionsPerformanceAnalysis = SolverOptionsPerformanceAnalysis()

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class Variable(BaseModel):
    """
    Model describing a variable.

    If the attributes are lists, each value corresponds to a seperate cascade.

    Checks that the type and length of the attributes are the same. 

    Attributes
    ----------
    value : float, str, list[float], list[str]
        The value(s) of the variable, which could be a float, string, list of floats, or list of strings.
    lower_bound : float, list[float], optional
        The lower bound of the variable. Default is None.
    upper_bound : float, list[float], optional
        The upper bound of the variable. Default is None.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    value : Union[float, Annotated[str, AfterValidator(lambda input: check_string(input, "cascade type", ["stator", "rotor"]))], List[float], List[Annotated[str, AfterValidator(lambda input: check_string(input, "cascade type", ["stator", "rotor"]))]]]
    lower_bound : Union[float, List[float]] = None
    upper_bound : Union[float, List[float]] = None

    @model_validator(mode='after')
    def check_length(self) -> Self:
        attributes = vars(self).values()
        input_type = type(self.value)
        if self.lower_bound is not None:
            if not all(isinstance(attr, input_type) for attr in attributes):
                raise ValueError('Variable input is not of same type')
            if isinstance(self.value, list):
                if not all(len(attr) == len(self.value) for attr in attributes):
                    raise ValueError('Variable input is not of same length')
        return self

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class Constraint(BaseModel):
    """
    Model describing a constraint.

    Attributes
    ----------
    type : str
        The type of constraint, indicating whether it is less than ('<'), equal to ('='), or greater than ('>').
    value : float
        The value of the constraint.
    normalize : bool
        Flag indicating whether the constraint should be normalized.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    type : Literal["<", "=", ">"]
    value : float
    normalize : bool

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class DesignOptimization(BaseModel):
    """
    Model describing design optimization parameters.

    Checks that variables contain all variables from a predifned list (VARIABLES). Returns a default dictionary if not. 

    Attributes
    ----------
    objective_function : str, optional
        The objective function for optimization. Should be one of the predefined objective functions. Default is 'efficiency_ts'.
    variables : dict[str], optional
        Dictionary containing variables for optimization. Should contain a dict for each variable in a predefined variable list (VARIABLES).
    constraints : dict[str], optional
        Dictionary containing constraints for optimization.
    solver_options : SolverOptionsOptimization, optional
        Object containing solver options for optimization.
    radius_type : str, optional
        The type of radius used in optimization. Should be one of the predefined radius types. Default is 'constant_mean'.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    objective_function : Annotated[str, AfterValidator(lambda input: check_string(input, "objective function",OBJECTIVE_FUNCTIONS)), Field(default = 'efficiency_ts')]
    variables : Dict[Annotated[str, AfterValidator(lambda input: check_string(input, "variable", VARIABLES))], Variable] = None
    constraints : Dict[Annotated[str, AfterValidator(lambda input: check_string(input, "constraint", CONSTRAINTS))], Constraint] = None
    solver_options : SolverOptionsOptimization = SolverOptionsOptimization()
    radius_type : Annotated[str, AfterValidator(lambda input: check_string(input, "radius type", RADIUS_TYPE)), Field(default = 'constant_mean')]     

    # After validator to check variables holds all variables 
    @field_validator('variables')
    def check_variables(cls, v : Dict) -> Dict:
        if v == None:
            # Give completely default turbine
            v = DEFAULT_VARIABLES
            print("Design optimization variables are not provided. Defualt set of variables are used.")
        lacking_keys = set(VARIABLES) - set(v.keys())
        if len(lacking_keys) > 0:
            # Fill in lacking keys
            v = DEFAULT_VARIABLES
            print(f"Design optimization variables are missing: {lacking_keys}. Defualt set of variables are used.")
        return v

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class AxialTurbine(BaseModel):
    """
    Model describing an axial turbine.

    Attributes
    ----------
    turbomachinery : str
        The type of turbomachinery, which for this class is "axial_turbine".
    operation_points : OperationPoints
        Object containing operation points data.
    simulation_options : SimulationOptions
        Object containing simulation options data.
    geometry : GeometryPerformanceAnalysis, optional
        Object containing geometry performance analysis data. Default is None.
    performance_analysis : PerformanceAnalysis, optional
        Object containing performance analysis data. Default is a new instance of PerformanceAnalysis().
    design_optimization : DesignOptimization, optional
        Object containing design optimization data. Default is a new instance of DesignOptimization().

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """
    turbomachinery : Literal["axial_turbine"]
    operation_points: OperationPoints
    simulation_options: SimulationOptions 
    geometry: GeometryPerformanceAnalysis = None
    performance_analysis : PerformanceAnalysis = PerformanceAnalysis() 
    design_optimization : DesignOptimization = DesignOptimization()

    # Avoid any other inputs
    class Config:
        extra = "forbid"



