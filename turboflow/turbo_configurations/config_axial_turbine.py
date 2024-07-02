from pydantic import (
    BaseModel,
    ValidationError,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
)
from typing import Optional, List, Union, Literal, Dict
from enum import Enum
from typing_extensions import Self
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator, BeforeValidator
from ..axial_turbine import (
    LOSS_MODELS,
    LOSS_COEFFICIENTS,
    CHOKING_MODELS,
    DEVIATION_MODELS,
    RADIUS_TYPE,
)
from ..pysolver_view import (
    SOLVER_OPTIONS,
    VALID_LIBRARIES_AND_METHODS,
    DERIVATIVE_METHODS,
)

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

    model_config = ConfigDict(extra="forbid")
    fluid_name: str
    T0_in: Annotated[float, Field(gt=0)]
    p0_in: Annotated[float, Field(gt=0)]
    p_out: Annotated[float, Field(gt=0)]
    omega: Annotated[float, Field(gt=0)]
    alpha_in: float


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

    model_config = ConfigDict(extra="forbid")
    fluid_name: Union[str, List[str]]
    T0_in: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    p0_in: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    p_out: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    omega: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    alpha_in: Union[float, List[float]]


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

    model_config = ConfigDict(extra="forbid")
    profile: float = 1.00
    incidence: float = 1.00
    secondary: float = 1.00
    trailing: float = 1.00
    clearance: float = 1.00

LossModelEnum = Enum('LossModels', dict(zip([model.upper() for model in LOSS_MODELS], LOSS_MODELS)))
LossCoefficientEnum = Enum('LossCoefficients', dict(zip([model.upper() for model in LOSS_COEFFICIENTS], LOSS_COEFFICIENTS)))

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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    model : LossModelEnum
    loss_coefficient : LossCoefficientEnum
    custom_value: Annotated[float, Field(ge=0)] = 0.0
    inlet_displacement_thickness_height_ratio: float = 0.011
    tuning_factors: TuningFactors = TuningFactors()

DeviationModelEnum = Enum('DeviationModels', dict(zip([model.upper() for model in DEVIATION_MODELS], DEVIATION_MODELS)))
ChokingModelEnum = Enum('ChokingModels', dict(zip([model.upper() for model in CHOKING_MODELS], CHOKING_MODELS)))

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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    deviation_model : DeviationModelEnum
    blockage_model: Union[str, float] = 0.00
    choking_model : ChokingModelEnum
    rel_step_fd: float = 1e-4
    loss_model: LossModel

SolverOptionEnum = Enum('SolverOptions', dict(zip([model.upper() for model in SOLVER_OPTIONS], SOLVER_OPTIONS)))
DerivativeMethodEnum = Enum('DerivativeMethods', dict(zip([model.upper() for model in DERIVATIVE_METHODS], DERIVATIVE_METHODS)))

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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    method : SolverOptionEnum = SOLVER_OPTIONS[0]
    tolerance: float = 1e-8
    max_iterations: int = 100
    derivative_method : DerivativeMethodEnum = DERIVATIVE_METHODS[0]
    derivative_abs_step: float = 1e-6
    print_convergence: bool = True
    plot_convergence: bool = False

CascadeTypeEnum = Enum('CascadeTypes', dict(zip([model.upper() for model in ["stator", "rotor"]], ["stator", "rotor"])))

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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    cascade_type: List[CascadeTypeEnum]
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

    @model_validator(mode="after")
    def check_length(self) -> Self:
        attributes = vars(self).values()
        if not all(len(attr) == len(self.cascade_type) for attr in attributes):
            raise ValueError("Geometry input is not of same length")
        return self

LibraryEnum = Enum('LibraryOptions', dict(zip([model.upper() for model in VALID_LIBRARIES_AND_METHODS.keys()], VALID_LIBRARIES_AND_METHODS.keys())))
MethodEnum = Enum('MethodOption', dict(zip([model.upper() for model in [x for xs in VALID_LIBRARIES_AND_METHODS.values() for x in xs]], [x for xs in VALID_LIBRARIES_AND_METHODS.values() for x in xs])))
UpdateOnEnum = Enum('UpdateOns', dict(zip([model.upper() for model in ["gradient", "function"]], ["gradient", "function"])))

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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    library: LibraryEnum = list(VALID_LIBRARIES_AND_METHODS.keys())[0]
    method: MethodEnum = VALID_LIBRARIES_AND_METHODS[list(VALID_LIBRARIES_AND_METHODS.keys())[0]][0]
    tolerance: float = 1e-5
    max_iterations: int = 100
    derivative_method: DerivativeMethodEnum = DERIVATIVE_METHODS[0]
    derivative_abs_step: float = None
    print_convergence: bool = True
    plot_convergence: bool = False
    update_on: UpdateOnEnum = "gradient"

    @model_validator(mode="after")
    def check_solver_method(self) -> Self:
        lib = self.library
        method = self.method
        if not method in VALID_LIBRARIES_AND_METHODS[lib]:
            raise ValueError(f"Method {method} is not available in {lib} library.")
        return self


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

    model_config = ConfigDict(extra="forbid")
    performance_map: PerformanceMap = None
    solver_options: SolverOptionsPerformanceAnalysis = (
        SolverOptionsPerformanceAnalysis()
    )


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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    value : Union[float, CascadeTypeEnum, List[float], List[CascadeTypeEnum]]
    lower_bound: Union[float, List[float]] = None
    upper_bound: Union[float, List[float]] = None

    @model_validator(mode="after")
    def check_length(self) -> Self:
        attributes = vars(self).values()
        input_type = type(self.value)
        if self.lower_bound is not None:
            if not all(isinstance(attr, input_type) for attr in attributes):
                raise ValueError("Variable input is not of same type")
            if isinstance(self.value, list):
                if not all(len(attr) == len(self.value) for attr in attributes):
                    raise ValueError("Variable input is not of same length")
        return self


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

    model_config = ConfigDict(extra="forbid")
    variable : str
    type: Literal["<", "=", ">"]
    value: float
    normalize: bool

RadiusTypeEnum = Enum('RadiusTypes', dict(zip([model.upper() for model in RADIUS_TYPE], RADIUS_TYPE)))

class ObjectiveFunction(BaseModel):
    """
    Model describing the objective function of a optimization problem.

    Attributes
    ----------
    variable : str
        Defines the objective function variable. Must be contained in the `results` structure of the problem and be on the form 'section.parameter', e.g. 'overall.efficiency_ts'.
    type : str
        Whether the objective function should be maxmized or minimized.
    scale : float
        Value to scale the objective function.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """

    model_config = ConfigDict(extra="forbid")
    variable : str = 'overall.efficieny_ts'
    type: Literal["maximize", "minimize"] = 'maximize'
    scale: float = 1

class Variables(BaseModel):

    model_config = ConfigDict(extra="forbid")
    v_in : Variable = Variable(value = 0.1, lower_bound = 0.001, upper_bound=0.5)
    w_out : Variable = Variable(value = [0.65, 0.65], lower_bound = [0.1, 0.1], upper_bound=[1.0, 1.0])
    s_out : Variable = Variable(value = [0.15, 0.15], lower_bound = [0.0, 0.0], upper_bound=[0.32, 0.32])
    beta_out : Variable = Variable(value = [0.83, 0.17], lower_bound = [0.72, 0.06], upper_bound=[0.94, 0.28])
    v_crit_in : Variable = Variable(value = [0.4, 0.4], lower_bound = [0.1, 0.1], upper_bound=[1.0, 1.0])
    beta_crit_throat : Variable = Variable(value = [0.83, 0.17], lower_bound = [0.72, 0.06], upper_bound=[0.94, 0.28])
    w_crit_throat : Variable = Variable(value = [0.65, 0.65], lower_bound = [0.1, 0.1], upper_bound=[1.0, 1.0])
    s_crit_throat : Variable = Variable(value = [0.15, 0.15], lower_bound = [0.0, 0.0], upper_bound=[0.32, 0.32])
    specific_speed : Variable = Variable(value = 1.2, lower_bound=0.01, upper_bound=10)
    blade_jet_ratio : Variable = Variable(value = 0.5)
    hub_tip_ratio_in : Variable = Variable(value = [0.6, 0.6])
    hub_tip_ratio_out : Variable = Variable(value = [0.6, 0.6])
    aspect_ratio : Variable = Variable(value = [1.5, 1.5])
    pitch_chord_ratio : Variable = Variable(value = [0.9, 0.9])
    trailing_edge_thickness_opening_ratio : Variable = Variable(value = [0.1, 0.1])
    leading_edge_angle : Variable = Variable(value = [0.41, 0.5])
    gauging_angle : Variable = Variable(value = [0.17, 0.94])
    throat_location_fraction : Variable = Variable(value = [1.0, 1.0])
    leading_edge_diameter : Variable = Variable(value = [2*0.00600448, 2*0.00600448])
    leading_edge_wedge_angle : Variable = Variable(value = [45.00, 45.00] )
    tip_clearance : Variable = Variable(value = [0.00, 5e-4])
    cascade_type : Variable = Variable(value = ["stator", "rotor"])

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

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    objective_function : ObjectiveFunction = ObjectiveFunction()
    variables : Variables = Variables()
    constraints : List[Constraint] = None
    solver_options: SolverOptionsOptimization = SolverOptionsOptimization()
    radius_type : RadiusTypeEnum = RADIUS_TYPE[0]



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

    model_config = ConfigDict(extra="forbid")
    turbomachinery: Literal["axial_turbine"]
    operation_points: Union[List[OperationPoints], OperationPoints]
    simulation_options: SimulationOptions
    geometry: GeometryPerformanceAnalysis = None
    performance_analysis: PerformanceAnalysis = PerformanceAnalysis()
    design_optimization: DesignOptimization = DesignOptimization()
