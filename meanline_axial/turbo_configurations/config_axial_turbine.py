
from pydantic import BaseModel, ValidationError, Field, model_validator
from typing import Optional, List, Union, Literal, Dict
from typing_extensions import Self
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator, BeforeValidator

LOSS_MODELS = ["benner","moustapha",  "kacker_okapuu", "isentropic","custom"]
LOSS_COEFFICIENTS = ["stagnation_pressure"]
DEVIATION_MODELS = ["aungier", "ainley_mathieson", "zero_deviation"]
CHOKING_MODELS = ["evaluate_cascade_critical",
                  "evaluate_cascade_throat",
                  "evaluate_cascade_isentropic_throat"]
ROOT_SOLVERS = ["hybr", "lm"]
DERIVATIVE_METHODS = ["2-point", "3-point"]
OBJECTIVE_FUNCTIONS = ["none", "efficiency_ts"]
VARIABLES = ["specific_speed", 
                    "blade_jet_ratio",
                    "hub_tip_ratio_in",
                    "hub_tip_ratio_out",
                    "aspect_ratio",
                    "pitch_chord_ratio",
                    "trailing_edge_thickness_opening_ratio",
                    "leading_edge_angle",
                    "gauging_angle",
                    "cascade_type"
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

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class PerformanceMap(BaseModel):
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
    profile: float = 1.00
    incidence: float = 1.00
    secondary: float = 1.00
    trailing: float = 1.00
    clearance: float = 1.00

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class LossModel(BaseModel):
    model : Annotated[str, AfterValidator(lambda input: check_string(input, "loss model", LOSS_MODELS))]
    loss_coefficient: Annotated[str, AfterValidator(lambda input: check_string(input, "loss coefficient",LOSS_COEFFICIENTS))]
    inlet_displacement_thickness_height_ratio: float = 0.011
    tuning_factors: TuningFactors = TuningFactors()

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class SimulationOptions(BaseModel):
    deviation_model: Annotated[str, AfterValidator(lambda input: check_string(input, "deviation model", DEVIATION_MODELS))]
    blockage_model: Union[str, float] = 0.00
    choking_model: Annotated[str, AfterValidator(lambda input: check_string(input, "choking model", CHOKING_MODELS))]
    rel_step_fd: float = 1e-4
    loss_model: LossModel

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class SolverOptionsPerformanceAnalysis(BaseModel):
    method: Annotated[str, AfterValidator(lambda input: check_string(input, "root solver",ROOT_SOLVERS)), Field(default = 'hybr')]
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
    library : Annotated[str, AfterValidator(lambda input: check_string(input, "optimization library",LIBRARIES)), Field(default = 'scipy')]
    method : str = "slsqp" # TODO: Update when integrated into TurboFlow
    tolerance : float = 1e-5
    max_iterations : int = 100
    derivative_method :  Annotated[str, AfterValidator(lambda input: check_string(input, "derivative method",DERIVATIVE_METHODS)), Field(default = '2-point')]
    derivative_abs_step : float = None
    print_convergence : bool = True
    plot_convergence : bool = False
    update_on : Annotated[str, AfterValidator(lambda input: check_string(input, "update_on input",["gradient", "function"])), Field(default = 'gradient')]    

    # Avoid any other inputs
    class Config:
        extra = "forbid"

# class GeometryOptimization(BaseModel):

#     hub_tip_ratio_in : List[Annotated[float, Field(le=1), Field(gt=0)]]
#     hub_tip_ratio_out : List[Annotated[float, Field(le=1), Field(gt=0)]]
#     aspect_ratio : List[Annotated[float, Field(gt=0)]]
#     pitch_chord_ratio : List[Annotated[float, Field(gt=0)]]
#     gauging_angle : List[Annotated[float, Field(le=90), Field(gt=-90)]]
#     leading_edge_angle : List[Annotated[float, Field(le=90), Field(gt=-90)]]
#     trailing_edge_thickness_opening_ratio : List[Annotated[float, Field(le=1), Field(gt=0)]]
#     cascade_type : List[Annotated[str, AfterValidator(lambda input: check_string(input, "cascade type", ["stator", "rotor"]))]]
#     leading_edge_diameter : List[Annotated[float, Field(gt=0)]]
#     leading_edge_wedge_angle : List[Annotated[float, Field(le=90), Field(gt=0)]]
#     tip_clearance : List[Annotated[float, Field(ge=0)]]
#     throat_location_fraction : List[Annotated[float, Field(le=1), Field(gt=0)]]

#     @model_validator(mode='after')
#     def check_length(self) -> Self:
#         attributes = vars(self).values()
#         if isinstance(self.value, list):
#             if not all(len(attr) == len(self.value) for attr in attributes):
#                 raise ValueError('Geometry input is not of same length')
#         return self

#     # Avoid any other inputs
#     class Config:
#         extra = "forbid"

class PerformanceAnalysis(BaseModel):
    performance_map: PerformanceMap = OperationPoints
    solver_options: SolverOptionsPerformanceAnalysis = SolverOptionsPerformanceAnalysis()

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class Variable(BaseModel):
    value : Union[float, List[float]]
    lower_bound : Union[float, List[float]] = None
    upper_bound : Union[float, List[float]] = None

    @model_validator(mode='after')
    def check_length(self) -> Self:
        attributes = vars(self).values()
        input_type = type(self.value)
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
    # variable : Annotated[str, AfterValidator(lambda input: check_string(input, "constraint", CONSTRAINTS))]
    type : Literal["<", "=", ">"]
    value : float
    normalize : bool

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class DesignOptimization(BaseModel):
    objective_function : Annotated[str, AfterValidator(lambda input: check_string(input, "objective function",OBJECTIVE_FUNCTIONS)), Field(default = 'efficiency_ts')]
    variables : Dict[Annotated[str, AfterValidator(lambda input: check_string(input, "variable", VARIABLES))], Variable] = None
    constraints : Dict[Annotated[str, AfterValidator(lambda input: check_string(input, "constraint", CONSTRAINTS))], Constraint] = None
    solver_options : SolverOptionsOptimization = SolverOptionsOptimization()
    radius_type : Annotated[str, AfterValidator(lambda input: check_string(input, "radius type", RADIUS_TYPE)), Field(default = 'constant_mean')]     

    # Avoid any other inputs
    class Config:
        extra = "forbid"

class AxialTurbine(BaseModel):
    turbomachinery : Literal["axial_turbine"]
    operation_points: OperationPoints
    simulation_options: SimulationOptions 
    geometry: GeometryPerformanceAnalysis = None
    performance_analysis : PerformanceAnalysis = PerformanceAnalysis() 
    design_optimization : DesignOptimization = DesignOptimization()

    # Avoid any other inputs
    class Config:
        extra = "forbid"



