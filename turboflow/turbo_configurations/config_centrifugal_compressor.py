
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
    root_validator,
)
from typing import Optional, List, Union, Literal, Tuple, Dict
from typing_extensions import Annotated
from enum import Enum

from ..centrifugal_compressor import (
    LOSS_MODELS,
    LOSS_COEFFICIENTS,
    SLIP_MODELS,
    CHOKING_CRITERIONS,
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
    mass_flow_rate : float
        Mass flow rate (must be greater than 0).
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
    mass_flow_rate: Annotated[float, Field(gt=0)]
    omega: Annotated[float, Field(gt=0)]
    alpha_in: float

LossModelEnum = Enum('LossModels', dict(zip([model.upper() for model in LOSS_MODELS], LOSS_MODELS)))
LossCoefficientEnum = Enum('LossCoefficients', dict(zip([model.upper() for model in LOSS_COEFFICIENTS], LOSS_COEFFICIENTS)))

# Define the basic loss configuration for each component
class LossConfig1(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    model: LossModelEnum = "isentropic"
    loss_coefficient: LossCoefficientEnum = "static_enthalpy_loss"

class LossConfig2(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    impeller: LossConfig1 = LossConfig1()
    vaneless_diffuser: LossConfig1 = LossConfig1()
    vaned_diffuser: LossConfig1 = LossConfig1()
    volute: LossConfig1 = LossConfig1()

class Factors(BaseModel):

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    skin_friction : Annotated[float, Field(ge=0)] = 0.00
    wall_heat_flux : Annotated[float, Field(ge=0)] = 0.00
    wake_width : Annotated[float, Field(gt=0)] = 0.366

SlipModelEnum = Enum('SlipModels', dict(zip([model.upper() for model in SLIP_MODELS], SLIP_MODELS)))
ChokingCriterionEnum = Enum('ChokingCriterionModels', dict(zip([model.upper() for model in CHOKING_CRITERIONS], CHOKING_CRITERIONS)))


class SimulationOptions(BaseModel):
    """
    Model describing simulation options.

    Attributes
    ----------
    slip_model : str
        The type of slip model. Should be one of the predefined slip models.
    loss_model : LossModel
        Object describing loss model setup and related parameters.

    Configurations
    --------------
    extra : str, optional
        Indicates that no extra input is allowed. Default is "forbid".
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    slip_model : SlipModelEnum
    loss_model: Union[LossConfig1, LossConfig2]
    vaneless_diffuser_model : Literal["algebraic", "1D"]
    factors : Factors
    rel_step_fd : Annotated[float, Field(gt=0)] = 1e-4
    choking_criterion : ChokingCriterionEnum 

class Impeller(BaseModel):

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    radius_hub_in: Annotated[float, Field(gt=0)] 
    radius_tip_in: Annotated[float, Field(gt=0)]
    radius_out : Annotated[float, Field(gt=0)]
    width_out : Annotated[float, Field(gt=0)]
    leading_edge_angle : Annotated[float, Field(le=90), Field(gt=-90)]
    trailing_edge_angle : Annotated[float, Field(le=90), Field(gt=-90)]
    number_of_blades : Annotated[float, Field(gt=0)]
    length_axial : Annotated[float, Field(gt=0)]
    length_meridional : Annotated[float, Field(gt=0)]
    tip_clearance : Annotated[float, Field(gt=0)]
    area_throat_ratio : Annotated[float, Field(gt=0)]

class VanedDiffuser(BaseModel):

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    radius_out : Annotated[float, Field(gt=0)]
    radius_in : Annotated[float, Field(gt=0)]
    width_in : Annotated[float, Field(gt=0)]
    width_out : Annotated[float, Field(gt=0)]
    leading_edge_angle : Annotated[float, Field(le=90), Field(gt=-90)]
    trailing_edge_angle : Annotated[float, Field(le=90), Field(gt=-90)]
    number_of_vanes : Annotated[float, Field(gt=0)]
    opening : Annotated[float, Field(gt=0)]
    throat_location_factor : Annotated[float, Field(le=1), Field(gt=0)]
    area_throat_ratio : Annotated[float, Field(gt=0)]

class VanelessDiffuser(BaseModel):

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    # wall_divergence : Annotated[float, Field(le=90), Field(gt=-90)]
    radius_out : Annotated[float, Field(gt=0)]
    radius_in : Annotated[float, Field(gt=0)]
    width_in : Annotated[float, Field(gt=0)]
    width_out : Annotated[float, Field(gt=0)]

class Volute(BaseModel):

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    radius_out : Annotated[float, Field(gt=0)] 
    radius_in : Annotated[float, Field(gt=0)]
    radius_scroll : Annotated[float, Field(gt=0)]
    width_in : Annotated[float, Field(gt=0)]

class Geometry(BaseModel):

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    impeller : Impeller
    vaneless_diffuser : VanelessDiffuser = None
    vaned_diffuser : VanedDiffuser = None
    volute : Volute = None

    # Model validator to remove None fields after validation
    @model_validator(mode="after")
    def remove_none_fields(self):
        # Collect fields with None values to remove after iteration
        fields_to_remove = [field_name for field_name, value in self.__dict__.items() if value is None]
        
        # Delete attributes with None values
        for field_name in fields_to_remove:
            delattr(self, field_name)
            
        return self

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
    mass_flow_rate : float, list
        Mass flow rate or list of mass flow rates (each must be greater than 0).
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
    mass_flow_rate: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    omega: Union[Annotated[float, Field(gt=0)], List[Annotated[float, Field(gt=0)]]]
    alpha_in: Union[float, List[float]]

SolverOptionEnum = Enum('SolverOptions', dict(zip([model.upper() for model in SOLVER_OPTIONS], SOLVER_OPTIONS)))
DerivativeMethodEnum = Enum('DerivativeMethods', dict(zip([model.upper() for model in DERIVATIVE_METHODS], DERIVATIVE_METHODS)))

class SolverOptions(BaseModel):

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

class PerformanceAnalysis(BaseModel):

    model_config = ConfigDict(extra="forbid")
    performance_map : PerformanceMap = None
    solver_options : SolverOptions = SolverOptions()
    initial_guess : Dict = {}

class CentrifugalCompressor(BaseModel):
    """
    Model describing an axial turbine.

    Attributes
    ----------
    turbomachinery : str
        The type of turbomachinery, which for this class is "centrifugal_compressor".
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
    turbomachinery: Literal["centrifugal_compressor"]
    operation_points: Union[List[OperationPoints], OperationPoints]
    simulation_options: SimulationOptions
    geometry: Geometry
    performance_analysis: PerformanceAnalysis = PerformanceAnalysis()
    # design_optimization: DesignOptimization = DesignOptimization()