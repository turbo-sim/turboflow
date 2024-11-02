import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

from cycler import cycler


COLORS_PYTHON = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

COLORS_MATLAB = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]


try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()


def set_plot_options(
    fontsize=13,
    grid=True,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
):
    """Set plot options for publication-quality figures"""

    if isinstance(color_order, str):
        if color_order.lower() == "default":
            color_order = COLORS_PYTHON

        elif color_order.lower() == "matlab":
            color_order = COLORS_MATLAB

    # Define dictionary of custom settings
    rcParams = {
        "text.usetex": False,
        "font.size": fontsize,
        "font.style": "normal",
        "font.family": "serif",  # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
        "font.serif": ["Times New Roman"],  # ['times new roman', 'cmr10']
        "mathtext.fontset": "stix",  # ["stix", 'cm']
        "axes.edgecolor": "black",
        "axes.linewidth": 1.25,
        "axes.titlesize": fontsize,
        "axes.titleweight": "normal",
        "axes.titlepad": fontsize * 1.4,
        "axes.labelsize": fontsize,
        "axes.labelweight": "normal",
        "axes.labelpad": fontsize,
        "axes.xmargin": margin,
        "axes.ymargin": margin,
        "axes.zmargin": margin,
        "axes.grid": grid,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.prop_cycle": cycler(color=color_order),
        "grid.alpha": 0.5,
        "grid.color": "black",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "legend.borderaxespad": 1,
        "legend.borderpad": 0.6,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.labelcolor": "black",
        "legend.labelspacing": 0.3,
        "legend.fancybox": True,
        "legend.fontsize": fontsize - 2,
        "legend.framealpha": 1.00,
        "legend.handleheight": 0.7,
        "legend.handlelength": 1.25,
        "legend.handletextpad": 0.8,
        "legend.markerscale": 1.0,
        "legend.numpoints": 1,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "lines.markeredgewidth": 1.25,
        "lines.markerfacecolor": "white",
        "xtick.direction": "in",
        "xtick.labelsize": fontsize - 1,
        "xtick.bottom": major_ticks,
        "xtick.top": major_ticks,
        "xtick.major.size": 6,
        "xtick.major.width": 1.25,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.75,
        "xtick.minor.visible": minor_ticks,
        "ytick.direction": "in",
        "ytick.labelsize": fontsize - 1,
        "ytick.left": major_ticks,
        "ytick.right": major_ticks,
        "ytick.major.size": 6,
        "ytick.major.width": 1.25,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.75,
        "ytick.minor.visible": minor_ticks,
        "savefig.dpi": 600,
    }

    # Update the internal Matplotlib settings dictionary
    mpl.rcParams.update(rcParams)


# Define property aliases
PROPERTY_ALIAS = {
    "P": "p",
    "rho": "rhomass",
    "d": "rhomass",
    "u": "umass",
    "h": "hmass",
    "s": "smass",
    "cv": "cvmass",
    "cp": "cpmass",
    "a": "speed_sound",
    "Z": "compressibility_factor",
    "mu": "viscosity",
    "k": "conductivity",
}

# Dynamically add INPUTS fields to the module
# for attr in dir(CP):
#     if attr.endswith('_INPUTS'):
#         globals()[attr] = getattr(CP, attr)

# Statically add phase indices to the module (IDE autocomplete)
iphase_critical_point = CP.iphase_critical_point
iphase_gas = CP.iphase_gas
iphase_liquid = CP.iphase_liquid
iphase_not_imposed = CP.iphase_not_imposed
iphase_supercritical = CP.iphase_supercritical
iphase_supercritical_gas = CP.iphase_supercritical_gas
iphase_supercritical_liquid = CP.iphase_supercritical_liquid
iphase_twophase = CP.iphase_twophase
iphase_unknown = CP.iphase_unknown

# Statically add INPUT fields to the module (IDE autocomplete)
QT_INPUTS = CP.QT_INPUTS
PQ_INPUTS = CP.PQ_INPUTS
QSmolar_INPUTS = CP.QSmolar_INPUTS
QSmass_INPUTS = CP.QSmass_INPUTS
HmolarQ_INPUTS = CP.HmolarQ_INPUTS
HmassQ_INPUTS = CP.HmassQ_INPUTS
DmolarQ_INPUTS = CP.DmolarQ_INPUTS
DmassQ_INPUTS = CP.DmassQ_INPUTS
PT_INPUTS = CP.PT_INPUTS
DmassT_INPUTS = CP.DmassT_INPUTS
DmolarT_INPUTS = CP.DmolarT_INPUTS
HmolarT_INPUTS = CP.HmolarT_INPUTS
HmassT_INPUTS = CP.HmassT_INPUTS
SmolarT_INPUTS = CP.SmolarT_INPUTS
SmassT_INPUTS = CP.SmassT_INPUTS
TUmolar_INPUTS = CP.TUmolar_INPUTS
TUmass_INPUTS = CP.TUmass_INPUTS
DmassP_INPUTS = CP.DmassP_INPUTS
DmolarP_INPUTS = CP.DmolarP_INPUTS
HmassP_INPUTS = CP.HmassP_INPUTS
HmolarP_INPUTS = CP.HmolarP_INPUTS
PSmass_INPUTS = CP.PSmass_INPUTS
PSmolar_INPUTS = CP.PSmolar_INPUTS
PUmass_INPUTS = CP.PUmass_INPUTS
PUmolar_INPUTS = CP.PUmolar_INPUTS
HmassSmass_INPUTS = CP.HmassSmass_INPUTS
HmolarSmolar_INPUTS = CP.HmolarSmolar_INPUTS
SmassUmass_INPUTS = CP.SmassUmass_INPUTS
SmolarUmolar_INPUTS = CP.SmolarUmolar_INPUTS
DmassHmass_INPUTS = CP.DmassHmass_INPUTS
DmolarHmolar_INPUTS = CP.DmolarHmolar_INPUTS
DmassSmass_INPUTS = CP.DmassSmass_INPUTS
DmolarSmolar_INPUTS = CP.DmolarSmolar_INPUTS
DmassUmass_INPUTS = CP.DmassUmass_INPUTS
DmolarUmolar_INPUTS = CP.DmolarUmolar_INPUTS

# Define dictionary with dynamically generated fields
PHASE_INDEX = {attr: getattr(CP, attr) for attr in dir(CP) if attr.startswith("iphase")}
INPUT_PAIRS = {attr: getattr(CP, attr) for attr in dir(CP) if attr.endswith("_INPUTS")}
INPUT_PAIRS = sorted(INPUT_PAIRS.items(), key=lambda x: x[1])


def states_to_dict(states):
    """
    Convert a list of state objects into a dictionary.
    Each key is a field name of the state objects, and each value is a NumPy array of all the values for that field.
    """
    state_dict = {}
    for field in states[0].keys():
        state_dict[field] = np.array([getattr(state, field) for state in states])
    return state_dict


class FluidState:
    """
    A class representing the thermodynamic state of a fluid.

    This class is used to store and access the properties of a fluid state.
    Properties can be accessed directly as attributes (e.g., `fluid_state.p` for pressure)
    or through dictionary-like access (e.g., `fluid_state['T']` for temperature).

    Methods
    -------
    to_dict():
        Convert the FluidState properties to a dictionary.
    keys():
        Return the keys of the FluidState properties.
    items():
        Return the items (key-value pairs) of the FluidState properties.

    """

    def __init__(self, properties):
        for key, value in properties.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __str__(self):
        properties_str = "\n   ".join(
            [f"{key}: {getattr(self, key)}" for key in self.__dict__]
        )
        return f"FluidState:\n   {properties_str}"

    # def get(self, key, default=None):
    #     return getattr(self, key, default)

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dict__}

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()


class Fluid:
    """
    Represents a fluid with various thermodynamic properties computed via CoolProp.

    This class provides a convenient interface to CoolProp for various fluid property calculations.

    Properties can be accessed directly as attributes (e.g., `fluid.properties["p"]` for pressure)
    or through dictionary-like access (e.g., `fluid.T` for temperature).

    Critical and triple point properties are computed upon initialization and stored internally for convenience.

    Attributes
    ----------
    name : str
        Name of the fluid.
    backend : str
        Backend used for CoolProp, default is 'HEOS'.
    exceptions : bool
        Determines if exceptions should be raised during state calculations. Default is True.
    converged_flag : bool
        Flag indicating whether properties calculations converged.
    properties : dict
        Dictionary of various fluid properties. Accessible directly as attributes (e.g., `fluid.p` for pressure).
    critical_point : FluidState
        Properties at the fluid's critical point.
    triple_point_liquid : FluidState
        Properties at the fluid's triple point in the liquid state.
    triple_point_vapor : FluidState
        Properties at the fluid's triple point in the vapor state.

    Methods
    -------
    set_state(input_type, prop_1, prop_2):
        Set the thermodynamic state of the fluid using specified property inputs.

    Examples
    --------
    Accessing properties:

        - fluid.T - Retrieves temperature directly as an attribute.
        - fluid.properties['p'] - Retrieves pressure through dictionary-like access.

    Accessing critical point properties:

        - fluid.critical_point.p - Retrieves critical pressure.
        - fluid.critical_point['T'] - Retrieves critical temperature.

    Accessing triple point properties:

        - fluid.triple_point_liquid.h - Retrieves liquid enthalpy at the triple point.
        - fluid.triple_point_vapor.s - Retrieves vapor entropy at the triple point.
    """

    def __init__(
        self,
        name,
        backend="HEOS",
        exceptions=True,
        initialize_critical=True,
        initialize_triple=True,
    ):
        self.name = name
        self.backend = backend
        self._AS = CP.AbstractState(backend, name)
        self.exceptions = exceptions
        self.converged_flag = False
        self.properties = {}

        # Initialize variables
        self.sat_liq = None
        self.sat_vap = None
        self.spinodal_liq = None
        self.spinodal_vap = None
        self.pseudo_critical_line = None
        self.Q_quality = None

        # Assign critical point properties
        if initialize_critical:
            self.critical_point = self._compute_critical_point()

        # Assign triple point properties
        if initialize_triple:
            self.triple_point_liquid = self._compute_triple_point_liquid()
            self.triple_point_vapor = self._compute_triple_point_vapor()

    def __getattr__(self, name):
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"'Fluid' object has no attribute '{name}'")

    def _compute_critical_point(self):
        """Calculate the properties at the critical point"""
        rho_crit, T_crit = self._AS.rhomass_critical(), self._AS.T_critical()
        self.set_state(DmassT_INPUTS, rho_crit, T_crit)
        return FluidState(self.properties)

    def _compute_triple_point_liquid(self):
        """Calculate the properties at the triple point (liquid state)"""
        self.set_state(QT_INPUTS, 0.00, self._AS.Ttriple())
        return FluidState(self.properties)

    def _compute_triple_point_vapor(self):
        """Calculate the properties at the triple point (vapor state)"""
        self.set_state(QT_INPUTS, 1.00, self._AS.Ttriple())
        return FluidState(self.properties)

    def set_state(self, input_type, prop_1, prop_2):
        """
        Set the thermodynamic state of the fluid based on input properties.

        This method updates the thermodynamic state of the fluid in the CoolProp ``abstractstate`` object
        using the given input properties. It then calculates either single-phase or two-phase
        properties based on the current phase of the fluid.

        If the calculation of properties fails, `converged_flag` is set to False, indicating an issue with
        the property calculation. Otherwise, it's set to True.

        Aliases of the properties are also added to the ``Fluid.properties`` dictionary for convenience.

        Parameters
        ----------
        input_type : str or int
            The variable pair used to define the thermodynamic state. This should be one of the
            predefined input pairs in CoolProp, such as ``PT_INPUTS`` for pressure and temperature.
            For all available input pairs, refer to :ref:`this list <module-input-pairs-table>`.
        prop_1 : float
            The first property value corresponding to the input type (e.g., pressure in Pa if the input
            type is CP.PT_INPUTS).
        prop_2 : float
            The second property value corresponding to the input type (e.g., temperature in K if the input
            type is CP.PT_INPUTS).

        Returns
        -------
        dict
            A dictionary of computed properties for the current state of the fluid. This includes both the
            raw properties from CoolProp and any additional alias properties.

        Raises
        ------
        Exception
            If `throw_exceptions` attribute is set to True and an error occurs during property calculation,
            the original exception is re-raised.


        """
        try:
            # Update Coolprop thermodynamic state
            self._AS.update(input_type, prop_1, prop_2)

            # Retrieve single-phase properties
            if self._AS.phase() != CP.iphase_twophase:
                self.properties = self.compute_properties_1phase()
            else:
                self.properties = self.compute_properties_2phase()

            # Add properties as aliases
            for key, value in PROPERTY_ALIAS.items():
                self.properties[key] = self.properties[value]

            # No errors computing the properies
            self.converged_flag = True

        # Something went wrong while computing the properties
        except Exception as e:
            self.converged_flag = False
            if self.exceptions:
                raise e

        return FluidState(self.properties)

    def compute_properties_1phase(self):
        """Get single-phase properties from CoolProp low level interface"""

        props = {}
        props["T"] = self._AS.T()
        props["p"] = self._AS.p()
        props["rhomass"] = self._AS.rhomass()
        props["umass"] = self._AS.umass()
        props["hmass"] = self._AS.hmass()
        props["smass"] = self._AS.smass()
        props["gibbsmass"] = self._AS.gibbsmass()
        props["cvmass"] = self._AS.cvmass()
        props["cpmass"] = self._AS.cpmass()
        props["gamma"] = props["cpmass"] / props["cvmass"]
        props["compressibility_factor"] = self._AS.compressibility_factor()
        props["speed_sound"] = self._AS.speed_sound()
        props["isentropic_bulk_modulus"] = props["rhomass"] * props["speed_sound"] ** 2
        props["isentropic_compressibility"] = 1 / props["isentropic_bulk_modulus"]
        props["isothermal_bulk_modulus"] = 1 / self._AS.isothermal_compressibility()
        props["isothermal_compressibility"] = self._AS.isothermal_compressibility()
        isobaric_expansion_coefficient = self._AS.isobaric_expansion_coefficient()
        props["isobaric_expansion_coefficient"] = isobaric_expansion_coefficient
        props["viscosity"] = self._AS.viscosity()
        props["conductivity"] = self._AS.conductivity()
        props["Q"] = np.nan
        props["quality_mass"] = np.nan
        props["quality_volume"] = np.nan

        return props

    def compute_properties_2phase(self):
        """Get two-phase properties from mixing rules and single-phase CoolProp properties"""

        # Basic properties of the two-phase mixture
        T_mix = self._AS.T()
        p_mix = self._AS.p()
        rho_mix = self._AS.rhomass()
        u_mix = self._AS.umass()
        h_mix = self._AS.hmass()
        s_mix = self._AS.smass()
        gibbs_mix = self._AS.gibbsmass()

        # Instantiate new fluid object to compute saturation properties without changing the state of the class
        temp = CP.AbstractState(self.backend, self.name)

        # Saturated liquid properties
        temp.update(CP.QT_INPUTS, 0.00, T_mix)
        rho_L = temp.rhomass()
        cp_L = temp.cpmass()
        cv_L = temp.cvmass()
        k_L = temp.conductivity()
        mu_L = temp.viscosity()
        speed_sound_L = temp.speed_sound()
        dsdp_L = temp.first_saturation_deriv(CP.iSmass, CP.iP)

        # Saturated vapor properties
        temp.update(CP.QT_INPUTS, 1.00, T_mix)
        rho_V = temp.rhomass()
        cp_V = temp.cpmass()
        cv_V = temp.cvmass()
        k_V = temp.conductivity()
        mu_V = temp.viscosity()
        speed_sound_V = temp.speed_sound()
        dsdp_V = temp.first_saturation_deriv(CP.iSmass, CP.iP)

        # Volume fractions of vapor and liquid
        vol_frac_V = (rho_mix - rho_L) / (rho_V - rho_L)
        vol_frac_L = 1.00 - vol_frac_V

        # Mass fractions of vapor and liquid
        mass_frac_V = (1 / rho_mix - 1 / rho_L) / (1 / rho_V - 1 / rho_L)
        mass_frac_L = 1.00 - mass_frac_V

        # Heat capacities of the two-phase mixture
        cp_mix = mass_frac_L * cp_L + mass_frac_V * cp_V
        cv_mix = mass_frac_L * cv_L + mass_frac_V * cv_V

        # Transport properties of the two-phase mixture
        k_mix = vol_frac_L * k_L + vol_frac_V * k_V
        mu_mix = vol_frac_L * mu_L + vol_frac_V * mu_V

        # Compressibility factor of the two-phase mixture
        M = self._AS.molar_mass()
        R = self._AS.gas_constant()
        Z_mix = p_mix / (rho_mix * (R / M) * T_mix)

        # Speed of sound of the two-phase mixture
        mechanical_equilibrium = vol_frac_L / (
            rho_L * speed_sound_L**2
        ) + vol_frac_V / (rho_V * speed_sound_V**2)
        thermal_equilibrium = T_mix * (
            vol_frac_L * rho_L / cp_L * dsdp_L**2
            + vol_frac_V * rho_V / cp_V * dsdp_V**2
        )
        compressibility_HEM = mechanical_equilibrium + thermal_equilibrium
        if mass_frac_V < 1e-6:  # Avoid discontinuity when Q_v=0
            a_HEM = speed_sound_L
        elif mass_frac_V > 1.0 - 1e-6:  # Avoid discontinuity when Q_v=1
            a_HEM = speed_sound_V
        else:
            a_HEM = (1 / rho_mix / compressibility_HEM) ** 0.5

        # Store properties in dictionary
        properties = {}
        properties["T"] = T_mix
        properties["p"] = p_mix
        properties["rhomass"] = rho_mix
        properties["umass"] = u_mix
        properties["hmass"] = h_mix
        properties["smass"] = s_mix
        properties["gibbsmass"] = gibbs_mix
        properties["cvmass"] = cv_mix
        properties["cpmass"] = cp_mix
        properties["gamma"] = properties["cpmass"] / properties["cvmass"]
        properties["compressibility_factor"] = Z_mix
        properties["speed_sound"] = a_HEM
        properties["isentropic_bulk_modulus"] = rho_mix * a_HEM**2
        properties["isentropic_compressibility"] = (rho_mix * a_HEM**2) ** -1
        properties["isothermal_bulk_modulus"] = np.nan
        properties["isothermal_compressibility"] = np.nan
        properties["isobaric_expansion_coefficient"] = np.nan
        properties["viscosity"] = mu_mix
        properties["conductivity"] = k_mix
        properties["Q"] = mass_frac_V
        properties["quality_mass"] = mass_frac_V
        properties["quality_volume"] = vol_frac_V

        return properties


def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored (can be None) and the second element
        is a dictionary representing the processed state of the system.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
    """
    # Initialize ode_out as a dictionary
    ode_out = {}
    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(t_i, y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    return ode_out


def get_geometry(length, total_length, area_in, area_ratio):
    """
    Calculates the cross-sectional area, area slope, perimeter, and diameter
    of a pipe or nozzle at a given length along its axis.

    This function is useful for analyzing variable area pipes or nozzles, where the
    area changes linearly from the inlet to the outlet. The area slope is calculated
    based on the total change in area over the total length, assuming a linear variation.

    Parameters
    ----------
    length : float
        The position along the pipe or nozzle from the inlet (m).
    total_length : float
        The total length of the pipe or nozzle (m).
    area_in : float
        The cross-sectional area at the inlet of the pipe or nozzle (m^2).
    area_ratio : float
        The ratio of the area at the outlet to the area at the inlet.

    Returns
    -------
    area : float
        The cross-sectional area at the specified length (m^2).
    area_slope : float
        The rate of change of the area with respect to the pipe or nozzle's length (m^2/m).
    perimeter : float
        The perimeter of the cross-section at the specified length (m).
    diameter : float
        The diameter of the cross-section at the specified length (m).
    """
    area_slope = (area_ratio - 1.0) * area_in / total_length
    area = area_in + area_slope * length
    radius = np.sqrt(area / np.pi)
    diameter = 2 * radius
    perimeter = np.pi * diameter
    return area, area_slope, perimeter, diameter


def get_wall_friction(velocity, density, viscosity, roughness, diameter):
    """
    Computes the frictional stress at the wall of a pipe due to viscous effects.

    The function first calculates the Reynolds number to characterize the flow.
    It then uses the Haaland equation to find the Darcy-Weisbach friction factor.
    Finally, it calculates the wall shear stress using the Darcy-Weisbach equation.

    Parameters
    ----------
    velocity : float
        The flow velocity of the fluid in the pipe (m/s).
    density : float
        The density of the fluid (kg/m^3).
    viscosity : float
        The dynamic viscosity of the fluid (Pa·s or N·s/m^2).
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    stress_wall : float
        The shear stress at the wall due to friction (Pa or N/m^2).
    friction_factor : float
        The Darcy-Weisbach friction factor, dimensionless.
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    """
    reynolds = velocity * density * diameter / viscosity
    friction_factor = get_friction_factor_haaland(reynolds, roughness, diameter)
    stress_wall = (1 / 8) * friction_factor * density * velocity**2
    return stress_wall, friction_factor, reynolds


def get_friction_factor_haaland(reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    f = (-1.8 * np.log10(6.9 / reynolds + (roughness / diameter / 3.7) ** 1.11)) ** -2
    return f


def get_heat_transfer_coefficient(
    velocity, density, heat_capacity, darcy_friction_factor
):
    """
    Estimates the heat transfer using the Reynolds analogy.

    This function is an adaptation of the Reynolds analogy which relates the heat transfer
    coefficient to the product of the Fanning friction factor, velocity, density, and heat
    capacity of the fluid.

    Parameters
    ----------
    velocity : float
        Velocity of the fluid (m/s).
    density : float
        Density of the fluid (kg/m^3).
    heat_capacity : float
        Specific heat capacity of the fluid at constant pressure (J/kg·K).
    darcy_friction_factor : float
        Darcy friction factor, dimensionless.

    Returns
    -------
    float
        Estimated heat transfer coefficient (W/m^2·K).

    Notes
    -----
    The Fanning friction factor used here is a quarter of the Darcy friction factor.
    """
    fanning_friction_factor = darcy_friction_factor / 4
    return 0.5 * fanning_friction_factor * velocity * density * heat_capacity


def pipeline_steady_state_1D(
    fluid_name,
    pressure_in,
    temperature_in,
    diameter_in,
    length,
    roughness,
    area_ratio=1.00,
    mass_flow=None,
    mach_in=None,
    include_friction=True,
    include_heat_transfer=True,
    temperature_external=None,
    number_of_points=None,
):
    """
    Simulates steady-state flow in a 1D pipeline system.

    This function integrates mass, momentum, and energy equations along the length
    of the pipeline. It models friction using the Darcy-Weisbach equation and the Haaland
    correlation for the friction factor. Heat transfer at the walls is calculated using
    an overall heat transfer coefficient based on the Reynolds analogy. Fluid properties
    are obtained using the CoolProp library.

    Parameters
    ----------
    fluid_name : str
        Name of the fluid as recognized by the CoolProp library.
    pressure_in : float
        Inlet pressure of the fluid (Pa).
    temperature_in : float
        Inlet temperature of the fluid (K).
    diameter_in : float
        Inner diameter of the pipeline (m).
    length : float
        Length of the pipeline (m).
    roughness : float
        Surface roughness of the pipeline (m).
    area_ratio : float, optional
        Ratio of the outlet area to the inlet area (default is 1.00).
    mass_flow : float, optional
        Mass flow rate of the fluid (kg/s). Either mass_flow or mach_in must be specified.
    mach_in : float, optional
        Inlet Mach number. Either mass_flow or mach_in must be specified.
    include_friction : bool, optional
        Whether to include friction in calculations (default is True).
    include_heat_transfer : bool, optional
        Whether to include heat transfer in calculations (default is True).
    temperature_external : float, optional
        External temperature for heat transfer calculations (K).
    number_of_points : int, optional
        Number of points for spatial discretization.

    Returns
    -------
    dict
        A dictionary containing the solution of the pipeline flow, with keys for distance,
        velocity, density, pressure, temperature, and other relevant flow properties.

    Raises
    ------
    ValueError
        If neither or both of mass_flow and mach_in are specified.
    """
    # Check that exactly one of mass_flow or mach_in is provided
    if (mass_flow is None and mach_in is None) or (
        mass_flow is not None and mach_in is not None
    ):
        raise ValueError(
            "Exactly one of 'mass_flow' or 'mach_in' must be specified, but not both."
        )

    # Define geometry
    radius_in = 0.5 * diameter_in
    area_in = np.pi * radius_in**2
    # perimeter_in = 2 * np.pi * radius_in

    # Create Fluid object
    fluid = Fluid(fluid_name, backend="HEOS", exceptions=True)

    # Calculate inlet density
    state_in = fluid.set_state(PT_INPUTS, pressure_in, temperature_in)
    density_in = state_in.rho

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * state_in.a

    # System of ODEs describing the flow equations
    def odefun(t, y):
        # Rename from ODE terminology to physical variables
        x = t
        v, rho, p = y

        # Calculate thermodynamic state
        state = fluid.set_state(DmassP_INPUTS, rho, p)

        # Calculate area
        area, area_slope, perimeter, diameter = get_geometry(
            length=x, total_length=length, area_in=area_in, area_ratio=area_ratio
        )

        # Compute friction at the walls
        stress_wall, friction_factor, reynolds = get_wall_friction(
            velocity=v,
            density=rho,
            viscosity=state.mu,
            roughness=roughness,
            diameter=diameter,
        )
        if not include_friction:
            stress_wall = 0.0
            friction_factor = 0.0

        # Calculate heat transfer
        if include_heat_transfer:
            U = get_heat_transfer_coefficient(v, rho, state.cp, friction_factor)
            heat_in = U * (temperature_external - fluid.T)
        else:
            U = 0.0
            heat_in = 0

        # Compute coefficient matrix
        M = np.asarray(
            [
                [rho, v, 0.0],
                [rho * v, 0.0, 1.0],
                [0.0, -state.a**2, 1.0],
            ]
        )

        # Compute right hand side
        G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
        b = np.asarray(
            [
                -rho * v / area * area_slope,
                -perimeter / area * stress_wall,
                +perimeter / area * G / v * (stress_wall * v + heat_in),
            ]
        )

        # Solve the linear system of equations
        dy = scipy.linalg.solve(M, b)

        # Save all relevant variables in dictionary
        out = {
            "distance": x,
            "velocity": v,
            "density": rho,
            "pressure": p,
            "temperature": state.T,
            "speed_of_sound": state.a,
            "viscosity": state.mu,
            "compressibility_factor": state.Z,
            "enthalpy": state.h,
            "entropy": state.s,
            "total_enthalpy": state.h + 0.5 * v**2,
            "mach_number": v / state.a,
            "mass_flow": v * rho * area,
            "area": area,
            "area_slope": area_slope,
            "perimeter": perimeter,
            "diameter": diameter,
            "stress_wall": stress_wall,
            "friction_factor": friction_factor,
            "reynolds": reynolds,
            "source_1": b[0],
            "source_2": b[1],
            "source_3": b[2],
        }

        return dy, out

    # Solve polytropic compression differential equation

    solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],  # Give only 'dy' to solver
        [0.0, length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, length, number_of_points) if number_of_points else None,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )

    solution = postprocess_ode(solution.t, solution.y, odefun)

    return solution


def get_choke_length_fanno(Ma, k, f, D):
    """
    Computes the dimensionless choke length for Fanno flow of a perfect gas.

    The dimensionless choke length is calculated using the formula:

        (fL*)/D = (1 - Ma^2)/(kMa^2) + (k+1)/(2k) * ln([(k + 1)Ma^2]/[2 + (k - 1)Ma^2])

    The formula is applicable for adiabatic flow with no heat transfer and friction in
    constant-area ducts.

    Parameters
    ----------
    Ma : float
        Mach number of the flow.
    k : float
        Specific heat ratio of the gas (cp/cv).
    f : float
        Darcy friction factor.
    D : float
        Diameter of the duct.

    Returns
    -------
    float
        The dimensionless choke length (fL*/D) for the given Fanno flow conditions.
    """
    term1 = (1 - Ma**2) / (k * Ma**2)
    term2 = (k + 1) / (2 * k)
    term3 = np.log(((k + 1) * Ma**2) / (2 + (k - 1) * Ma**2))

    dimensionless_length = term1 + term2 * term3

    return dimensionless_length * D / f


def get_critical_area_ratio_isentropic(Ma, k):
    """
    Calculates the critical area ratio for isentropic flow of a perfect gas.

    This function computes the ratio of the area of the flow passage (A) to the
    area at the throat (A*) where the flow is sonic (Mach number, Ma = 1), for a
    given Mach number and specific heat ratio (k) of a perfect gas.

    The formula used for the calculation is given by:

        A/A* = (1/Ma) * ((2/(k + 1)) * (1 + (k - 1)/2 * Ma^2))**((k + 1)/(2*(k - 1)))

    The formula is applicable for isentropic flow with no heat transfer and friction in
    variable-area ducts.

    Parameters
    ----------
    Ma : float
        Mach number of the flow at the area A where the ratio is being calculated.
    k : float
        Specific heat ratio (cp/cv) of the perfect gas.

    Returns
    -------
    float
        The critical area ratio (A/A*) for the specified Mach number and specific
        heat ratio.
    """
    term1 = 2 / (k + 1)
    term2 = 1 + (k - 1) / 2 * Ma**2
    exponent = (k + 1) / (2 * (k - 1))

    area_ratio = (1 / Ma) * (term1 * term2) ** exponent

    return area_ratio
