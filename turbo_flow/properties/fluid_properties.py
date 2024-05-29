import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import copy

from ..pysolver_view import (
    NonlinearSystemSolver,
    NonlinearSystemProblem,
    OptimizationProblem,
    OptimizationSolver,
)

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
PHASE_INDEX = sorted(PHASE_INDEX.items(), key=lambda x: x[1])
INPUT_PAIRS = sorted(INPUT_PAIRS.items(), key=lambda x: x[1])


def _generate_coolprop_input_table():
    """Create table of input pairs as string to be copy-pasted in Sphinx documentation"""

    inputs_table = ".. list-table:: CoolProp Input Mappings\n"
    inputs_table += "   :widths: 50 30\n"
    inputs_table += "   :header-rows: 1\n\n"
    inputs_table += "   * - Input pair name\n"
    inputs_table += "     - Input pair mapping\n"

    for name, value in INPUT_PAIRS:
        inputs_table += f"   * - {name}\n"
        inputs_table += f"     - {value}\n"

    return inputs_table


def states_to_dict(states):
    """
    Convert a list of state objects into a dictionary.
    Each key is a field name of the state objects, and each value is a NumPy array of all the values for that field.
    """
    state_dict = {}
    for field in states[0].keys():
        state_dict[field] = np.array([getattr(state, field) for state in states])
    return state_dict


def states_to_dict_2d(states_grid):
    """
    Convert a 2D list (grid) of state objects into a dictionary.
    Each key is a field name of the state objects, and each value is a 2D NumPy array of all the values for that field.

    Parameters
    ----------
    states_grid : list of list of objects
        A 2D grid where each element is a state object with the same keys.

    Returns
    -------
    dict
        A dictionary where keys are field names and values are 2D arrays of field values.
    """
    state_dict_2d = {}
    for i, row in enumerate(states_grid):
        for j, state in enumerate(row):
            for field in state.keys():
                if field not in state_dict_2d:
                    state_dict_2d[field] = np.empty(
                        (len(states_grid), len(row)), dtype=object
                    )
                state_dict_2d[field][i, j] = getattr(state, field)

    return state_dict_2d


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

    def __init__(self, fluid):
        # Use an internal dictionary to store properties
        self._properties = fluid.properties
        self._properties["fluid_name"] = fluid.name
        self._properties["converged"] = fluid.converged_flag
        self._properties["identifier"] = fluid.identifier

    def __getattr__(self, name):
        # This method is called when an attribute is accessed
        try:
            return self._properties[name]
        except KeyError:
            raise AttributeError(f"Attribute '{name}' not found in FluidState.")

    def __setattr__(self, name, value):
        # This method is called when an attribute is set
        if name == "_properties":
            # Initialize the _properties dictionary
            super().__setattr__(name, value)
        else:
            self._properties[name] = value

    def __getitem__(self, key):
        # This method is called when using dictionary-like access
        try:
            return self._properties[key]
        except KeyError:
            raise KeyError(
                f"Key '{key}' not found in FluidState. Available keys: {', '.join(self._properties.keys())}"
            )

    def __setitem__(self, key, value):
        # This method is called when using dictionary-like assignment
        self._properties[key] = value

    def __str__(self):
        properties_str = "\n   ".join(
            [f"{key}: {value}" for key, value in self._properties.items()]
        )
        return f"FluidState:\n   {properties_str}"

    def to_dict(self):
        return self._properties.copy()

    def keys(self):
        return self._properties.keys()

    def items(self):
        return self._properties.items()

    def values(self):
        return self._properties.values()


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
        identifier=None,
    ):
        self.name = name
        self.backend = backend
        self._AS = CP.AbstractState(backend, name)
        self.exceptions = exceptions
        self.converged_flag = False
        self.identifier = identifier
        self.properties = {}

        # Initialize variables
        self.sat_liq = None
        self.sat_vap = None
        self.spinodal_liq = None
        self.spinodal_vap = None
        self.pseudo_critical_line = None
        self.q_mesh = None
        self.graphic_elements = {}

        # Get critical and triple point properties
        if self._AS.fluid_param_string("pure") == "true":
            self.critical_point = self._compute_critical_point()
            self.triple_point_liquid = self._compute_triple_point_liquid()
            self.triple_point_vapor = self._compute_triple_point_vapor()

        # Pressure and temperature limits
        self.p_min = 1
        self.p_max = self._AS.pmax()
        self.T_min = self._AS.Tmin()
        self.T_max = self._AS.Tmax()

    def __getattr__(self, name):
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"'Fluid' object has no attribute '{name}'")

    def _compute_critical_point(self):
        """Calculate the properties at the critical point"""
        rho_crit, T_crit = self._AS.rhomass_critical(), self._AS.T_critical()
        self.set_state(DmassT_INPUTS, rho_crit, T_crit, generalize_quality=False)
        return FluidState(self)

    def _compute_triple_point_liquid(self):
        """Calculate the properties at the triple point (liquid state)"""
        self.set_state(QT_INPUTS, 0.00, self._AS.Ttriple(), generalize_quality=False)
        return FluidState(self)

    def _compute_triple_point_vapor(self):
        """Calculate the properties at the triple point (vapor state)"""
        self.set_state(QT_INPUTS, 1.00, self._AS.Ttriple(), generalize_quality=False)
        return FluidState(self)

    def get_props(self, input_type, prop_1, prop_2, generalize_quality=True):
        return self.set_state(
            input_type, prop_1, prop_2, generalize_quality=generalize_quality
        )

    def set_state(self, input_type, prop_1, prop_2, generalize_quality=True):
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

            if self._AS.fluid_param_string("pure") == "false":
                generalize_quality = False

            # Retrieve single-phase properties
            if self._AS.phase() != CP.iphase_twophase:
                self.properties = self.compute_properties_1phase(
                    generalize_quality=generalize_quality
                )
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

        # TODO Return a new state that is not mutated?
        return FluidState(self)

    def compute_properties_1phase(self, generalize_quality=True):
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

        if generalize_quality:
            # Instantiate new fluid object to compute saturation properties without changing the state of the class
            temp = CP.AbstractState(self.backend, self.name)
            # Extend quality calculation beyond the two-phase region
            if props["p"] < self.critical_point.p:
                # Set the saturation state of the fluid at the given pressure
                temp.update(PQ_INPUTS, props["p"], 0.00)
                h_liq = temp.hmass()
                temp.update(PQ_INPUTS, props["p"], 1.00)
                h_vap = temp.hmass()
                # print(props)
                quality = (props["hmass"] - h_liq) / (h_vap - h_liq)
            else:
                # For states at or above the critical pressure, the concept of saturation states is not applicable
                # Instead, use a 'pseudo-critical' state for comparison, where the density is set to the critical density
                # but the pressure is the same as the state of interest
                # Use a band of a certain width to prevent a discontinuity
                temp.update(DmassP_INPUTS, self.critical_point.rho, props["p"])
                # print(props)
                quality = (props["hmass"] - 0.95 * temp.hmass()) / (
                    1.05 * temp.hmass() - 0.95 * temp.hmass()
                )

        else:
            quality = np.nan

        props["Q"] = quality
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

    def set_state_metastable(
        self, prop_1, prop_1_value, prop_2, prop_2_value, rho_guess, T_guess
    ):
        # problem = PropertyRoot()

        return

    def set_state_metastable_rhoT(self, rho, T):
        # TODO: Add check to see if we are inside thespinodal and return two phase properties if yes
        # TODO: implement root finding functionality to accept p-h, T-s, p-s arguments [good initial guess required]
        # TODO: can it be generalized so that it uses equilibrium as initial guess with any inputs? (even if they are T-d)
        try:
            # Update Coolprop thermodynamic state
            self.properties = self.compute_properties_metastable_rhoT(rho, T, self._AS)

            # Add properties as aliases
            for key, value in self.aliases.items():
                self.properties[key] = self.properties[value]

            # No errors computing the properies
            self.converged_flag = True

        # Something went wrong while computing the properties
        except Exception as e:
            self.converged_flag = False
            if self.exceptions:
                raise e

        return self.properties

    @staticmethod
    def compute_properties_metastable_rhoT(rho, T, AS):
        """
        Compute the thermodynamic properties of a fluid using the Helmholtz
        energy equation of state. All properties thermodynamic properties can
        be derived as combinations of the Helmholtz energy and its
        derivatives with respect to density and pressure.

        This function can be used to estimate metastable properties using the
        equation of state beyond the saturation lines.
        """

        # Update thermodynamic state
        AS.update(CP.DmassT_INPUTS, rho, T)

        # Get fluid constant properties
        R = AS.gas_constant()
        M = AS.molar_mass()
        T_crit = AS.T_critical()
        rho_crit = AS.rhomass_critical()

        # Compute reduced variables
        tau = T_crit / T
        delta = rho / rho_crit

        # Compute from the Helmholtz energy derivatives
        alpha = AS.alpha0() + AS.alphar()
        dalpha_dTau = AS.dalpha0_dTau() + AS.dalphar_dTau()
        dalpha_dDelta = AS.dalpha0_dDelta() + AS.dalphar_dDelta()
        d2alpha_dTau2 = AS.d2alpha0_dTau2() + AS.d2alphar_dTau2()
        d2alpha_dDelta2 = AS.d2alpha0_dDelta2() + AS.d2alphar_dDelta2()
        d2alpha_dDelta_dTau = AS.d2alpha0_dDelta_dTau() + AS.d2alphar_dDelta_dTau()

        # Compute thermodynamic properties from Helmholtz energy EOS
        properties = {}
        properties["T"] = T
        properties["p"] = (R / M) * T * rho * delta * dalpha_dDelta
        properties["rhomass"] = rho
        properties["umass"] = (R / M) * T * (tau * dalpha_dTau)
        properties["hmass"] = (R / M) * T * (tau * dalpha_dTau + delta * dalpha_dDelta)
        properties["smass"] = (R / M) * (tau * dalpha_dTau - alpha)
        properties["gibbsmass"] = (R / M) * T * (alpha + delta * dalpha_dDelta)
        properties["cvmass"] = (R / M) * (-(tau**2) * d2alpha_dTau2)
        properties["cpmass"] = (R / M) * (
            -(tau**2) * d2alpha_dTau2
            + (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau) ** 2
            / (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
        )
        properties["gamma"] = properties["cpmass"] / properties["cvmass"]
        properties["compressibility_factor"] = delta * dalpha_dDelta
        properties["speed_sound"] = (
            (R / M * T)
            * (
                (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
                - (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau) ** 2
                / (tau**2 * d2alpha_dTau2)
            )
        ) ** 0.5
        properties["isentropic_bulk_modulus"] = (rho * R / M * T) * (
            (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
            - (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau) ** 2
            / (tau**2 * d2alpha_dTau2)
        )
        properties["isentropic_compressibility"] = (
            1 / properties["isentropic_bulk_modulus"]
        )
        properties["isothermal_bulk_modulus"] = (
            R / M * T * rho * (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
        )
        properties["isothermal_compressibility"] = 1 / (
            R / M * T * rho * (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
        )
        properties["isobaric_expansion_coefficient"] = (
            (1 / T)
            * (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau)
            / (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
        )
        properties["viscosity"] = AS.viscosity()
        properties["conductivity"] = AS.conductivity()
        properties["Q"] = np.nan
        properties["quality_mass"] = np.nan
        properties["quality_volume"] = np.nan

        return properties

    def get_property(self, propname):
        """Get the value of a single property"""
        if propname in self.properties:
            return self.properties[propname]
        else:
            valid_options = "\n\t".join(self.properties.keys())
            raise ValueError(
                f"The requested property '{propname}' is not available. The valid options are:\n\t{valid_options}"
            )

    def compute_properties_meanline(self, input_type, prop_1, prop_2):
        """Extract fluid properties for meanline model"""

        # Compute properties in the normal way
        self.set_state(input_type, prop_1, prop_2)

        # Store a subset of the properties in a dictionary
        fluid_properties = {}
        property_subset = [
            "p",
            "T",
            "h",
            "s",
            "d",
            "Z",
            "a",
            "mu",
            "k",
            "cp",
            "cv",
            "gamma",
        ]
        for item in property_subset:
            fluid_properties[item] = self.properties[item]

        return fluid_properties

    def compute_sonic_state(self, input_type, prop_1, prop_2):
        props = {}

        return FluidState(props)

    # ------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------ #

    def _get_label(self, label, show_in_legend):
        """Returns the appropriate label value based on whether it should be shown in the legend."""
        return label if show_in_legend else "_no_legend_"

    def _plot_or_update_line(self, axes, x_data, y_data, line_name, **plot_params):
        # Ensure there is a dictionary for this axes
        if axes not in self.graphic_elements:
            self.graphic_elements[axes] = {}

        # Check if the line exists for this axes
        if line_name in self.graphic_elements[axes]:
            line = self.graphic_elements[axes][line_name]
            line.set_data(np.atleast_1d(x_data), np.atleast_1d(y_data))
            # Update line properties
            for param, value in plot_params.items():
                setattr(line, param, value)
            line.set_visible(True)
        else:
            # Create a new line with the provided plot parameters
            (line,) = axes.plot(x_data, y_data, **plot_params)
            self.graphic_elements[axes][line_name] = line
        return line

    def _plot_or_update_contours(
        self, axes, x_data, y_data, z_data, contour_levels, line_name, **contour_params
    ):
        # Ensure there is a dictionary for this axes
        if axes not in self.graphic_elements:
            self.graphic_elements[axes] = {}

        # Check if the contour exists for this axes
        if line_name in self.graphic_elements[axes]:
            for coll in self.graphic_elements[axes][line_name].collections:
                coll.remove()  # Remove the old contour collections

        # Create a new contour
        contour = axes.contour(x_data, y_data, z_data, contour_levels, **contour_params)
        self.graphic_elements[axes][line_name] = contour
        return contour

    def _set_visibility(self, axes, line_name, visible):
        if axes in self.graphic_elements and line_name in self.graphic_elements[axes]:
            self.graphic_elements[axes][line_name].set_visible(visible)

    def plot_phase_diagram(
        self,
        x_variable="s",
        y_variable="T",
        axes=None,
        num_points=200,
        plot_saturation_line=True,
        plot_critical_point=True,
        plot_triple_point_liquid=False,
        plot_triple_point_vapor=False,
        plot_spinodal_line=False,
        spinodal_line_method="standard",
        spinodal_line_color=0.5 * np.array([1, 1, 1]),
        spinodal_line_width=0.75,
        plot_quality_isolines=False,
        plot_pseudocritical_line=False,
        quality_levels=np.linspace(0.1, 1.0, 10),
        quality_labels=False,
        show_in_legend=False,
        **kwargs,
    ):
        if axes is None:
            axes = plt.gca()

        # Saturation line
        if plot_saturation_line:
            if self.sat_liq is None or self.sat_vap is None:
                self.sat_liq, self.sat_vap = compute_saturation_line(self, num_points)
            x = self.sat_liq[x_variable] + self.sat_vap[x_variable]
            y = self.sat_liq[y_variable] + self.sat_vap[y_variable]
            label = self._get_label("Saturation line", show_in_legend)
            params = {"label": label, "color": "black"}
            self._graphic_saturation_line = self._plot_or_update_line(
                axes, x, y, "saturation_line", **params
            )
        else:
            self._set_visibility(axes, "saturation_line", False)

        # Plot pseudocritical line
        if plot_pseudocritical_line:
            if self.pseudo_critical_line is None:
                self.pseudo_critical_line = compute_pseudocritical_line(self)
            x = self.pseudo_critical_line[x_variable]
            y = self.pseudo_critical_line[y_variable]
            label = self._get_label("Pseudocritical line", show_in_legend)
            params = {
                "label": label,
                "color": "black",
                "linestyle": "--",
                "linewidth": 0.75,
            }
            self._graphic_pseudocritical_line = self._plot_or_update_line(
                axes, x, y, "pseudocritical_line", **params
            )
        else:
            self._set_visibility(axes, "pseudocritical_line", False)

        # Plot quality isolines
        if plot_quality_isolines:
            if self.q_mesh is None:
                self.q_mesh = compute_quality_grid(self, num_points, quality_levels)
            x = self.q_mesh[x_variable]
            y = self.q_mesh[y_variable]
            _, m = np.shape(x)
            z = np.tile(quality_levels, (m, 1)).T
            params = {"colors": "black", "linestyles": ":", "linewidths": 0.75}
            self._graphics_q_lines = self._plot_or_update_contours(
                axes, x, y, z, quality_levels, "quality_isolines", **params
            )

            if quality_labels:
                axes.clabel(self._graphics_q_lines, fontsize=9, rightside_up=True)

        else:
            # Remove existing contour lines if they exist
            if "quality_isolines" in self.graphic_elements.get(axes, {}):
                for coll in self.graphic_elements[axes]["quality_isolines"].collections:
                    coll.remove()
                del self.graphic_elements[axes]["quality_isolines"]

        # Plot critical point
        params = {
            "color": "black",
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "w",
        }
        if plot_critical_point:
            x = self.critical_point[x_variable]
            y = self.critical_point[y_variable]
            label = self._get_label("Critical point", show_in_legend)
            self._graphic_critical_point = self._plot_or_update_line(
                axes, x, y, "critical_point", label=label, **params
            )
        else:
            self._set_visibility(axes, "critical_point", False)

        # Plot liquid triple point
        if plot_triple_point_liquid:
            x = self.triple_point_liquid[x_variable]
            y = self.triple_point_liquid[y_variable]
            label = self._get_label("Triple point liquid", show_in_legend)
            self._graphic_triple_point_liquid = self._plot_or_update_line(
                axes, x, y, "triple_point_liquid", label=label, **params
            )
        else:
            self._set_visibility(axes, "triple_point_liquid", False)

        # Plot vapor triple point
        if plot_triple_point_vapor:
            x = self.triple_point_vapor[x_variable]
            y = self.triple_point_vapor[y_variable]
            label = self._get_label("Triple point vapor", show_in_legend)
            self._graphic_triple_point_vapor = self._plot_or_update_line(
                axes, x, y, "triple_point_vapor", label=label, **params
            )
        else:
            self._set_visibility(axes, "triple_point_vapor", False)

        return axes


def compute_saturation_line(fluid, N_points=100):
    # Initialize objects to store properties
    prop_names = fluid.properties.keys()
    liquid_line = {name: [] for name in prop_names}
    vapor_line = {name: [] for name in prop_names}

    # Define temperature array with refinement close to the critical point
    ratio = 1 - fluid.triple_point_liquid.T / fluid.critical_point.T
    t1 = np.logspace(
        np.log10(1 - 0.9999), np.log10(ratio / 10), int(np.ceil(N_points / 2))
    )
    t2 = np.logspace(np.log10(ratio / 10), np.log10(ratio), int(np.floor(N_points / 2)))
    T_sat = (1 - np.concatenate([t1, t2])) * fluid.critical_point.T

    # Loop over temperatures and property names in an efficient way
    for T in T_sat:
        # Compute liquid saturation line
        for name in prop_names:
            fluid.set_state(CP.QT_INPUTS, 0.00, T)
            liquid_line[name].append(fluid.properties[name])

        # Compute vapor saturation line
        for name in prop_names:
            fluid.set_state(CP.QT_INPUTS, 1.00, T)
            vapor_line[name].append(fluid.properties[name])

    # Add critical point as part of the spinodal line
    for name in prop_names:
        liquid_line[name] = [fluid.critical_point[name]] + liquid_line[name]
        vapor_line[name] = [fluid.critical_point[name]] + vapor_line[name]

    # Re-format for easy concatenation
    for name in prop_names:
        liquid_line[name] = list(reversed(liquid_line[name]))

    return liquid_line, vapor_line


def compute_spinodal_line(fluid, N_points=100, method="standard"):
    raise NotImplementedError(
        "The 'compute_spinodal_line' function has not been implemented yet."
    )


def compute_pseudocritical_line(fluid, N_points=100):
    # Initialize objects to store properties
    prop_names = fluid.properties.keys()
    pseudocritical_line = {name: [] for name in prop_names}

    # Define temperature array with refinement close to the critical point
    tau = np.logspace(np.log10(1e-3), np.log10(1), N_points)
    T_range = (1 + tau) * fluid.critical_point.T

    # Loop over temperatures and compute pseudocritical properties
    for T in T_range:
        for name in prop_names:
            fluid.set_state(DmassT_INPUTS, fluid.critical_point.d, T)
            pseudocritical_line[name].append(fluid.properties[name])

    return pseudocritical_line


def compute_quality_grid(fluid, num_points, quality_levels):
    # Define temperature levels
    t1 = np.logspace(np.log10(1 - 0.9999), np.log10(0.1), int(num_points / 2))
    t2 = np.logspace(
        np.log10(0.1),
        np.log10(1 - (fluid.triple_point_liquid.T) / fluid.critical_point.T),
        int(num_points / 2),
    )
    temperature_levels = (1 - np.hstack((t1, t2))) * fluid.critical_point.T

    # Calculate property grid
    quality_grid = []
    for q in quality_levels:
        row = []
        for T in temperature_levels:
            row.append(fluid.set_state(CP.QT_INPUTS, q, T))
        quality_grid.append(row)

    return states_to_dict_2d(quality_grid)


def compute_properties_meshgrid(fluid, input_pair, range_1, range_2):
    """
    Compute fluid properties over a specified range and store them in a dictionary.

    This function creates a meshgrid of property values based on the specified ranges and input pair,
    computes the properties of the fluid at each point on the grid, and stores the results in a
    dictionary where each key corresponds to a fluid property.

    Parameters
    ----------
    fluid : Fluid object
        An instance of the Fluid class.
    input_pair : tuple
        The input pair specifying the property type (e.g., PT_INPUTS for pressure-temperature).
    range1 : tuple
        The range linspace(min, max, n) for the first property of the input pair.
    range2 : tuple
        The range linspace(min, max, n) for the second property of the input pair.

    Returns
    -------
    properties_dict : dict
        A dictionary where keys are property names and values are 2D numpy arrays of computed properties.
    grid1, grid2 : numpy.ndarray
        The meshgrid arrays for the first and second properties.
    """

    # Create the meshgrid
    grid1, grid2 = np.meshgrid(range_1, range_2)

    # Initialize dictionary to store properties and pre-allocate storage
    properties_dict = {key: np.zeros_like(grid1) for key in fluid.properties}

    # Compute properties at each point
    for i in range(len(range_2)):
        for j in range(len(range_1)):
            # Set state of the fluid
            fluid.set_state(input_pair, grid1[i, j], grid2[i, j])

            # Store the properties
            for key in fluid.properties:
                properties_dict[key][i, j] = fluid.properties[key]

    return properties_dict


# Implement class to calculate intersection with saturation line?


class PropertyRoot(NonlinearSystemProblem):
    """
    Find the root for thermodynamic state by iterating on the density-temperature
    native inputs to the helmholtz energy equations of state.

    Attributes
    ----------
    prop_1 : str
        The first property to be compared.
    prop_1_value : float
        The value of the first property.
    prop_2 : str
        The second property to be compared.
    prop_2_value : float
        The value of the second property.
    fluid : object
        An instance of the fluid class which has the helmholtz energy equations of state.

    Methods
    -------
    get_values(x)
        Calculates the residuals based on the given input values of density and temperature.
    """

    def __init__(self, prop_1, prop_1_value, prop_2, prop_2_value, fluid):
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.prop_1_value = prop_1_value
        self.prop_2_value = prop_2_value
        self.fluid = fluid

    def get_values(self, x):
        """
        Compute the residuals for the given density and temperature.

        Parameters
        ----------
        x : list
            List containing the values for density and temperature.

        Returns
        -------
        np.ndarray
            Array containing residuals (difference) for the two properties.
        """
        # Ensure x can be indexed and contains exactly two elements
        if not hasattr(x, "__getitem__") or len(x) != 2:
            raise ValueError(
                "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
            )

        rho, T = x
        props = self.fluid.set_state_metastable_rhoT(rho, T)

        residual = np.asarray(
            [
                self.prop_1_value - props[self.prop_1],
                self.prop_2_value - props[self.prop_2],
            ]
        )

        return residual


class SonicStateProblem(NonlinearSystemProblem):
    """ """

    def __init__(self, fluid, property_pair, prop_1, prop_2):
        # Calculate the thermodynamic state
        self.fluid = fluid
        self.state = fluid.set_state(property_pair, prop_1, prop_2)

        # Initial guess based in input sstate
        self.initial_guess = [self.state.d, self.state.T]

        # # Initial guess based on perfect gass relations
        # gamma = self.state.gamma
        # d_star = (2/(gamma + 1)) ** (1/(gamma-1)) * self.state.rho
        # T_star =  (2/(gamma + 1)) * self.state.T
        # self.initial_guess = [d_star, T_star]

    def get_values(self, x):
        # Ensure x can be indexed and contains exactly two elements
        if not hasattr(x, "__getitem__") or len(x) != 2:
            raise ValueError(
                "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
            )

        # Calculate state for the current density-temperature pair
        crit_state = self.fluid.set_state(DmassT_INPUTS, x[0], x[1])

        # Calculate the sonic state residual
        residual = np.asarray(
            [
                1.0 - (crit_state.h + 0.5 * crit_state.a**2) / self.state.h,
                1.0 - crit_state.s / self.state.s,
            ]
        )

        return residual


class SonicStateProblem2(OptimizationProblem):
    """ """

    def __init__(self, fluid, property_pair, prop_1, prop_2):
        # Calculate the thermodynamic state
        self.fluid = fluid
        self.state = fluid.set_state(property_pair, prop_1, prop_2)

        # Initial guess based in input sstate
        self.initial_guess = [self.state.d, self.state.T * 0.9]

        # # Initial guess based on perfect gass relations
        # gamma = self.state.gamma
        # d_star = (2/(gamma + 1)) ** (1/(gamma-1)) * self.state.rho
        # T_star =  (2/(gamma + 1)) * self.state.T
        # self.initial_guess = [d_star, T_star]

    def get_values(self, x):
        """
        Compute the residuals for the given density and temperature.

        Parameters
        ----------
        x : list
            List containing the values for density and temperature.

        Returns
        -------
        np.ndarray
            Array containing residuals (difference) for the two properties.
        """

        # Ensure x can be indexed and contains exactly two elements
        if not hasattr(x, "__getitem__") or len(x) != 2:
            raise ValueError(
                "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
            )

        # Calculate state for the current density-temperature pair
        crit_state = self.fluid.set_state(DmassT_INPUTS, x[0], x[1])

        # Calculate the sonic state residual
        residual = [
            1.0 - crit_state.s / self.state.s,
        ]

        # Objective function
        self.f = crit_state.d**2 * (self.state.h - crit_state.h)
        self.f = -self.f / (self.state.d * self.state.a) ** 2

        # Equality constraints
        self.c_eq = residual

        # No inequality constraints given for this problem
        self.c_ineq = []

        # Combine objective function and constraints
        objective_and_constraints = self.merge_objective_and_constraints(
            self.f, self.c_eq, self.c_ineq
        )

        return objective_and_constraints

    def get_bounds(self):
        bound_density = (
            self.fluid.triple_point_vapor.d * 1.5,
            self.fluid.critical_point.d * 3,
        )
        bound_temperature = (
            self.fluid.triple_point_vapor.T * 1,
            self.fluid.critical_point.T * 3,
        )
        # return [bound_density, bound_temperature]
        return None

    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)

    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)


def calculate_superheating(state, fluid):
    """
    Calculates the degree of superheating for a given state and adds this information to the state.

    Parameters
    ----------
    state : dict
        A dictionary representing the thermodynamic state, containing at least pressure (p), temperature (T),
        and enthalpy (h) of the fluid.
    fluid : object
        An object representing the fluid with its properties, including methods to set state and critical point data.

    Returns
    -------
    dict
        The input state dictionary with an added field 'superheating' representing the degree of superheating.
    """

    # Check if the pressure is below the critical pressure of the fluid
    if state["p"] < fluid.critical_point.p:
        # Set the saturation state of the fluid at the given pressure
        sat_state = fluid.set_state(PQ_INPUTS, state["p"], 1.00)

        # Check if the fluid is in the two-phase region
        if fluid._AS.phase() == CP.iphase_twophase:
            # In the two-phase region, define superheating as the normalized difference in enthalpy
            # The normalization is done using the specific heat capacity at saturation (cp)
            # This provides a continuous measure of superheating, even in the two-phase region
            state["superheating"] = (state["h"] - sat_state.h) / sat_state.cp
        else:
            # Outside the two-phase region, superheating is the difference in temperature
            # from the saturation temperature at the same pressure
            state["superheating"] = state["T"] - sat_state.T
    else:
        # For states at or above the critical pressure, the concept of saturation temperature is not applicable
        # Instead, use a 'pseudo-critical' state for comparison, where the density is set to the critical density
        # but the pressure is the same as the state of interest
        pseudo_crit = fluid.set_state(
            DmassP_INPUTS, fluid.critical_point.rho, state["p"]
        )

        # Define superheating as the difference in enthalpy from this 'pseudo-critical' state
        # This approach extends the definition of superheating to conditions above the critical pressure
        state["superheating"] = state.T - pseudo_crit.T

    return state


def calculate_subcooling(state, fluid):
    """
    Calculates the degree of subcooling for a given state and adds this information to the state.

    Parameters
    ----------
    state : dict
        A dictionary representing the thermodynamic state, containing at least pressure (p), temperature (T),
        and enthalpy (h) of the fluid.
    fluid : object
        An object representing the fluid with its properties, including methods to set state and critical point data.

    Returns
    -------
    dict
        The input state dictionary with an added field 'subcooling' representing the degree of subcooling.
    """

    # Check if the pressure is below the critical pressure of the fluid
    if state["p"] < fluid.critical_point.p:
        # Set the saturation state of the fluid at the given pressure
        sat_state = fluid.set_state(PQ_INPUTS, state["p"], 0.00)

        # Check if the fluid is in the two-phase region
        if fluid._AS.phase() == CP.iphase_twophase:
            # In the two-phase region, define subcooling as the normalized difference in enthalpy
            # The normalization is done using the specific heat capacity at saturation (cp)
            # This provides a continuous measure of subcooling, even in the two-phase region
            state["subcooling"] = (sat_state.h - state["h"]) / sat_state.cp
        else:
            # Outside the two-phase region, subcooling is the difference in temperature
            # from the saturation temperature at the same pressure
            state["subcooling"] = sat_state.T - state["T"]
    else:
        # For states at or above the critical pressure, the concept of saturation temperature is not applicable
        # Instead, use a 'pseudo-critical' state for comparison, where the density is set to the critical density
        # but the pressure is the same as the state of interest
        pseudo_crit = fluid.set_state(
            DmassP_INPUTS, fluid.critical_point.rho, state["p"]
        )

        # Define subcooling as the difference in enthalpy from this 'pseudo-critical' state
        # This approach extends the definition of subcooling to conditions above the critical pressure
        state["subcooling"] = pseudo_crit.T - state.T

    return state


# def compute_properties_metastable_rhoT(rho, T, fluid):
#     """
#     Compute the thermodynamic properties of a fluid using the Helmholtz
#     energy equation of state. All properties thermodynamic properties can
#     be derived as combinations of the Helmholtz energy and its
#     derivatives with respect to density and pressure.

#     This function can be used to estimate metastable properties using the
#     equation of state beyond the saturation lines.
#     """

#     # Update thermodynamic state
#     fluid.update(CP.DmassT_INPUTS, rho, T)

#     # Get fluid constant properties
#     R = fluid.gas_constant()
#     M = fluid.molar_mass()
#     T_crit = fluid.T_critical()
#     rho_crit = fluid.rhomass_critical()

#     # Compute reduced variables
#     tau = T_crit / T
#     delta = rho / rho_crit

#     # Compute from the Helmholtz energy derivatives
#     alpha = fluid.alpha0() + fluid.alphar()
#     dalpha_dTau = fluid.dalpha0_dTau() + fluid.dalphar_dTau()
#     dalpha_dDelta = fluid.dalpha0_dDelta() + fluid.dalphar_dDelta()
#     d2alpha_dTau2 = fluid.d2alpha0_dTau2() + fluid.d2alphar_dTau2()
#     d2alpha_dDelta2 = fluid.d2alpha0_dDelta2() + fluid.d2alphar_dDelta2()
#     d2alpha_dDelta_dTau = fluid.d2alpha0_dDelta_dTau() + fluid.d2alphar_dDelta_dTau()

#     # Compute thermodynamic properties from Helmholtz energy EOS
#     properties = {}
#     properties['T'] = T
#     properties['p'] = (R/M)*T*rho*delta*dalpha_dDelta
#     properties['rhomass'] = rho
#     properties['umass'] = (R/M)*T*(tau*dalpha_dTau)
#     properties['hmass'] = (R/M)*T*(tau*dalpha_dTau+delta*dalpha_dDelta)
#     properties['smass'] = (R/M)*(tau*dalpha_dTau - alpha)
#     properties['gibbsmass'] = (R/M)*T*(alpha + delta*dalpha_dDelta)
#     properties['cvmass'] = (R/M)*(-tau**2*d2alpha_dTau2)
#     properties['cpmass'] = (R/M)*(-tau**2*d2alpha_dTau2 + (delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)**2/(2*delta*dalpha_dDelta + delta**2*d2alpha_dDelta2))
#     properties['gamma'] = properties['cpmass']/properties['cvmass']
#     properties['compressibility_factor'] = delta*dalpha_dDelta
#     properties['speed_sound'] = ((R/M)*T*(2*delta*dalpha_dDelta + delta**2*d2alpha_dDelta2 - (delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)**2/(tau**2*d2alpha_dTau2)))**0.5
#     properties['isentropic_bulk_modulus'] = rho*(R/M)*T*(2*delta*dalpha_dDelta + delta**2*d2alpha_dDelta2 - (delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)**2/(tau**2*d2alpha_dTau2))
#     properties['isentropic_compressibility'] = 1 / properties["isentropic_bulk_modulus"]
#     properties['isothermal_bulk_modulus'] = (R/M)*T*rho*(2*delta*dalpha_dDelta + delta**2*d2alpha_dDelta2)
#     properties['isothermal_compressibility'] = 1/((R/M)*T*rho*(2*delta*dalpha_dDelta + delta**2*d2alpha_dDelta2))
#     properties['isobaric_expansion_coefficient'] = 1/T*(delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)/(2*delta*dalpha_dDelta + delta**2*d2alpha_dDelta2)
#     properties['viscosity'] = fluid.viscosity()
#     properties['conductivity'] = fluid.conductivity()
#     properties['Q'] = np.nan
#     properties['quality_mass'] = np.nan
#     properties['quality_volume'] = np.nan

#     return properties


if __name__ == "__main__":
    fluid = Fluid("Water", backend="HEOS")

    # # Properties of liquid water
    # props_stable = fluid.set_state(CP.PT_INPUTS, 101325, 300)
    # print()
    # print("Properties of liquid water")
    # print(f"{'Property':35} {'value':6}")
    # for key, value in props_stable.items():
    #     print(f"{key:35} {value:.6e}")

    # # Properties of water/steam mixture
    # props = fluid.set_state(CP.QT_INPUTS, 0.5, 300)
    # print()
    # print("Properties of water/steam mixture")
    # print(f"{'Property':35} {'value':6}")
    # for key, value in props.items():
    #     print(f"{key:35} {value:.6e}")

    # # Get subset of properties for meanline code
    # props = fluid.compute_properties_meanline(CP.QT_INPUTS, 0.5, 300)
    # print()
    # print("Properties for the meanline code")
    # print(f"{'Property':15} {'value':6}")
    # for key, value in props.items():
    #     print(f"{key:15} {value:.6e}")

    # #
    # props = compute_properties_metastable_rhoT(10, 500, fluid.abstractstate)
    # print("Metastable properties of water")
    # print(f"{'Property':35} {'value':6}")
    # for key, value in props.items():
    #     print(f"{key:35} {value:.6e}")

    # Check that the metastable property calculations match in the single-phase region
    p, T = 101325, 300
    props_stable = fluid.set_state(CP.PT_INPUTS, p, T)
    print()
    print(f"Properties of water at p={p} Pa and T={T} K")
    print(f"{'Property':35} {'Equilibrium':>15} {'Metastable':>15} {'Deviation':>15}")
    props_metastable = fluid.set_state_metastable_rhoT(
        props_stable["rho"], props_stable["T"]
    )
    for key in props_stable.keys():
        value_stable = props_stable[key]
        value_metastable = props_metastable[key]
        print(
            f"{key:35} {value_stable:+15.6e} {value_metastable:+15.6e} {(value_stable - value_metastable)/value_stable:+15.6e}"
        )
