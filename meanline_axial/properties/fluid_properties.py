import numpy as np
import CoolProp.CoolProp as CP

class FluidCoolProp_2Phase:
    r"""

    Parameters
    ----------
    fluid_name : Name of the fluid

    Methods
    -------
    compute_critical_properties()
        retrieve properties at the critical point

    """
    
    def __init__(self, fluid_name, backend="HEOS", throw_exceptions=True):
        self.fluid_name = fluid_name
        self.backend = backend
        self.abstractstate = CP.AbstractState(backend, fluid_name)
        self.throw_exceptions = throw_exceptions
        self.converged_flag = False
        self.fluid_properties = {}
        self.compute_critical_properties()
        self.compute_triple_properties()

    def compute_critical_properties(self):
        self.T_critical = self.abstractstate.T_critical()
        self.p_critical = self.abstractstate.p_critical()
        self.rho_critical = self.abstractstate.rhomass_critical()

    def compute_triple_properties(self):
        self.T_triple = self.abstractstate.Ttriple()
        temp = CP.AbstractState(self.backend, self.fluid_name)
        temp.update(CP.QT_INPUTS, 0, self.T_triple)
        self.p_triple = temp.p            

    def compute_properties(self, input_type, prop_1, prop_2):

        try:

            # Update Coolprop thermodynamic state
            self.abstractstate.update(input_type, prop_1, prop_2)

            # Retrieve single-phase properties
            if self.abstractstate.phase() != CP.iphase_twophase:
                self.fluid_properties = self.compute_properties_1phase()
            else:
                self.fluid_properties = self.compute_properties_2phase()

            # Add property aliases for convenience
            aliases = {"P": "p",
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
                       "k": "conductivity"}
            
            for key, value in aliases.items():
                self.fluid_properties[key] = self.fluid_properties[value]

            # No errors computing the properies
            self.converged_flag = True

        # Something went wrong while computing the properties
        except Exception as e:
            self.converged_flag = False
            if self.throw_exceptions:
                raise e
            
        return self.fluid_properties
    

    def compute_properties_1phase(self):
        """Get single-phase properties from CoolProp""" 
        
        fluid_properties = {}
        fluid_properties['T'] = self.abstractstate.T()
        fluid_properties['p'] = self.abstractstate.p()
        fluid_properties['rhomass'] = self.abstractstate.rhomass()
        fluid_properties['umass'] = self.abstractstate.umass()
        fluid_properties['hmass'] = self.abstractstate.hmass()
        fluid_properties['smass'] = self.abstractstate.smass()
        fluid_properties['cvmass'] = self.abstractstate.cvmass()
        fluid_properties['cpmass'] = self.abstractstate.cpmass()
        fluid_properties['gamma'] = fluid_properties['cpmass']/fluid_properties['cvmass']
        fluid_properties['compressibility_factor'] = self.abstractstate.compressibility_factor()
        fluid_properties['speed_sound'] = self.abstractstate.speed_sound()
        fluid_properties['isentropic_bulk_modulus'] = self.abstractstate.rhomass() * self.abstractstate.speed_sound() ** 2
        fluid_properties['isentropic_compressibility'] = 1 / fluid_properties["isentropic_bulk_modulus"]
        fluid_properties['isothermal_bulk_modulus'] = 1 / self.abstractstate.isothermal_compressibility()
        fluid_properties['isothermal_compressibility'] = self.abstractstate.isothermal_compressibility()
        fluid_properties['isobaric_expansion_coefficient'] = self.abstractstate.isobaric_expansion_coefficient()
        fluid_properties['viscosity'] = self.abstractstate.viscosity()
        fluid_properties['conductivity'] = self.abstractstate.conductivity()
        fluid_properties['Q'] = np.nan
        fluid_properties['quality_mass'] = np.nan
        fluid_properties['quality_volume'] = np.nan

        return fluid_properties


    def compute_properties_2phase(self):
        """Get two-phase properties from mixing rules""" 
        
        # Basic properties of the two-phase mixture
        T_mix = self.abstractstate.T()
        p_mix = self.abstractstate.p()
        rho_mix = self.abstractstate.rhomass()
        u_mix = self.abstractstate.umass()
        h_mix = self.abstractstate.hmass()
        s_mix = self.abstractstate.smass()

        # Instantiate new fluid object to compute saturation properties without changing the state of the class
        temp = CP.AbstractState(self.backend, self.fluid_name)

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
        mass_frac_V = (1/rho_mix - 1/rho_L) / (1/rho_V - 1/rho_L)
        mass_frac_L = 1.00 - mass_frac_V

        # Heat capacities of the two-phase mixture
        cp_mix = mass_frac_L*cp_L + mass_frac_V*cp_V
        cv_mix = mass_frac_L*cv_L + mass_frac_V*cv_V

        # Transport properties of the two-phase mixture
        k_mix = vol_frac_L*k_L + vol_frac_V*k_V
        mu_mix = vol_frac_L*mu_L + vol_frac_V*mu_V

        # Compressibility factor of the two-phase mixture
        M = self.abstractstate.molar_mass()
        R = self.abstractstate.gas_constant()
        Z_mix = p_mix / (rho_mix * (R/M) * T_mix)

        # Speed of sound of the two-phase mixture
        mechanical_equilibrium = vol_frac_L/(rho_L*speed_sound_L**2) + vol_frac_V/(rho_V*speed_sound_V**2)
        thermal_equilibrium = T_mix*(vol_frac_L*rho_L/cp_L*dsdp_L**2 + vol_frac_V*rho_V/cp_V*dsdp_V**2)
        compressibility_HEM = mechanical_equilibrium + thermal_equilibrium
        if mass_frac_V < 1e-6:  # Avoid discontinuity when Q_v=0
            a_HEM = speed_sound_L
        elif mass_frac_V > 1.0 - 1e-6:  # Avoid discontinuity when Q_v=1
            a_HEM = speed_sound_V
        else:
            a_HEM = (1/rho_mix/compressibility_HEM)**0.5

        # Store properties in dictionary
        fluid_properties = {}
        fluid_properties['T'] = T_mix
        fluid_properties['p'] = p_mix
        fluid_properties['rhomass'] = rho_mix
        fluid_properties['umass'] = u_mix
        fluid_properties['hmass'] = h_mix
        fluid_properties['smass'] = s_mix
        fluid_properties['cvmass'] = cv_mix
        fluid_properties['cpmass'] = cp_mix
        fluid_properties['gamma'] = fluid_properties['cpmass']/fluid_properties['cvmass']
        fluid_properties['compressibility_factor'] = Z_mix
        fluid_properties['speed_sound'] = a_HEM
        fluid_properties['isentropic_bulk_modulus'] = rho_mix * a_HEM**2
        fluid_properties['isentropic_compressibility'] = 1 / fluid_properties["isentropic_bulk_modulus"]
        fluid_properties['isothermal_bulk_modulus'] = np.nan
        fluid_properties['isothermal_compressibility'] = np.nan
        fluid_properties['isobaric_expansion_coefficient'] = np.nan
        fluid_properties['viscosity'] = mu_mix
        fluid_properties['conductivity'] = k_mix
        fluid_properties['Q'] = mass_frac_V
        fluid_properties['quality_mass'] = mass_frac_V
        fluid_properties['quality_volume'] = vol_frac_V

        return fluid_properties


    def get_property(self, propname):
        """Get the value of a single property"""
        if propname in self.fluid_properties:
            return self.fluid_properties[propname]
        else:
            valid_options = "\n\t".join(self.fluid_properties.keys())
            raise ValueError(f"The requested property '{propname}' is not available. The valid options are:\n\t{valid_options}")

     
    def get_props(self, input_type, prop_1, prop_2):
        """Extract fluid properties for meanline model"""

        # Compute properties in the normal way
        self.compute_properties(input_type, prop_1, prop_2)
        
        # Store a subset of the properties in a dictionary
        fluid_properties={}
        property_subset = ["p", "T", "h", "s", "d", "Z", "a", "mu", "k", "cp", "cv", "gamma"]
        for item in property_subset:
            fluid_properties[item] = self.fluid_properties[item]

        return fluid_properties


if __name__ == "__main__":
    fluid = FluidCoolProp_2Phase('Water', backend="HEOS")

    # Properties of liquid water
    props = fluid.compute_properties(CP.PT_INPUTS, 101325, 300)
    print()
    print("Properties of liquid water")
    print(f"{'Property':35} {'value':6}")
    for key, value in props.items():
        print(f"{key:35} {value:.6e}")

    # # Properties of water/steam mixture
    # props = fluid.compute_properties(CP.QT_INPUTS, 0.5, 300)
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

