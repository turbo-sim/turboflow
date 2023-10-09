
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp

import meanline_axial as ml

from scipy import optimize





def find_critical_mass_flux(Ma_crit, p0, T0, fluid):

    # Define residual function
    def get_residual(v):
        h = h0 - v**2 / 2
        props = fluid.compute_properties(cp.HmassSmass_INPUTS, h, s0)
        res = Ma_crit - v/props["a"]
        return res

    # Compute the velocity corresponding to critical Mach
    props_in = fluid.compute_properties(cp.PT_INPUTS, p0, T0)
    s0 = props_in["s"]
    h0 = props_in["h"]
    sol = optimize.root_scalar(get_residual, bracket=[0, 500], method='brentq')
    v_crit = sol.root

    # Compute the critical mass flux
    h = h0 - v_crit**2 / 2
    props = fluid.compute_properties(cp.HmassSmass_INPUTS, h, s0)
    mass_flux_crit = v_crit * props["rho"]
    return mass_flux_crit, v_crit




def get_beta_subsonic(v_out, p0, T0, fluid, opening, pitch, radius_curvature, Ma_crit):

    Ma = get_mach(v_out, p0, T0, fluid)
    beta_g = np.arccos(opening/pitch)*180/np.pi
    beta_0 = 7/6*(beta_g-10) + 4*pitch/radius_curvature
    if Ma < 0.50:
        beta = beta_0

    elif Ma > 0.50:
        beta = beta_0 + (beta_g - beta_0) * (2*Ma -1) / (2*Ma_crit -1)


    return beta
          

def get_beta_supersonic(v_out, p0, T0, fluid, phi_crit):

    props_in = fluid.compute_properties(cp.PT_INPUTS, p0, T0)
    s_out = props_in["s"]
    h_out = props_in["h"] - v_out**2 / 2
    props_out = fluid.compute_properties(cp.HmassSmass_INPUTS, h_out, s_out)
    rho_out = props_out["rho"]
    beta = np.arccos(np.min([1, phi_crit/rho_out/v_out]))*180/np.pi

    return beta


def get_mach(v_out, p0, T0, fluid):
    props_in = fluid.compute_properties(cp.PT_INPUTS, p0, T0)
    s_out = props_in["s"]
    h_out = props_in["h"] - v_out**2 / 2
    props_out = fluid.compute_properties(cp.HmassSmass_INPUTS, h_out, s_out)
    a_out = props_out["a"]
    Ma = v_out/a_out
    return Ma


def create_blended_function(f1, f2, x0, alpha=10):
    """
    Smooth blending of functions f1 and f2 at x0.

    Parameters:
    - f1: First function
    - f2: Second function
    - x0: Blending point
    - alpha: Blending parameter

    Returns:
    - blended_function: A function that represents the blended version of f1 and f2
    """
    
    sigma = lambda x: (1 + np.tanh((x - x0) / alpha)) / 2
    blended_function = lambda x: (1 - sigma(x)) * f1(x) + sigma(x) * f2(x)
    
    return blended_function





# Define case parameters
fluid_name = "air"
Fluid = ml.FluidCoolProp_2Phase(fluid_name, "HEOS")
radius_curvature = np.inf
pitch = 1.00
opening = 0.5
Ma_crit = 1.00
p0 = 101325
T0 = 300

# Compute mass flux at the exit plane
phi_crit, v_crit = find_critical_mass_flux(Ma_crit, p0, T0, Fluid)
phi_crit *= opening/pitch

# Define function handles to compute exit angle
f_subsonic = lambda v: get_beta_subsonic(v, p0, T0, Fluid, opening, pitch, radius_curvature, Ma_crit)
f_supersonic = lambda v: get_beta_supersonic(v, p0, T0, Fluid, phi_crit)
f_blended = create_blended_function(f_subsonic, f_supersonic, v_crit, 1)

# Compute the exit angle at different speeds
v_out = np.linspace(250, 350, 500)
Ma_exit = []
beta_subsonic = []
beta_supersonic = []
beta_blended =[]
for v in v_out:
    Ma_exit.append(get_mach(v, p0, T0, Fluid))
    beta_subsonic.append(f_subsonic(v))
    beta_supersonic.append(f_supersonic(v))
    beta_blended.append(f_blended(v))


# Create figure
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Deviation model testing")
ax.set_xlabel(r"$\mathrm{Ma}_{out}$ - Exit Mach number")
ax.set_ylabel(r"$\beta_{\mathrm{out}}$ - Exit flow angle")
ax.set_xscale("linear")
ax.set_yscale("linear")

# Plot simulation data
ax.plot(Ma_exit, beta_subsonic, linewidth=1.25, label="Subsonic")
ax.plot(Ma_exit, beta_supersonic, linewidth=1.25, label="Supersonic")
ax.plot(Ma_exit, beta_blended, linewidth=1.25, linestyle="--", color='black', label="Blended")

# Create legend
leg = ax.legend(loc="upper left")

# Adjust PAD
fig.tight_layout(pad=1, w_pad=None, h_pad=None)

# Show figure
plt.show()


