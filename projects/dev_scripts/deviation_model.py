
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
from scipy.optimize._numdiff import approx_derivative
import turboflow as tf
    
from scipy import optimize

fac = 1.00


def find_critical_mass_flux(Ma_crit, p0, T0, fluid):

    # Define residual function
    def get_residual(v):
        h = h0 - v**2 / 2
        props = fluid.compute_properties(cp.HmassSmass_INPUTS, h, s0*fac)
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
    props = fluid.compute_properties(cp.HmassSmass_INPUTS, h, s0*fac)
    mass_flux_crit = v_crit * props["rho"]
    return mass_flux_crit, v_crit


def get_beta_subsonic(v_out, p0, T0, fluid, opening, pitch, radius_curvature, Ma_crit):

    Ma = get_mach(v_out, p0, T0, fluid)
    beta_g = np.arccos(opening/pitch)*180/np.pi
    beta_0 = 7/6*(beta_g-10) + 4*pitch/radius_curvature
    if Ma < 0.50:
        beta = beta_0

    elif Ma > 0.50:
        beta = beta_0 + (beta_g - beta_0) * (2*Ma -1)/(2*Ma_crit -1)
        
    return beta

def get_Aungier_beta(v_out, p0, T0, fluid, opening, pitch, radius_curvature, Ma_crit):
    
    beta_g = np.arcsin(opening/pitch)*180/np.pi
    delta_0 = np.arcsin(opening/pitch*(1+(1-opening/pitch)*(beta_g/90)**2))*180/np.pi-beta_g
    Ma = get_mach(v_out, p0, T0, fluid)

    if Ma < 0.50:
        delta = delta_0

    elif Ma > 0.50 and Ma < Ma_crit:
        X = (2*Ma-1) / (2*Ma_crit-1)
        delta = delta_0*(1-10*X**3+15*X**4-6*X**5)
    
    else:
        delta = 0
                        
    beta = np.arccos(opening/pitch)*180/np.pi-delta

    
    return beta

def get_beta_supersonic(v_out, p0, T0, fluid, phi_crit):
    
    if isinstance(v_out, np.ndarray):
        v_out = v_out[0]
        
    props_in = fluid.compute_properties(cp.PT_INPUTS, p0, T0)
    s_out = props_in["s"]
    h_out = props_in["h"] - v_out**2 / 2
    props_out = fluid.compute_properties(cp.HmassSmass_INPUTS, h_out, s_out*fac)
    rho_out = props_out["rho"]
    beta = np.arccos(np.min([1, phi_crit/rho_out/v_out]))*180/np.pi
    # beta = np.arccos(phi_crit/rho_out/v_out)*180/np.pi

    return beta


def get_mach(v_out, p0, T0, fluid):
    props_in = fluid.compute_properties(cp.PT_INPUTS, p0, T0)
    s_out = props_in["s"]
    h_out = props_in["h"] - v_out**2 / 2
    props_out = fluid.compute_properties(cp.HmassSmass_INPUTS, h_out, s_out*fac)
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

def polynomial_blending(f1, f2, x0, x, alpha):
    xx = (x-x0+alpha/2)/alpha
    sigma = 0 + xx**2*(3-2*xx)*(xx > 0)*(xx < 1)+1*(xx > 1)
    return (1-sigma)*f1 + sigma*f2

from scipy.linalg import solve

def polynomial_blending_asymmetric(f1, f2, x0, x, a, alpha):
    
    # Coefficients of the equations
    coefficients = np.array([[1, 1, 1],
                             [2, 3, 4],
                             [a**2, a**3, a**4]])

    # Constants on the right-hand side
    constants = np.array([1, 0, 0.5])

    # Solve the system of equations
    solution = solve(coefficients, constants)
    
    xx = (x-x0)/alpha+a
    sigma = 0 + (solution[0]*xx**2+solution[1]*xx**3+ solution[2]*xx**4)*(xx > 0)*(xx < 1)+1*(xx > 1)
    
    return (1-sigma)*f1 + sigma*f2

def sigmoid_blending_asymmetric(f1, f2, x, n, m):
    sigma = x**n/(x**n+(1-x)**m)
    # sigma = np.clip(sigma, 0, 1)  # Limit sigma between 0 and 1
    return (1-sigma)*f1 + sigma*f2


#########################################################################################

# # Calculate slope of subsonic beta function
# J_subsonic = approx_derivative(get_beta_subsonic, v_crit, method='2-point',
#                       args=(p0, T0, Fluid, opening, pitch, np.inf, Ma_crit))


# def find_supesonic_slope(v):
    
#     J_supersonic = approx_derivative(get_beta_supersonic, v, method='2-point', 
#                                       args=(p0, T0, Fluid, phi_crit))
#     return J_supersonic-J_subsonic

# sol = optimize.root_scalar(find_supesonic_slope, bracket = [200,400], method = 'brentq')
# v_crit = sol.root

# # Calculate beta at point where supersonic slope is equal to the subsonic slope
# beta_change = get_beta_supersonic(sol.root, p0, T0, Fluid, phi_crit)

# def find_Ma_change(Ma):
#     beta = get_beta_subsonic(sol.root, p0, T0, Fluid, opening, pitch, np.inf, Ma)
#     return beta-beta_change

# # Find mach number that the subsonic beta should see in order to get a smooth transfer
# sol = optimize.root_scalar(find_Ma_change, bracket = [0.9,1], method = 'brentq')
# Ma_crit = sol.root

##########################################################################################



# Define case parameters
fluid_name = "air"
Fluid = tf.FluidCoolProp_2Phase(fluid_name, "HEOS")
radius_curvature = np.inf
pitch = 1.00
opening = 0.5
Ma_crit = 1
p0 = 101325
T0 = 300
stagnation_props = Fluid.compute_properties(cp.PT_INPUTS, p0, T0)
h0 = stagnation_props["h"]
s0 = stagnation_props["s"]

k1 = 1
k2 = 1
Ma_inc = 0.5
Ma_blend = (Ma_crit*k1+Ma_inc*k2)/2
alpha = Ma_crit*k1-Ma_inc*k2

# Compute mass flux at the exit plane
phi_crit, v_crit = find_critical_mass_flux(Ma_crit, p0, T0, Fluid)
phi_crit *= opening/pitch


# Define function handles to compute exit angle
# f_subsonic = lambda v: get_Aungier_beta(v, p0, T0, Fluid, opening, pitch, radius_curvature, Ma_crit)

beta_crit = get_beta_supersonic(v_crit, p0, T0, Fluid, phi_crit)
beta_inc = 58
f_subsonic = lambda Ma: beta_inc + (beta_crit - beta_inc)* (Ma - Ma_inc) / (Ma_crit - Ma_inc)

def f_subsonic2(Ma, Ma_inc, Ma_crit, beta_inc, beta_crit):
    x = (Ma - Ma_inc) / (Ma_crit - Ma_inc)
    y = x**2 * (2-x) * (x > 0)
    beta = beta_inc + (beta_crit - beta_inc) * y
    return beta

def f_subsonic3(Ma, Ma_inc, Ma_crit, beta_inc, beta_crit):
    x = (Ma - Ma_inc) / (Ma_crit - Ma_inc)
    y = x**2 * (3-2*x) * (x > 0)
    beta = beta_inc + (beta_crit - beta_inc) * y
    return beta

f_supersonic = lambda v: get_beta_supersonic(v, p0, T0, Fluid, phi_crit)
f_blended = create_blended_function(f_subsonic, f_supersonic, v_crit*0.9, 10)


# Compute the exit angle at different speeds
v_out = np.linspace(100, 350, 100)
Ma_exit = []
beta_subsonic = []
beta_supersonic = []
beta_blended =[]
phi = []
rho = []
for v in v_out:
    Ma_exit.append(get_mach(v, p0, T0, Fluid))
    # beta_subsonic.append(f_subsonic(Ma_exit[-1]))
    
    beta_subsonic.append(f_subsonic3(Ma_exit[-1], Ma_inc, Ma_crit, beta_inc, beta_crit))
    # print(beta_subsonic)
    
    # beta_subsonic.append(58)
    beta_supersonic.append(f_supersonic(v))
    # beta_blended.append(f_subsonic(Ma_exit[-1]))
    # if Ma_exit[-1] < Ma_inc*k2:
    #     beta_blended.append(beta_subsonic[-1])
    if Ma_exit[-1] < Ma_crit*k1:
        x = (Ma_exit[-1] - Ma_inc) / (Ma_crit - Ma_inc)
        x = x*(x>0)*(x<1) + 0 * (x<0) + 1*(x>1)
        beta_blended.append(sigmoid_blending_asymmetric(beta_subsonic[-1], beta_supersonic[-1],x, n=3, m=0.5))
    else:
        beta_blended.append(beta_supersonic[-1])
    h = h0-0.5*v**2
    static_props = Fluid.compute_properties(cp.HmassSmass_INPUTS, h, s0)
    d = static_props["d"]
    rho.append(d)
    phi.append(v*d)

x = np.asarray([beta_subsonic, beta_supersonic])
x0 = 0.8
# x0 = (Ma_crit+0.5)/2
beta_subsonic = np.asarray(beta_subsonic)
beta_supersonic = np.asarray(beta_supersonic)
Ma_exit = np.asarray(Ma_exit)

# beta_blended = polynomial_blending(beta_subsonic, beta_supersonic, x0, Ma_exit, alpha = 0.2)
# beta_blended = polynomial_blending_asymmetric(beta_subsonic, beta_supersonic, x0, Ma_exit, a = 0.5, alpha = 0.1)


# beta_blended = np.max(x, axis = 0)


# Create figure
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Deviation model testing")
ax.set_xlabel(r"$\mathrm{Ma}_{out}$ - Exit Mach number")
ax.set_ylabel(r"$\beta_{\mathrm{out}}$ - Exit flow angle")
ax.set_xscale("linear")
ax.set_yscale("linear")

# Plot simulation data
ax.plot(Ma_exit, np.asarray(beta_subsonic), linewidth=1.25, label="Subsonic")
ax.plot(Ma_exit, np.asarray(beta_supersonic), linewidth=1.25, label="Supersonic")
ax.plot(Ma_exit, np.asarray(beta_blended), linewidth=1.25, linestyle="--", color='black', label="Blended")
ax.set_ylim([55,62])

# Create legend
leg = ax.legend(loc="best")

# fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
# ax1.plot(Ma_exit, phi)
# ax1.set_ylabel('Mass flow flux')
# ax1.set_xlabel(r"$\mathrm{Ma}_{out}$ - Exit Mach number")

# fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
# ax2.plot(Ma_exit, rho)

# fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
# ax3.plot(Ma_exit, np.cos(np.array(beta_blended)*np.pi/180))
# ax3.plot([Ma_exit[0],Ma_exit[-1]],opening/pitch*np.ones(2), 'k--')

# Adjust PAD
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# fig1.tight_layout(pad=1, w_pad=None, h_pad=None)
# fig2.tight_layout(pad=1, w_pad=None, h_pad=None)
# fig3.tight_layout(pad=1, w_pad=None, h_pad=None)

# Show figure
plt.show()


