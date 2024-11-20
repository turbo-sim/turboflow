
import CoolProp as cp 
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import turboflow as tf

# get_res_throat return the mass flow rate residual for a given throat velocity
# Investigating the residual for a range of given velocities, and at different mass flow rate, illustrates how
# the resiudal cannot be zero for too high mass flow rates

p01 = 101325
T01 = 273.15 + 15
fluid_name = "air"

radius_hub1 = 40.9/2*1e-3
radius_tip1 = 85.6/2*1e-3
radius_hub1 = 0.09/2
radius_tip1 = 0.28/2
r1 = (radius_tip1+radius_hub1)/2
A1 = np.pi * (radius_tip1**2 - radius_hub1**2)
A2 = A1*0.35
omega = 52000*2*np.pi/60
omega = 14000*2*np.pi/60
u2 = omega*r1
fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
s1 = fluid.smass()
h01 = fluid.hmass()

def get_res_throat(w2, mass_flow_rate, rothalpy, u2):

    """ 
    Impeller: rothalpy is rothalpy, w2 is relative flow and u2 is throat blade speed
    Stationary components: rothalpy is the stagnation enthalpy, w2 is absolute velocity and u2 is zero  
    """

    def get_mass(w2):
        # Throat
        h0_rel2 = rothalpy + 0.5*u2**2 # Assume same radius at throat and inlet
        h2 = h0_rel2 - 0.5*w2**2
        s2 = s1 # Assume isentropic flow
        # s2 = s1 + 1  /  T01 * w2**2 / 2
        fluid.update(cp.HmassSmass_INPUTS, h2, s2)
        d2 = fluid.rhomass()
        m2 = d2*w2*A2
        a2 = fluid.speed_sound()
        Ma = w2/a2
        return m2, Ma
   
    m2, Ma = get_mass(w2)
 
 
    # Mass flow residual
    res = 1-m2/mass_flow_rate
 
    h = 1e-3
    slope = (get_mass(w2+h/2.)[0] - get_mass(w2-h/2.)[0])/h
    # res_mod = res * slope
    res_mod = res * slope / ((0.8 - Ma)**2 + 0.001)
 
    return res, res_mod, Ma
    # return tf.smooth_abs(slope, method="logarithmic", epsilon=1e-3) , Ma

# Explore solutions
# h0_rel1 =  429202.99461396
rothalpy = 423569.8891 - 0.5*u2**2
N = 2000
w2 = np.linspace(0, 450, N)
# mass_flow_rate = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
mass_flow_rate = [4.0, 4.5, 5.0, 5.25, 5.5, 6.0]

 
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
for j in range(len(mass_flow_rate)):
    residual = np.zeros(N)
    residual_mod = np.zeros(N)
    Mach = np.zeros(N)
    for i in range(N):
        res, res_mod, Ma = get_res_throat(w2[i], mass_flow_rate[j], rothalpy, u2)
        residual[i] = res
        residual_mod[i] = res_mod
        Mach[i] = Ma
 
    ax.plot(w2, residual, label = f"Mass flow rate: {mass_flow_rate[j]}")
    ax1.plot(w2, residual_mod, label = f"Mass flow rate: {mass_flow_rate[j]}")
ax.legend()
ax1.legend()
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(False)
ax1.grid(False)
ax.set_ylabel("Residual")
ax.set_xlabel("Mach number")
plt.tight_layout()
plt.show()