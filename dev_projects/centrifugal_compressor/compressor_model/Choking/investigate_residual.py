
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
r1 = (radius_tip1+radius_hub1)/2
A1 = np.pi * (radius_tip1**2 - radius_hub1**2)
A2 = A1*0.35

fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
s1 = fluid.smass()

def get_res_throat(w2, mass_flow_rate, rothalpy, u2):

    """ 
    Impeller: rothalpy is rothalpy, w2 is relative flow and u2 is throat blade speed
    Stationary components: rothalpy is the stagnation enthalpy, w2 is absolute velocity and u2 is zero  
    """

    # Throat
    h0_rel2 = rothalpy + 0.5*u2**2 # Assume same radius at throat and inlet
    h2 = h0_rel2 - 0.5*w2**2
    s2 = s1 # Assume isentropic flow
    fluid.update(cp.HmassSmass_INPUTS, h2, s2)
    d2 = fluid.rhomass()
    m2 = d2*w2*A2

    res = 1-m2/mass_flow_rate

    return res


# Explore solutions
h0_rel1 =  429202.99461396
N = 100
w2 = np.linspace(200, 450, N)
mass_flow_rate = [0.4, 0.45] 

fig, ax = plt.subplots()
for j in range(len(mass_flow_rate)):
    residual = np.zeros(N)
    for i in range(N):
        res = get_res_throat(w2[i], mass_flow_rate[j], h0_rel1)
        residual[i] = res

    ax.plot(w2, residual, label = f"Mass flow rate: {mass_flow_rate[j]}")
ax.legend()
ax.plot([150, 550], [0.00, 0.00], 'k--')
ax.set_xlim([190, 460])
ax.set_ylim([-0.1, 0.23])
ax.grid(False)
ax.set_ylabel("Residual")
ax.set_xlabel("Throat velocity [m/s]")
plt.tight_layout()
plt.show()