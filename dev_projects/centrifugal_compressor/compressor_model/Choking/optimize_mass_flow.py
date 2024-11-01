

import CoolProp as cp 
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import time 

# This script maxmizes the mass flow rate based on the given relative enthalpy


p01 = 101325
T01 = 273.15 + 15
fluid_name = "air"
omega = 52000*2*np.pi/60
alpha1 = 0.00

radius_hub1 = 40.9/2*1e-3
radius_tip1 = 85.6/2*1e-3
r1 = (radius_tip1+radius_hub1)/2
A1 = np.pi * (radius_tip1**2 - radius_hub1**2)
A2 = A1*0.35
u = omega*r1

fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
h01 = fluid.hmass()
s1 = fluid.smass()


def get_res_throat_isentropic(x):

    """
    Can introduce losses
    Assumes radius at throat is the same as inlet (constant relative enthalpy)
    """
    v1 = x[0]*u
    w2 = x[1]*u

    # Inlet
    vm1 = v1*np.cos(alpha1)
    vt1 = v1*np.sin(alpha1)
    wt1 = vt1 - u
    wm1 = vm1
    w1 = np.sqrt(wt1**2 + wt1**2)

    h1 = h01 - 0.5*v1**2
    fluid.update(cp.HmassSmass_INPUTS, h1, s1)
    d1 = fluid.rhomass()
    m1 = d1*v1*np.cos(alpha1)*A1
    h0_rel1 = h1 + 0.5*w1**2

    # Throat
    h0_rel2 = h0_rel1 # Assume same radius at throat and inlet
    h2 = h0_rel2 - 0.5*w2**2
    s2 = s1
    fluid.update(cp.HmassSmass_INPUTS, h2, s2)
    d2 = fluid.rhomass()
    p2 = fluid.p()
    a2 = fluid.speed_sound()
    m2 = d2*w2*A2
    Ma2 = w2/a2

    # Constraint
    delta_m = m1-m2
    cons = np.array([delta_m])

    # Store results
    results = {
        "Ma2" : Ma2
    }

    return -1*m2, cons, results 

def get_res_throat(x, Y_given):

    """
    Can introduce losses
    Assumes radius at throat is the same as inlet (constant relative enthalpy)
    """

    w2 = x[0]*u
    s2 = x[1]*s1
    v1 = x[2]*u

    # Inlet
    vm1 = v1*np.cos(alpha1)
    vt1 = v1*np.sin(alpha1)
    wt1 = vt1 - u
    wm1 = vm1
    w1 = np.sqrt(wt1**2 + wt1**2)

    h1 = h01 - 0.5*v1**2
    fluid.update(cp.HmassSmass_INPUTS, h1, s1)
    d1 = fluid.rhomass()
    m1 = d1*v1*np.cos(alpha1)*A1
    h0_rel1 = h1 + 0.5*w1**2

    # Throat
    h0_rel2 = h0_rel1 # Assume same radius at throat and inlet
    h2 = h0_rel2 - 0.5*w2**2
    fluid.update(cp.HmassSmass_INPUTS, h2, s2)
    d2 = fluid.rhomass()
    p2 = fluid.p()
    a2 = fluid.speed_sound()
    m2 = d2*w2*A2
    Ma2 = w2/a2

    # Calculate loss coefficient
    fluid.update(cp.PSmass_INPUTS, p2, s1)
    h2s = fluid.hmass()
    Y = 2*(h2-h2s)/v1**2

    # Constraint
    delta_m = m1-m2
    delta_Y = Y-Y_given
    cons = np.array([delta_m, delta_Y])

    # Store results
    results = {
        "Ma2" : Ma2
    }

    return -1*m2, cons, results 

# Optimize mass flow rate (isentropic)
constraints = {
    "type" : "eq",
    "fun" : lambda x: get_res_throat_isentropic(x)[1],
}
start_time = time.time()
soln = optimize.minimize(lambda x: get_res_throat_isentropic(x)[0], x0 = np.array([0.5, 0.5]), constraints=constraints)
print(soln)
print(f"Optimization used {time.time() - start_time} seconds")

# Optimize mass flow rate (losses)
# Y_given = 1.0
# constraints = {
#     "type" : "eq",
#     "fun" : lambda x, Y: get_res_throat(x, Y)[1],
#     "args" : (Y_given,)
# }
# soln = optimize.minimize(lambda x, Y: get_res_throat(x,Y)[0], x0 = np.array([0.5, 1.0, 0.5]), constraints = constraints, args = (Y_given,))

# m, cons, results = get_res_throat(soln.x, Y_given)
# print(results)