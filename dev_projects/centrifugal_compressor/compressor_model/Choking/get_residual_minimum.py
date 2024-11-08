
import CoolProp as cp 
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative
from scipy import optimize
import turboflow as tf
import time

# get_res_throat return the mass flow rate residual for a given throat velocity
# Investigating the residual for a range of given velocities, and at different mass flow rate, illustrates how
# the resiudal cannot be zero for too high mass flow rates

p01 = 101325
T01 = 273.15 + 15
fluid_name = "air"
omega = 52000*2*np.pi/60
omega = 14000*2*np.pi/60

radius_hub1 = 40.9/2*1e-3
radius_tip1 = 85.6/2*1e-3
radius_hub1 = 0.09/2
radius_tip1 = 0.28/2
r1 = (radius_tip1+radius_hub1)/2
A1 = np.pi * (radius_tip1**2 - radius_hub1**2)
A2 = A1*0.5

fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
h01 = fluid.hmass()
s1 = fluid.smass()

def get_res_throat(x, mass_flow_rate, rothalpy, u2):

    """ 
    Get mass flow rate residual at the throat for normalized  velocity x

    Impeller: rothalpy is rothalpy, w2 is relative flow and u2 is throat blade speed
    Stationary components: rothalpy is the stagnation enthalpy, w2 is absolute velocity and u2 is zero  
    """
    w2 = x[0]*u2

    # Throat
    h0_rel2 = rothalpy + 0.5*u2**2 # Assume same radius at throat and inlet
    h2 = h0_rel2 - 0.5*w2**2
    s2 = s1 # Assume isentropic flow
    fluid.update(cp.HmassSmass_INPUTS, h2, s2)
    d2 = fluid.rhomass()
    m2 = d2*w2*A2

    res = 1-m2/mass_flow_rate

    return res
    # return tf.smooth_abs(res, method="logarithmic", epsilon=1e-1)

def get_gradient(x0, mass_flow_rate, rothalpy, u2, eps):

    """
    Get gradient of get_res_throat at point x
    """
    jac = approx_derivative(
        get_res_throat,
        x0,
        abs_step = eps,
        method="3-point",
        args = (
            mass_flow_rate,
            rothalpy,
            u2,
        )
    )

    return jac[0]

# Test function
w2 = 275
mass_flow_rate = 5.0
u2 = r1*omega
rothalpy = h01
rothalpy = 423569.8891 - 0.5*u2**2

start_time = time.time()
rel_step_fd = 1e-4
x = np.array([w2])
eps = rel_step_fd * x

x0 = x/u2
soln = optimize.root(get_gradient, x0, args = (mass_flow_rate, rothalpy, u2, eps))

second_deriv = approx_derivative(
    get_gradient,
    soln.x,
    abs_step = eps,
    method="3-point",
    args = (
        mass_flow_rate,
        rothalpy,
        u2,
        eps,
    )
)

print(second_deriv)
print(get_res_throat(soln.x, mass_flow_rate, rothalpy, u2))
print(soln.x*u2)
print(get_gradient(soln.x, mass_flow_rate, rothalpy, u2, eps))
print(f"Time: {time.time() - start_time}")

# Get second derivative


