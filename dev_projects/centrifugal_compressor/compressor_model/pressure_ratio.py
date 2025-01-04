
import CoolProp as cp
import numpy as np
import turboflow as tf

# Boundary conditions
p0_in = 101325*0.1
T0_in = 273.15 + 15
fluid_name = "air"
omega = 52000*2*np.pi/60

# Geometry
r_out = 143/2*1e-3
theta_out = -24.5
z = 12

# Guessed performance variables
eta_tt = 1.0
phi_out = 1.0

# Calculate exit total pressure
fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p0_in, T0_in)
gamma = fluid.cpmass()/fluid.cvmass()
s_in = fluid.smass()
h0_in = fluid.hmass()

u_out = omega*r_out
h0_max = h0_in + 0.5*u_out**2
fluid.update(cp.PSmass_INPUTS, p0_in/2, s_in)
v0 = np.sqrt(2*(h0_max-fluid.hmass()))
s_max = s_in + u_out**2/T0_in

print(h0_max/fluid.hmass())
print(v0)
print(s_max/s_in)

