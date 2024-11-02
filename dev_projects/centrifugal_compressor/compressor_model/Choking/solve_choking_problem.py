
import CoolProp as cp 
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# This gives an example of how the throat area cannot sustain the given mass flow rate 
# When the specified mass flow rate exceed the critical, get_res has no solution

p01 = 101325
T01 = 273.15 + 15
mass_flow_rate = 0.45
fluid_name = "air"
omega = 52000*2*np.pi/60
alpha1 = 0.0

radius_hub1 = 40.9/2*1e-3
radius_tip1 = 85.6/2*1e-3
r1 = (radius_tip1+radius_hub1)/2
A1 = np.pi * (radius_tip1**2 - radius_hub1**2)
A2 = A1*0.35

fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
h01 = fluid.hmass()
s1 = fluid.smass()

def get_res(x, mass_flow_rate):

    v1 = x[0]*omega*r1
    w2 = x[1]*omega*r1

    # Inlet
    u1 = r1*omega
    vt1 = v1*np.sin(alpha1)
    vm1 = v1*np.cos(alpha1)
    wt1 = vt1 - u1
    wm1 = vm1
    w1 = np.sqrt(wm1**2 + wt1**2)
    h1 = h01 - 0.5*v1**2
    fluid.update(cp.HmassSmass_INPUTS, h1, s1)
    d1 = fluid.rhomass()
    m1 = d1*vm1*A1
    h0_rel1 = h1 + 0.5*w1**2

    # Throat
    h0_rel2 = h0_rel1 # Assume same radius at throat and inlet
    h2 = h0_rel2 - 0.5*w2**2
    s2 = s1 # Assume isentropic flow
    fluid.update(cp.HmassSmass_INPUTS, h2, s2)
    d2 = fluid.rhomass()
    a2 = fluid.speed_sound()
    p2 = fluid.p()
    m2 = d2*w2*A2
    Ma2 = w2/a2

    res = [
        1-m1/mass_flow_rate,
        1-m2/mass_flow_rate,
    ]

    return res, p2


# Solve equations
x0 = [0.5, 0.5]
N = 60
mass_flow_rates = np.linspace(0.2, 0.45, N)
machs = np.zeros(N)
messages = []
for i in range(N):
    mass_flow_rate = mass_flow_rates[i]
    soln = optimize.root(lambda x0, mass_flow_rate: get_res(x0, mass_flow_rate)[0], x0, args = (mass_flow_rate,), method = "hybr", tol = 1e-6)
    res, mach = get_res(soln.x, mass_flow_rate)
    print(soln.success)
    machs[i] = mach

fig, ax = plt.subplots()
ax.plot(mass_flow_rates, machs)
print(machs)
# ax.set_ylim([4e5, 4.3e5])
plt.show()

# Why is the relative stagnation state constant across different mass flow rates
# When get_res is possible to solve, the given mass flow rate is feasable (and not above critical)