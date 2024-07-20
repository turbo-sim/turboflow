# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:31:07 2024

@author: lasseba
"""

import CoolProp as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import os

# Define inputs
p01 = 10e5
T01 = 300
fluid_name = "air"
N = 100
M = 4
pressure_ratio = np.linspace(0.1, 0.9, N)
# Y = np.linspace(0.0, 0.5, M)
Y = np.array([0.01, 0.1, 0.2, 0.3])

# Define input state
fluid = cp.AbstractState("HEOS", fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
h01 = fluid.hmass()
s1 = fluid.smass()

# Set script options
plot_figures = True
export_results = True


def calculate_residual(v2, p01, h01, Y, v0):
    v2 *= v0
    h2 = h01 - 0.5 * v2**2
    fluid.update(cp.HmassP_INPUTS, h2, p2)
    s2 = fluid.smass()
    fluid.update(cp.HmassSmass_INPUTS, h01, s2)
    p02 = fluid.p()
    Y_calc = (p01 - p02) / (p02 - p2)
    return (Y_calc - Y) / Y_calc


# Define solution vectors
Ma_crit = np.zeros(M)
phi = np.zeros((M, N))
Ma = np.zeros((M, N))
s = np.zeros((M, N))

for j in range(len(Y)):
    for i in range(N):
        p2 = p01 * pressure_ratio[i]
        # Calculate isentropic exit state
        fluid.update(cp.PSmass_INPUTS, p2, s1)
        h2_s = fluid.hmass()
        v0 = np.sqrt(2 * (h01 - h2_s))
        sol = optimize.root_scalar(
            calculate_residual, args=(p01, h01, Y[j], v0), bracket=[0.01, 0.999]
        )
        v2 = sol.root * v0
        h2 = h01 - 0.5 * v2**2
        fluid.update(cp.HmassP_INPUTS, h2, p2)
        a2 = fluid.speed_sound()
        d2 = fluid.rhomass()
        s2 = fluid.smass()
        phi2 = d2 * v2
        Ma2 = v2 / a2
        phi[j, i] = phi2
        Ma[j, i] = Ma2
        s[j, i] = s2

    index = np.where(phi[j] == max(phi[j]))[0][0]
    print(Ma[j, index])
    Ma_crit[j] = Ma[j, index]

    # ax.plot(Ma, phi, label = f'Y = {Y[j]:.2f}')

if plot_figures:
    # Create figure 1
    fig1, ax1 = plt.subplots()
    for i in range(M):
        ax1.plot(Ma[i], phi[i], label=f"Y = {Y[i]:.2f}")

    ax1.legend(title="Loss coefficient")
    ax1.set_ylabel("Mass flux [kg/$\mathrm{m}^2\cdot$s]")
    ax1.set_xlabel("Mach")

    # Create figure 2
    fig2, ax2 = plt.subplots()
    ax2.plot(Y, Ma_crit)
    ax2.set_xlabel("Loss coefficient [Y]")
    ax2.set_ylabel("Critical mach [Ma]")

    # Create figure 3
    fig3, ax3 = plt.subplots()
    for i in range(M):
        ax3.plot(pressure_ratio, s[i], label=f"Y = {Y[i]:.2f}")
    ax3.set_xlabel(
        "Total-to-static pressure ratio [$p_\mathrm{out}/p_{0, \mathrm{in}}$]"
    )
    ax3.set_ylabel("Entropy [J/kg$\cdot$K]")

    plt.show()

if export_results:
    # Create output directory
    out_dir = "output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_filename = "mass_flux_curves_compact"
    filepath = os.path.join(out_dir, f"{out_filename}.xlsx")

    pr, y = np.meshgrid(pressure_ratio, Y)
    pr = np.ndarray.flatten(pr)
    y = np.ndarray.flatten(y)
    Ma = np.ndarray.flatten(Ma)
    phi = np.ndarray.flatten(phi)

    data = {"Loss coefficient": y, "Pressure ratio": pr, "Mach": Ma, "Mass flux": phi}

    df = pd.DataFrame(data)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)


