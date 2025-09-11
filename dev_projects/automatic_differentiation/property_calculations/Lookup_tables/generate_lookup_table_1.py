#%% Initialization

import os,psutil #Look at the number of physical cores.

#NOTE: On modern machines might want to use the number of high performance
#cores, because different core architectures can cause problems in parallel
#processing

NCORES=psutil.cpu_count(logical=False) #Before loading jax, force it to see the CPU count we want
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=%d"%NCORES
import jax
jax.config.update("jax_enable_x64", True) #By default jax uses 32 bit, for scientific computing we need 64 bit precision
import jax.numpy as jnp
import pickle
import pandas as pd

#To read more regarding the automatic parallelization in jax
#https://jax.readthedocs.io/en/latest/sharded-computation.html

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec , NamedSharding

import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import time
from jax_bicubic_HEOS_interpolation_1 import (
    compute_bicubic_coefficients_of_ij,
    bicubic_interpolant,
    inverse_interpolant_scalar_DP,
    inverse_interpolant_scalar_hD,
)
import turboflow as tf

float64=jnp.dtype('float64')
complex128=jnp.dtype('complex128')

#-----------------------------------------------------------------------
#%% Grid and Fluid definition

# Defining the Grid
N=10       # Grid size in h direction
M=10      # Grid size in log(P) direction
name="CO2" # Fluid selection

# Grid boundaries
hmin=200000  # units - J/kg
hmax=600000 # units - J/kg
Pmin=20*1e5 # units - Pa
Pmax=200*1e5 # units - Pa
Lmin=jnp.log(Pmin) # Log of P
Lmax=jnp.log(Pmax) # Log of P

# Number of random points to test the error
Npoints=50000

#Number of times to repeat the computation. used to stabilize the timing of  
#very fast computations when using time.time()
Nrepeats=int(1e6/Npoints)

fluid = tf.Fluid(name)

#-----------------------------------------------------------------------
#%% Properties to be calculated

properties = {
    'T': 'T',       # Temperature [K]
    'd': 'D',       # Density [kg/m³]
    's': 'S',       # Entropy [J/kg/K]
    'mu': 'V',      # Viscosity [Pa·s]
    'k': 'L',       # Thermal conductivity [W/m/K]
}

# Create the grid of (h, P)
h_vals = jnp.linspace(hmin, hmax, N)
P_vals = jnp.linspace(Lmin, Lmax, M)
h_grid, P_grid = jnp.meshgrid(h_vals, jnp.exp(P_vals), indexing='ij')
deltah = h_vals[1]-h_vals[0]
deltaL = P_vals[1]-P_vals[0]

# Evaluate each property on the grid using CoolProp
raw_property_grids = {}

print(" Generating raw property data using CoolProp...")
for key, coolprop_symbol in properties.items():
    prop_grid = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            h = h_vals[i]
            P = jnp.exp(P_vals[j])
            try:
                # val = cp.PropsSI(coolprop_symbol, 'H', h, 'P', P, name)
                val = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h, P)
            except:
                val = np.nan  # or use extrapolation/default value
            prop_grid[i, j] = val[key]
    raw_property_grids[key] = prop_grid

print(" Raw property grids created.")

## Check raw_property_grids here to see if the values are proper or not ##

print(" Checking raw_property_grids for NaNs and basic stats...\n")

for key, grid in raw_property_grids.items():
    num_nans = np.isnan(grid).sum()
    num_zeros = np.sum(grid == 0)
    min_val = np.nanmin(grid)
    max_val = np.nanmax(grid)
    mean_val = np.nanmean(grid)
    total_vals = grid.size

    print(f"  Property: {key}")
    print(f"   → Shape        : {grid.shape}")
    print(f"   → NaN count    : {num_nans}")
    print(f"   → Zero count   : {num_zeros} ({num_zeros / total_vals * 100:.2f}%)")
    print(f"   → Min, Max     : {min_val:.3e}, {max_val:.3e}")
    print(f"   → Mean (valid) : {mean_val:.3e}")
    print("-" * 50)

#-----------------------------------------------------------------------
#%% Property function from fluid properties
# Wrapper: takes h and log(P), returns the property
def func(h, logP):
    P = jnp.exp(logP)  # Convert log(P) back to actual P
    return tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h, P)

#-----------------------------------------------------------------------
#%% Fixing the zero coeffs values at the edges

# Function to fix zero coefficients
# def fix_zero_coeffs(coeffs):
#     Nh, Np, *_ = coeffs.shape
#     fixed_coeffs = coeffs.copy()

#     for i in range(Nh):
#         for j in range(Np):
#             if jnp.all(fixed_coeffs[i, j] == 0):
#                 neighbor_coeffs = []

#                 for di in [-1, 0, 1]:
#                     for dj in [-1, 0, 1]:
#                         ni, nj = i + di, j + dj
#                         if (0 <= ni < Nh) and (0 <= nj < Np) and not (di == 0 and dj == 0):
#                             neighbor = fixed_coeffs[ni, nj]
#                             if not jnp.all(neighbor == 0):
#                                 neighbor_coeffs.append(neighbor)

#                 if neighbor_coeffs:
#                     fixed_coeffs = fixed_coeffs.at[i, j].set(jnp.mean(jnp.stack(neighbor_coeffs), axis=0))

#     return fixed_coeffs

#-----------------------------------------------------------------------
#%% Interpolation table generation

def generate_interpolation_table(h_vals, P_vals, raw_property_grids, save_path='interpolation_table.pkl'):
    """
    Compute and save interpolated values and partial derivatives using bicubic interpolation.
    """
    Nh, Np = len(h_vals), len(P_vals)
    table = {
        'h': h_vals,
        'P': jnp.exp(P_vals),
    }
    
    # Start time for progress tracking
    t0 = time.time()

    for prop, raw_grid in raw_property_grids.items():
        # Initialize coeffs as a 3D array of shape (N, M, 16)
        coeffs = jnp.zeros((Nh, Np, 16), dtype=jnp.float32)
        print(f"Processing property: {prop}")
        
        # Initialize storage arrays
        val_grid = np.zeros((Nh, Np))
        d_dh_grid = np.zeros((Nh, Np))
        d_dP_grid = np.zeros((Nh, Np))
        d2_dhdP_grid = np.zeros((Nh, Np))

        ## To check the gradient values ##
        # Allocate arrays
        fx = jnp.zeros((Nh, Np))    # ∂f/∂h
        fy = jnp.zeros((Nh, Np))    # ∂f/∂log(P)
        fxy = jnp.zeros((Nh, Np))   # ∂²f/∂h∂log(P)

        # Loop through the grid
        for i in range(Nh):
            for j in range(Np):
                h_val = hmin + deltah * i
                logP_val = Lmin + deltaL * j

                # First-order derivatives
                df_dh = jax.jacrev(tf.get_props_custom_jvp, argnums=(2))(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))
                df_dlogP = jax.jacrev(tf.get_props_custom_jvp, argnums=(3))(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))

                # Mixed second derivative ∂²f/∂h∂logP
                # d2f_dhdlogP = jax.jacrev(lambda fluid_info, input_state, h, logP: jax.jacrev(tf.get_props_custom_jvp, argnums=2)(fluid_info, input_state, h, logP), argnums=3)
                # d2f_dhdlogP = jax.jacrev(jax.jacrev(tf.get_props_custom_jvp, argnums=(2)), argnums=(3))(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))
                # mixed = d2f_dhdlogP(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))
                eps = 1e-5

                # Evaluate df/dh at logP + eps
                df_dh_plus = jax.jacrev(tf.get_props_custom_jvp, argnums=2)(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val + eps))

                # Evaluate df/dh at logP - eps
                df_dh_minus = jax.jacrev(tf.get_props_custom_jvp, argnums=2)(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val - eps))

                # Approximate mixed derivative
                d2f_dhdlogP = (df_dh_plus[prop] - df_dh_minus[prop]) / (2 * eps)


                fx = fx.at[i, j].set(df_dh[prop])
                fy = fy.at[i, j].set(df_dlogP[prop])
                fxy = fxy.at[i, j].set(d2f_dhdlogP)
                

        # Inspect gradient values at a few points
        # print(f"\n Gradient inspection for property: {prop}")
        # num_samples = 2  # adjust as needed
        # i_indices = jnp.linspace(0, Nh - 1, num_samples, dtype=int)
        # j_indices = jnp.linspace(0, Np - 1, num_samples, dtype=int)

        # for i in i_indices:
        #     for j in j_indices:
        #         print(f"(i={i}, j={j})  h = {h_vals[i]:.2f},  P = {jnp.exp(P_vals[j])/1e5:.2f} bar")
        #         print(f"   raw      = {raw_grid[i, j]:.3e}")
        #         print(f"   ∂f/∂h    = {fx[i, j]:.3e}")
        #         print(f"   ∂f/∂P    = {fy[i, j]:.3e}")
        #         print(f"   ∂²f/∂h∂P = {fxy[i, j]:.3e}")
        #         print("-" * 40)

        for i, hi in enumerate(h_vals):
            for j, Pj in enumerate(P_vals):
                # Extract scalar values at the current grid point

                # f_val = raw_grid[i, j]
                # fx_val = fx[i, j]
                # fy_val = fy[i, j]
                # fxy_val = fxy[i, j]

                # Compute bicubic coefficients from point-wise values
                # temp_coeffs = compute_bicubic_coefficients_of_ij(i, j, f_val, fx_val, fy_val, fxy_val)
                temp_coeffs = compute_bicubic_coefficients_of_ij(i, j, raw_grid, fx, fy, fxy)

                # Store the computed coefficients
                coeffs = coeffs.at[i, j].set(temp_coeffs)

        # print("\n Inspecting all edge coefficients...")

        # Nh, Np = coeffs.shape[:2]
        # zero_count = 0
        # edge_total = 0

        # # Top and Bottom rows
        # for i in [0, Nh - 1]:
        #     for j in range(Np):
        #         coeff_val = coeffs[i, j]
        #         is_zero = jnp.all(coeff_val == 0).item()
        #         print(f"Edge ({i},{j}) → All zero? {is_zero} | Coeffs: {coeff_val}")
        #         zero_count += is_zero
        #         edge_total += 1

        # # Left and Right columns (excluding corners to avoid double-counting)
        # for i in range(1, Nh - 1):
        #     for j in [0, Np - 1]:
        #         coeff_val = coeffs[i, j]
        #         is_zero = jnp.all(coeff_val == 0).item()
        #         print(f"Edge ({i},{j}) → All zero? {is_zero} | Coeffs: {coeff_val}")
        #         zero_count += is_zero
        #         edge_total += 1

        # print(f"\n Total edge nodes checked: {edge_total}")
        # print(f" Zero-valued coefficient nodes on edges: {zero_count}")

        # print(" Coefficient grid sanity check:")
        # print("   → Total cells        :", coeffs.shape[0] * coeffs.shape[1])
        # print("   → Non-zero coeff cells:",
        # jnp.count_nonzero(jnp.linalg.norm(coeffs, axis=-1) > 0))

        # Now coeffs is a 3D array with the correct shape (Nh, Np, 16) ready for interpolation
        # Loop through the grid to perform the interpolation
        for i, hi in enumerate(h_vals):
            if i % (Nh / 10) < 1:  # Example progress print every 10% progress
                print(f'Progress: {i / Nh * 100:.2f}% done in {time.time() - t0:.2f}s')
            
            for j, Pj in enumerate(P_vals):
                # Extract the relevant coefficients for interpolation

                val = bicubic_interpolant(hi, Pj, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
                dval_dh = jax.grad(bicubic_interpolant, argnums=0)(hi, Pj, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
                dval_dP = jax.grad(bicubic_interpolant, argnums=1)(hi, Pj, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
                d2val_dhdP = jax.grad(jax.grad(bicubic_interpolant, argnums=1), argnums=0)(hi, Pj, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
 
                # Store the interpolated values and derivatives
                val_grid[i, j] = val
                d_dh_grid[i, j] = dval_dh
                d_dP_grid[i, j] = dval_dP
                d2_dhdP_grid[i, j] = d2val_dhdP

        # Store property and derivatives
        table[prop] = {
            'value': val_grid,
            'd_dh': d_dh_grid,
            'd_dP': d_dP_grid,
            'd2_dhdP': d2_dhdP_grid,
            'coeffs': coeffs,
        }

    with open(save_path, 'wb') as f:
        pickle.dump(table, f)

    print(f"\n Interpolation table saved to: {os.path.abspath(save_path)}")

    save_interpolation_table_as_csv(table, 'property_table.csv')
    # or
    save_interpolation_table_as_parquet(table, 'property_table.parquet')

#-----------------------------------------------------------------------
#%% Saving the Table codes

def save_interpolation_table_as_csv(table, filename='interpolation_table.csv'):
    h_vals = table['h']
    P_vals = table['P']

    records = []

    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            row = {
                'h': h,
                'P': P
            }
            for prop, data in table.items():
                if prop in ['h', 'P']:
                    continue
                row[f'{prop}'] = data['value'][i, j]
                row[f'd{prop}_dh'] = data['d_dh'][i, j]
                row[f'd{prop}_dP'] = data['d_dP'][i, j]
                row[f'd2{prop}_dhdP'] = data['d2_dhdP'][i, j]
            records.append(row)

    df = pd.DataFrame.from_records(records)
    df.to_csv(filename, index=False)
    print(f" CSV property table saved to: {os.path.abspath(filename)}")



def save_interpolation_table_as_parquet(table, filename):
    """
    Convert the dictionary-based table into a flat tabular format and save as a Parquet file.
    """
    rows = []

    h_vals = np.array(table['h'])  # convert from jax to numpy
    P_vals = np.array(table['P'])

    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            row = {'h': float(h), 'P': float(P)}
            for prop, data in table.items():
                if prop in ['h', 'P']:
                    continue
                row[f'{prop}'] = float(data['value'][i, j])
                row[f'{prop}_d_dh'] = float(data['d_dh'][i, j])
                row[f'{prop}_d_dP'] = float(data['d_dP'][i, j])
                row[f'{prop}_d2_dhdP'] = float(data['d2_dhdP'][i, j])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(filename, index=False)
    print(f" Property table saved as: {filename}")

#-----------------------------------------------------------------------
#%% Call interpolation table generator

generate_interpolation_table(h_vals, P_vals, raw_property_grids, save_path='interpolation_table.pkl')

#-----------------------------------------------------------------------
#%% Plotting Property Surfaces

# Global font settings for consistency and readability
rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
})

def plot_property_surface(property_name, h_vals, P_vals, raw_grid):
    H, P = np.meshgrid(h_vals, np.exp(P_vals), indexing='ij')  # pressure was in log-space
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(H, P / 1e5, raw_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('Enthalpy [J/kg]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_zlabel(property_name)
    ax.set_title(f'{property_name} Surface from CoolProp', pad=20)

    # Add colorbar and legend using proxy artist
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(raw_grid)
    fig.colorbar(m, ax=ax, shrink=0.6, aspect=10, label=property_name)

    # Fake legend for surface
    legend_proxy = [Line2D([0], [0], marker='s', color='w', label='Raw CoolProp Values',
                           markerfacecolor='green', markersize=10)]
    ax.legend(handles=legend_proxy, loc='upper right')

    plt.tight_layout()
    plt.show()

# Example: Plot raw Temperature and Viscosity surfaces
plot_property_surface('Temperature [K]', h_vals, P_vals, raw_property_grids['T'])
plot_property_surface('Viscosity [Pa·s]', h_vals, P_vals, raw_property_grids['mu'])

#-----------------------------------------------------------------------
#%% Comparing to Interpolated data

with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

# Plot interpolated Temperature
plot_property_surface('Interpolated Temperature [K]', interp_table['h'], np.log(interp_table['P']), interp_table['T']['value'])

# Plot interpolated Viscosity
plot_property_surface('Interpolated Viscosity [Pa·s]', interp_table['h'], np.log(interp_table['P']), interp_table['mu']['value'])

# -----------------------------------------------------------------------
# %% Verification of Interpolation Table with CoolProp values

# Load the generated table
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']

properties_to_test = ['T', 'd', 's', 'mu', 'k']

# Map properties to CoolProp symbols
properties = {
    'T': 'T',
    'd': 'D',
    's': 'S',
    'mu': 'V',
    'k': 'L',
}

fluid = 'CO2'

# Define function to compute relative error
def compute_relative_error(true_val, interp_val):
    return np.abs((interp_val - true_val) / true_val)

# Loop over each property
for prop in properties_to_test:
    print(f"\n Testing property: {prop}")
    table_vals = interp_table[prop]['value']
    true_vals = np.zeros_like(table_vals)

    # Compute true values using CoolProp
    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            try:
                true_val = cp.PropsSI(properties[prop], 'H', h, 'P', P, fluid)
            except:
                true_val = np.nan
            true_vals[i, j] = true_val

    # Compute errors
    abs_error = np.abs(table_vals - true_vals)
    rel_error = compute_relative_error(true_vals, table_vals)
    percent_error = rel_error * 100

    # Mask NaNs if needed
    valid_mask = ~np.isnan(true_vals)

    # Print stats
    print(f"   → Absolute error: min={np.nanmin(abs_error):.3e}, max={np.nanmax(abs_error):.3e}, mean={np.nanmean(abs_error):.3e}, std={np.nanstd(abs_error):.3e}")
    print(f"   → Relative error: min={np.nanmin(rel_error):.3e}, max={np.nanmax(rel_error):.3e}, mean={np.nanmean(rel_error):.3e}, std={np.nanstd(rel_error):.3e}")
    print(f"   → Percentage error: min={np.nanmin(percent_error):.3e}%, max={np.nanmax(percent_error):.3e}%, mean={np.nanmean(percent_error):.3e}%, std={np.nanstd(percent_error):.3e}%")

    # ---------- Plot percentage error contour ----------
    plt.figure(figsize=(8, 5))
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals, indexing='ij')

    # Linear spaced levels
    # levels = np.linspace(0, np.nanmax(percent_error), 15)
    # levels = np.linspace(0, 100, 21)  # np.logspace(-12, 2, 15) 
    levels = np.logspace(-12, 2, 15)

    contour = plt.contourf(H_mesh, P_mesh / 1e5, percent_error, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(contour, label='Percentage error [%]')
    plt.xlabel('Enthalpy [J/kg]')
    plt.ylabel('Pressure [bar]')
    plt.title(f'Percentage error contour for {prop}')
    # Remove log scale
    # plt.yscale('log')

    # ---------- Saturation curve overlay ----------
    P_sats = np.linspace(P_vals[0], P_vals[-1], 200)
    h_l = []
    h_v = []
    P_sats_bar = []

    for P_sat in P_sats:
        try:
            hl = cp.PropsSI('H', 'P', P_sat, 'Q', 0, fluid)
            hv = cp.PropsSI('H', 'P', P_sat, 'Q', 1, fluid)
            h_l.append(hl)
            h_v.append(hv)
            P_sats_bar.append(P_sat / 1e5)
        except:
            pass

    plt.plot(h_l, P_sats_bar, 'w--', lw=1.5, label='Saturation liquid')
    plt.plot(h_v, P_sats_bar, 'w--', lw=1.5, label='Saturation vapor')
    plt.legend()

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------
#%% Validation: Compare table values to raw property grid points

print("\n Checking interpolation table against raw property grids data")

for prop in properties_to_test:
    print(f"\n🔍 Property: {prop}")

    abs_error = np.abs(interp_table[prop]['value'] - raw_property_grids[prop])
    rel_error = np.abs((interp_table[prop]['value'] - raw_property_grids[prop]) / raw_property_grids[prop])
    percent_error = rel_error * 100

    # Mask NaNs if needed
    valid_mask = ~np.isnan(raw_property_grids[prop])
    abs_error = abs_error[valid_mask]
    rel_error = rel_error[valid_mask]

    print(f"   → Absolute error: max={np.max(abs_error):.3e}, mean={np.mean(abs_error):.3e}")
    print(f"   → Relative error: max={np.max(rel_error):.3e}, mean={np.mean(rel_error):.3e}")

    # --------- Contour plot ---------
    plt.figure(figsize=(8, 5))
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals, indexing='ij')
    levels = np.logspace(-12, 0, 13) * 100  # percentage error levels

    contour = plt.contourf(H_mesh, P_mesh / 1e5, percent_error, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(contour, label='Percentage error [%]')
    plt.xlabel('Enthalpy [J/kg]')
    plt.ylabel('Pressure [bar]')
    plt.title(f'Percentage error contour for {prop}')
    plt.yscale('log')

    # --------- Saturation curve overlay ---------
    P_sats = np.logspace(np.log10(Pmin), np.log10(Pmax), 200)
    h_l = []
    h_v = []
    P_sats_bar = []

    for P_sat in P_sats:
        try:
            hl = cp.PropsSI('H', 'P', P_sat, 'Q', 0, fluid)
            hv = cp.PropsSI('H', 'P', P_sat, 'Q', 1, fluid)
            h_l.append(hl)
            h_v.append(hv)
            P_sats_bar.append(P_sat / 1e5)
        except:
            # Skip points where CO2 is supercritical (no saturation data)
            pass

    plt.plot(h_l, P_sats_bar, 'w--', lw=1.5, label='Saturation liquid')
    plt.plot(h_v, P_sats_bar, 'w--', lw=1.5, label='Saturation vapor')
    plt.legend()

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------
# %% Midpoint interpolation accuracy check (between grid points)

# Load the generated table
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']


Nh = len(h_vals)
Np = len(P_vals)
hmin, hmax = h_vals[0], h_vals[-1]
Lmin, Lmax = np.log(P_vals[0]), np.log(P_vals[-1])

# Loop over each property
for prop in properties_to_test:
    print(f"\n Testing midpoints for property: {prop}")

    symbol = properties[prop]
    coeffs = interp_table[prop]['coeffs'] if 'coeffs' in interp_table[prop] else None  # If you have coeffs stored
    abs_error_grid = np.zeros((Nh - 1, Np - 1))
    percent_error_grid = np.zeros((Nh - 1, Np - 1))

    for i in range(Nh - 1):
        for j in range(Np - 1):
            h_mid = 0.5 * (h_vals[i] + h_vals[i + 1])
            P_mid = 0.5 * (P_vals[j] + P_vals[j + 1])
            try:
                val_true = cp.PropsSI(symbol, 'H', h_mid, 'P', P_mid, fluid)
                val_interp = bicubic_interpolant(
                    h_mid, np.log(P_mid),
                    h_vals, np.log(P_vals),
                    interp_table[prop]['coeffs'],
                    Nh, Np, hmin, hmax, Lmin, Lmax
                )
                abs_error = np.abs(val_interp - val_true)
                percent_error = (abs_error / np.abs(val_true)) * 100
            except:
                abs_error = np.nan
                percent_error = np.nan

            abs_error_grid[i, j] = abs_error
            percent_error_grid[i, j] = percent_error

    # Print summary stats
    valid_mask = ~np.isnan(abs_error_grid)
    print(f"   → Absolute error: min={np.nanmin(abs_error_grid):.3e}, max={np.nanmax(abs_error_grid):.3e}, mean={np.nanmean(abs_error_grid):.3e}, std={np.nanstd(abs_error_grid):.3e}")
    print(f"   → Percentage error: min={np.nanmin(percent_error_grid):.3e}%, max={np.nanmax(percent_error_grid):.3e}%, mean={np.nanmean(percent_error_grid):.3e}%, std={np.nanstd(percent_error_grid):.3e}%")

    # ---------- Plot percentage error contour ----------
    plt.figure(figsize=(8, 5))

    # Midpoint mesh
    h_mid_vals = 0.5 * (h_vals[:-1] + h_vals[1:])
    P_mid_vals = 0.5 * (P_vals[:-1] + P_vals[1:])
    H_mesh, P_mesh = np.meshgrid(h_mid_vals, P_mid_vals, indexing='ij')

    levels = np.linspace(0, 100, 21)

    contour = plt.contourf(H_mesh, P_mesh / 1e5, percent_error_grid, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(contour, label='Percentage error [%]')
    plt.xlabel('Enthalpy [J/kg]')
    plt.ylabel('Pressure [bar]')
    plt.title(f'Percentage error at midpoints for {prop}')

    # ---------- Saturation curve overlay ----------
    P_sats = np.linspace(P_vals[0], P_vals[-1], 200)
    h_l, h_v, P_sats_bar = [], [], []

    for P_sat in P_sats:
        try:
            hl = cp.PropsSI('H', 'P', P_sat, 'Q', 0, fluid)
            hv = cp.PropsSI('H', 'P', P_sat, 'Q', 1, fluid)
            h_l.append(hl)
            h_v.append(hv)
            P_sats_bar.append(P_sat / 1e5)
        except:
            pass

    plt.plot(h_l, P_sats_bar, 'w--', lw=1.5, label='Saturation liquid')
    plt.plot(h_v, P_sats_bar, 'w--', lw=1.5, label='Saturation vapor')
    plt.legend()

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------
# %% Numerical derivatives from interpolated values

# Load the interpolation table
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']

properties_to_test = ['T', 'd', 's', 'mu', 'k']

# Define finite difference function
def finite_diff(f_p, f_m, delta):
    return (f_p - f_m) / (2 * delta)

# Loop over each property
for prop in properties_to_test:
    print(f"\n Verifying numerical derivatives on interpolated values: {prop}")

    d_dh_table = interp_table[prop]['d_dh']
    d_dP_table = interp_table[prop]['d_dP']
    d2_dhdP_table = interp_table[prop]['d2_dhdP']

    Nh = len(h_vals)
    Np = len(P_vals)

    fd_dh_num = np.zeros_like(d_dh_table)
    fd_dP_num = np.zeros_like(d_dP_table)
    fd_dhdP_num = np.zeros_like(d2_dhdP_table)

    # Loop through grid
    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):

            # delta_h = 1e-4 * np.abs(h) if np.abs(h) > 1 else 1e-4
            # delta_P = 1e-5 * np.abs(P) if np.abs(P) > 1 else 1e-4

            delta_h = 1e2 
            delta_P = 1e3 

            try:
                # ∂f/∂h
                f_p_h = bicubic_interpolant(h + delta_h, np.log(P), h_vals, np.log(P_vals),
                                            interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                f_m_h = bicubic_interpolant(h - delta_h, np.log(P), h_vals, np.log(P_vals),
                                            interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                fd_dh_num[i, j] = finite_diff(f_p_h, f_m_h, delta_h)

                # ∂f/∂P
                f_p_P = bicubic_interpolant(h, np.log(P + delta_P), h_vals, np.log(P_vals),
                                            interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                f_m_P = bicubic_interpolant(h, np.log(P - delta_P), h_vals, np.log(P_vals),
                                            interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                fd_dP_num[i, j] = finite_diff(f_p_P, f_m_P, delta_P)

                # ∂²f/∂h∂P
                f_pp = bicubic_interpolant(h + delta_h, np.log(P + delta_P), h_vals, np.log(P_vals),
                                           interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                f_pm = bicubic_interpolant(h + delta_h, np.log(P - delta_P), h_vals, np.log(P_vals),
                                           interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                f_mp = bicubic_interpolant(h - delta_h, np.log(P + delta_P), h_vals, np.log(P_vals),
                                           interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                f_mm = bicubic_interpolant(h - delta_h, np.log(P - delta_P), h_vals, np.log(P_vals),
                                           interp_table[prop]['coeffs'], Nh, Np, h_vals[0], h_vals[-1], np.log(P_vals[0]), np.log(P_vals[-1]))
                fd_dhdP_num[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * delta_h * delta_P)

            except:
                fd_dh_num[i, j] = np.nan
                fd_dP_num[i, j] = np.nan
                fd_dhdP_num[i, j] = np.nan

    # ---------- Compute percentage errors ----------
    # print(d_dh_table.shape, fd_dh_num.shape)
    # print(f"Derivatives from the table: {d2_dhdP_table}")
    # print(f"Derivatives from the interpolated values: {fd_dh_num}")
    rel_error_dh = np.abs((d_dh_table - fd_dh_num) / fd_dh_num) * 100
    # print(d_dP_table.shape, fd_dP_num.shape)
    rel_error_dP = np.abs((d_dP_table - fd_dP_num) / fd_dP_num) * 100
    # print(d2_dhdP_table.shape, fd_dhdP_num.shape)
    rel_error_dhdP = np.abs((d2_dhdP_table - fd_dhdP_num) / fd_dhdP_num) * 100

    # ---------- Print stats ----------
    def print_stats(label, arr):
        valid = ~np.isnan(arr)
        if np.any(valid):
            print(f"   → {label}: min={np.nanmin(arr):.3e}%, max={np.nanmax(arr):.3e}%, mean={np.nanmean(arr):.3e}%, std={np.nanstd(arr):.3e}%")
        else:
            print(f"   → {label}: All NaN")

    print_stats("Percentage error ∂f/∂h", rel_error_dh)
    print_stats("Percentage error ∂f/∂P", rel_error_dP)
    print_stats("Percentage error ∂²f/∂h∂P", rel_error_dhdP)

    # ---------- Contour plotting ----------
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals, indexing='ij')
    levels = np.linspace(0, 100, 21)

    for err, name in zip([rel_error_dh, rel_error_dP, rel_error_dhdP], ['∂f/∂h', '∂f/∂P', '∂²f/∂h∂P']):
        plt.figure(figsize=(8, 5))
        contour = plt.contourf(H_mesh, P_mesh / 1e5, err, levels=levels, cmap='viridis', extend='both')
        plt.colorbar(contour, label='Percentage error [%]')
        plt.xlabel('Enthalpy [J/kg]')
        plt.ylabel('Pressure [bar]')
        plt.title(f'Percentage error on numerical derivative {name} ({prop})')

        plt.tight_layout()
        plt.show()


#-----------------------------------------------------------------------
# %% Derivative accuracy verification with error stats

# Load the generated table
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']

properties_to_test = ['T', 'd', 's', 'mu', 'k']
properties = {
    'T': 'T',
    'd': 'D',
    's': 'S',
    'mu': 'V',
    'k': 'L',
}

fluid = 'CO2'
# Use relative step size for each point
# delta_h = 1e-5 * np.abs(h) if np.abs(h) > 1 else 1e-4
# delta_P = 1e-5 * np.abs(P) if np.abs(P) > 1 else 1e-4

delta_h = 1e2
delta_P = 1e-2

for prop in properties_to_test:
    print(f"\n Verifying gradients for property: {prop}")

    d_dh_table = interp_table[prop]['d_dh']
    d_dP_table = interp_table[prop]['d_dP']
    d2_dhdP_table = interp_table[prop]['d2_dhdP']

    Nh = len(h_vals)
    Np = len(P_vals)

    d_dh_fd = np.zeros_like(d_dh_table)
    d_dP_fd = np.zeros_like(d_dP_table)
    d2_dhdP_fd = np.zeros_like(d2_dhdP_table)

    symbol = properties[prop]

    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            try:
                f_p = cp.PropsSI(symbol, 'H', h + delta_h, 'P', P, fluid)
                f_m = cp.PropsSI(symbol, 'H', h - delta_h, 'P', P, fluid)
                d_dh_fd[i, j] = (f_p - f_m) / (2 * delta_h)

                f_p = cp.PropsSI(symbol, 'H', h, 'P', P + delta_P, fluid)
                f_m = cp.PropsSI(symbol, 'H', h, 'P', P - delta_P, fluid)
                d_dP_fd[i, j] = (f_p - f_m) / (2 * delta_P)

                f_pp = cp.PropsSI(symbol, 'H', h + delta_h, 'P', P + delta_P, fluid)
                f_pm = cp.PropsSI(symbol, 'H', h + delta_h, 'P', P - delta_P, fluid)
                f_mp = cp.PropsSI(symbol, 'H', h - delta_h, 'P', P + delta_P, fluid)
                f_mm = cp.PropsSI(symbol, 'H', h - delta_h, 'P', P - delta_P, fluid)
                d2_dhdP_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * delta_h * delta_P)

            except:
                d_dh_fd[i, j] = np.nan
                d_dP_fd[i, j] = np.nan
                d2_dhdP_fd[i, j] = np.nan

    # ---------- Relative error calculation ----------
    rel_error_dh = np.abs((d_dh_table - d_dh_fd) / d_dh_fd) * 100
    rel_error_dP = np.abs((d_dP_table - d_dP_fd) / d_dP_fd) * 100
    rel_error_dhdP = np.abs((d2_dhdP_table - d2_dhdP_fd) / d2_dhdP_fd) * 100

    # ---------- Print stats ----------
    def print_stats(label, arr):
        valid = ~np.isnan(arr)
        if np.any(valid):
            print(f"   → {label}: min={np.nanmin(arr):.3e}%, max={np.nanmax(arr):.3e}%, mean={np.nanmean(arr):.3e}%, std={np.nanstd(arr):.3e}%")
        else:
            print(f"   → {label}: All NaN")

    print_stats("Percentage error ∂f/∂h", rel_error_dh)
    print_stats("Percentage error ∂f/∂P", rel_error_dP)
    print_stats("Percentage error ∂²f/∂h∂P", rel_error_dhdP)

    # ---------- Contour plotting ----------
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals, indexing='ij')
    levels = np.linspace(0, 100, 21)

    for err, name in zip([rel_error_dh, rel_error_dP, rel_error_dhdP], ['∂f/∂h', '∂f/∂P', '∂²f/∂h∂P']):
        plt.figure(figsize=(8, 5))
        contour = plt.contourf(H_mesh, P_mesh / 1e5, err, levels=levels, cmap='viridis', extend='both')
        plt.colorbar(contour, label='Percentage error [%]')
        plt.xlabel('Enthalpy [J/kg]')
        plt.ylabel('Pressure [bar]')
        plt.title(f'Percentage error for {name} ({prop})')

        # Saturation curve overlay
        P_sats = np.linspace(P_vals[0], P_vals[-1], 200)
        h_l, h_v, P_sats_bar = [], [], []

        for P_sat in P_sats:
            try:
                hl = cp.PropsSI('H', 'P', P_sat, 'Q', 0, fluid)
                hv = cp.PropsSI('H', 'P', P_sat, 'Q', 1, fluid)
                h_l.append(hl)
                h_v.append(hv)
                P_sats_bar.append(P_sat / 1e5)
            except:
                pass

        plt.plot(h_l, P_sats_bar, 'w--', lw=1.5, label='Saturation liquid')
        plt.plot(h_v, P_sats_bar, 'w--', lw=1.5, label='Saturation vapor')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
#-----------------------------------------------------------------------

