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
from jax.sharding import Mesh, PartitionSpec, NamedSharding

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

#%% Grid and Fluid definition

# Defining the Grid
N=50       # Grid size in h direction
M=50      # Grid size in log(P) direction
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

#%% Properties to be calculated

properties = {
    'T': 'T',       # Temperature [K]
    'D': 'D',       # Density [kg/m³]
    'S': 'S',       # Entropy [J/kg/K]
    'mu': 'V',      # Viscosity [Pa·s]
    'k': 'L',       # Thermal conductivity [W/m/K]
}

# Create the grid of (h, P)
h_vals = jnp.linspace(hmin, hmax, N)
P_vals = jnp.linspace(Lmin, Lmax, M)
h_grid, P_grid = jnp.meshgrid(h_vals, jnp.exp(P_vals), indexing='ij')

# Evaluate each property on the grid using CoolProp
raw_property_grids = {}

print("⚙️ Generating raw property data using CoolProp...")
for key, coolprop_symbol in properties.items():
    prop_grid = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            h = h_vals[i]
            P = jnp.exp(P_vals[j])
            try:
                val = cp.PropsSI(coolprop_symbol, 'H', h, 'P', P, name)
            except:
                val = np.nan  # or use extrapolation/default value
            prop_grid[i, j] = val
    raw_property_grids[key] = prop_grid

print("✅ Raw property grids created.")

## Check raw_property_grids here to see if the values are proper or not ##

print("🔍 Checking raw_property_grids for NaNs and basic stats...\n")

for key, grid in raw_property_grids.items():
    num_nans = np.isnan(grid).sum()
    num_zeros = np.sum(grid == 0)
    min_val = np.nanmin(grid)
    max_val = np.nanmax(grid)
    mean_val = np.nanmean(grid)
    total_vals = grid.size

    print(f"📦 Property: {key}")
    print(f"   → Shape        : {grid.shape}")
    print(f"   → NaN count    : {num_nans}")
    print(f"   → Zero count   : {num_zeros} ({num_zeros / total_vals * 100:.2f}%)")
    print(f"   → Min, Max     : {min_val:.3e}, {max_val:.3e}")
    print(f"   → Mean (valid) : {mean_val:.3e}")
    print("-" * 50)

#%% Fixing the zero coeffs values at the edges

# Function to fix zero coefficients
def fix_zero_coeffs(coeffs):
    Nh, Np, *_ = coeffs.shape
    fixed_coeffs = coeffs.copy()

    for i in range(Nh):
        for j in range(Np):
            if jnp.all(fixed_coeffs[i, j] == 0):
                neighbor_coeffs = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < Nh) and (0 <= nj < Np) and not (di == 0 and dj == 0):
                            neighbor = fixed_coeffs[ni, nj]
                            if not jnp.all(neighbor == 0):
                                neighbor_coeffs.append(neighbor)

                if neighbor_coeffs:
                    fixed_coeffs = fixed_coeffs.at[i, j].set(jnp.mean(jnp.stack(neighbor_coeffs), axis=0))

    return fixed_coeffs

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
        fx = jnp.gradient(raw_grid, 1e-5, axis=0)  
        fy = jnp.gradient(raw_grid, 1e-5, axis=1)   # ∂f/∂P
        fxy = jnp.gradient(fx, 1e-5, axis=1)        # ∂²f/∂x∂y = ∂/∂y (∂f/∂x)

        for i in range(1, Nh - 2):
            for j in range(1, Np - 2):
                # Extract 4x4 stencils of values and derivatives ## Change this!!
                f = raw_grid[i-1:i+3, j-1:j+3]
                fx_stencil = fx[i-1:i+3, j-1:j+3]
                fy_stencil = fy[i-1:i+3, j-1:j+3]
                fxy_stencil = fxy[i-1:i+3, j-1:j+3]

                # Compute coefficients for cell (i, j)
                temp_coeffs = compute_bicubic_coefficients_of_ij(1, 1, f, fx_stencil, fy_stencil, fxy_stencil)

                # Store the computed coefficients in the `coeffs` array ## Check Coeffs ##
                coeffs = coeffs.at[i, j].set(temp_coeffs)  

        # Fix zero coefficients (using the function above)
        coeffs = fix_zero_coeffs(coeffs) 

        # for i, hi in enumerate(h_vals):
        #     for j, Pj in enumerate(P_vals):
        #         # Extract scalar values at the current grid point
        #         f_val = raw_grid
        #         fx_val = fx
        #         fy_val = fy
        #         fxy_val = fxy

        #         # Compute bicubic coefficients from point-wise values
        #         temp_coeffs = compute_bicubic_coefficients_of_ij(i, j, f_val, fx_val, fy_val, fxy_val)

        #         # Store the computed coefficients
        #         coeffs = coeffs.at[i, j,:].set(temp_coeffs)

        print("\n🔎 Inspecting all edge coefficients...")

        Nh, Np = coeffs.shape[:2]
        zero_count = 0
        edge_total = 0

        # Top and Bottom rows
        for i in [0, Nh - 1]:
            for j in range(Np):
                coeff_val = coeffs[i, j]
                is_zero = jnp.all(coeff_val == 0).item()
                print(f"Edge ({i},{j}) → All zero? {is_zero} | Coeffs: {coeff_val}")
                zero_count += is_zero
                edge_total += 1

        # Left and Right columns (excluding corners to avoid double-counting)
        for i in range(1, Nh - 1):
            for j in [0, Np - 1]:
                coeff_val = coeffs[i, j]
                is_zero = jnp.all(coeff_val == 0).item()
                print(f"Edge ({i},{j}) → All zero? {is_zero} | Coeffs: {coeff_val}")
                zero_count += is_zero
                edge_total += 1

        print(f"\n✅ Total edge nodes checked: {edge_total}")
        print(f"❌ Zero-valued coefficient nodes on edges: {zero_count}")

        print("🧪 Coefficient grid sanity check:")
        print("   → Total cells        :", coeffs.shape[0] * coeffs.shape[1])
        print("   → Non-zero coeff cells:",
        jnp.count_nonzero(jnp.linalg.norm(coeffs, axis=-1) > 0))

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
        }

    with open(save_path, 'wb') as f:
        pickle.dump(table, f)

    print(f"\n✅ Interpolation table saved to: {os.path.abspath(save_path)}")

    save_interpolation_table_as_csv(table, 'property_table.csv')
    # or
    save_interpolation_table_as_parquet(table, 'property_table.parquet')

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
    print(f"📁 CSV property table saved to: {os.path.abspath(filename)}")



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
    print(f"📦 Property table saved as: {filename}")

#%% Call interpolation table generator

generate_interpolation_table(h_vals, P_vals, raw_property_grids, save_path='interpolation_table.pkl')


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


#%% Comparing to Interpolated data

with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

# Plot interpolated Temperature
plot_property_surface('Interpolated Temperature [K]', interp_table['h'], np.log(interp_table['P']), interp_table['T']['value'])

# Plot interpolated Viscosity
plot_property_surface('Interpolated Viscosity [Pa·s]', interp_table['h'], np.log(interp_table['P']), interp_table['mu']['value'])

# %%
