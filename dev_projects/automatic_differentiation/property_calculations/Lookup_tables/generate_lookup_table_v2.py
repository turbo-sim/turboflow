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
N=120       # Grid size in h direction
M=120      # Grid size in log(P) direction
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
f=cp.AbstractState("HEOS",name)

#-----------------------------------------------------------------------
#%% Properties to be calculated

properties = {
    'T': 'T',       # Temperature [K]
    'd': 'D',       # Density [kg/m³]
    's': 'S',       # Entropy [J/kg/K]
    'mu': 'V',      # Viscosity [Pa·s]
    'k': 'L',       # Thermal conductivity [W/m/K]
}

prop_keys = {
    'T': cp.iT,
    'd': cp.iDmass,
    's': cp.iSmass,
    # 'mu': cp.iViscosity,
    # 'k': cp.iConductivity,
}



# Create the grid of (h, P)
h_vals = jnp.linspace(hmin, hmax, N, dtype=float64)
P_vals = jnp.linspace(Lmin, Lmax, M, dtype=float64)    # Logarithmic scale for P
# h_grid, P_grid = jnp.meshgrid(h_vals, jnp.exp(P_vals), indexing='ij')
P_grid, h_grid = jnp.meshgrid(jnp.exp(P_vals), h_vals)
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
                # val = cp.PropsSI(key, 'H', h, 'P', P, name)
                val = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h, P)
            except:
                val = np.nan  # or use extrapolation/default value
            prop_grid[i, j] = val[key]
    raw_property_grids[key] = prop_grid

print(" Raw property grids created.")

# Save raw_property_grids to pickle
with open('raw_property_grids.pkl', 'wb') as f:
    pickle.dump(raw_property_grids, f)

#-----------------------------------------------------------------------
#%% Property function from fluid properties
# Wrapper: takes h and log(P), returns the property
def func(h, logP):
    P = jnp.exp(logP)  # Convert log(P) back to actual P
    return tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h, P)

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
        coeffs = jnp.zeros((Nh, Np, 16), dtype=jnp.float64)
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
                eps_h =  0.01*deltah
                eps_P =  1e-6*Pmin
                
                # First-order derivatives 
                # df_dh = jax.jacrev(tf.get_props_custom_jvp, argnums=(2))(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))[prop]
                # df_dP = jax.jacrev(tf.get_props_custom_jvp, argnums=(3))(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))[prop]

                
                
                # Mixed second derivative ∂²f/∂h∂logP
                # d2f_dhdlogP = jax.jacrev(jax.jacrev(tf.get_props_custom_jvp, argnums=(2)), argnums=(3))(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))
                # mixed = d2f_dhdlogP(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))

                # # Evaluate df/dh at logP + eps
                # df_dh_plus = jax.jacrev(tf.get_props_custom_jvp, argnums=2)(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val + eps_P))

                # # Evaluate df/dh at logP - eps
                # df_dh_minus = jax.jacrev(tf.get_props_custom_jvp, argnums=2)(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val - eps_P))

                # # Approximate mixed derivative
                # d2f_dhdP = (df_dh_plus[prop] - df_dh_minus[prop]) / (2 * eps_P)


                ###### Using Finite Differences for calulating the derivatives ######
                df_dh = (tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, (h_val + eps_h), jnp.exp(logP_val))[prop] - 
                         tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, (h_val), jnp.exp(logP_val))[prop])/eps_h
                
                df_dP = (tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val) + eps_P)[prop] - 
                         tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))[prop])/eps_P

                f_hp = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h_val + eps_h, jnp.exp(logP_val) + eps_P)
                f_h  = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h_val + eps_h, jnp.exp(logP_val))
                f_p  = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val) + eps_P)
                f_0  = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h_val, jnp.exp(logP_val))

                d2f_dhdP = (f_hp[prop] - f_h[prop] - f_p[prop] + f_0[prop]) / (eps_h * eps_P)

                ###### CoolProp derivatives ######
                # df_dh = 0.0
                # df_dP = 0.0
                # d2f_dhdP = 0.0

                # key = prop_keys[prop]
                # try:
                #     f.update(cp.HmassP_INPUTS,h_val,jnp.exp(logP_val))
                #     df_dP=f.first_two_phase_deriv(key,cp.iP,cp.iHmass) # dDdP at constant h
                #     df_dh=f.first_two_phase_deriv(key,cp.iHmass,cp.iP)
                #     d2f_dhdP=f.second_two_phase_deriv(key,cp.iP,cp.iHmass,cp.iHmass,cp.iP)
                #     print(df_dh, df_dP, d2f_dhdP )
                # except:
                #     pass
                
                
                # Taking time here
                fx = fx.at[i, j].set(df_dh)
                fy = fy.at[i, j].set(df_dP*jnp.exp(logP_val))
                fxy = fxy.at[i, j].set(d2f_dhdP*jnp.exp(logP_val))
                
        for i, hi in enumerate(h_vals):
            for j, Pj in enumerate(P_vals):
                # Extract scalar values at the current grid point
                temp_coeffs = compute_bicubic_coefficients_of_ij(i, j, raw_grid, fx*deltah, 
                                                                 fy*deltaL, 
                                                                 fxy*deltah*deltaL)
                # Store the computed coefficients
                coeffs = coeffs.at[i, j, :].set(temp_coeffs)

        # print(coeffs)
        # print(coeffs.shape)

        # print(
        #     f"h_vals (shape: {h_vals.shape}): {h_vals}\n"
        #     f"P_vals (shape: {P_vals.shape}, log(P)): {P_vals}\n"
        #     f"Nh (number of h points): {Nh}\n"
        #     f"Np (number of P points): {Np}\n"
        #     f"hmin: {hmin:.3e}, hmax: {hmax:.3e}\n"
        #     f"Lmin (log(Pmin)): {Lmin:.3e}, Lmax (log(Pmax)): {Lmax:.3e}\n"
        # )

        # Now coeffs is a 3D array with the correct shape (Nh, Np, 16) ready for interpolation
        # Loop through the grid to perform the interpolation
        for i, hi in enumerate(h_vals):
            if i % (Nh / 10) < 1:  # Example progress print every 10% progress
                print(f'Progress: {i / Nh * 100:.2f}% done in {time.time() - t0:.2f}s')
            
            for j, Pj in enumerate(P_vals):
                # Extract the relevant coefficients for interpolation
                # print(Pj)
                val = bicubic_interpolant(hi, jnp.exp(Pj), h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
                # true_val = cp.PropsSI('T', 'H', hi, 'P', jnp.exp(Pj), "CO2")
                # print(
                #         f"h = {hi:.2f} J/kg, P = {jnp.exp(Pj):.2f} Pa | "
                #         f"Interpolated T = {val:.6f} K, "
                #         f"True T = {true_val:.6f} K, "
                #         f"Diff = {true_val - val:.6e} K")

                
                dval_dh = jax.grad(bicubic_interpolant, argnums=0)(hi, jnp.exp(Pj), h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
                dval_dP = jax.grad(bicubic_interpolant, argnums=1)(hi, jnp.exp(Pj), h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
                d2val_dhdP = jax.grad(jax.grad(bicubic_interpolant, argnums=1), argnums=0)(hi, jnp.exp(Pj), h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
 
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

    save_interpolation_table_as_csv(table, f'property_table_{N}.csv')
    # or
    save_interpolation_table_as_parquet(table, f'property_table_{N}.parquet')

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
