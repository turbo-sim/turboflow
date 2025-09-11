# ================== plot_property_surfaces.py ==================

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import pickle
import jax.numpy as jnp

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
# ----------------- Global font settings -----------------
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

# ----------------- Plot function -----------------
def plot_property_surface(property_name, h_vals, P_vals_log, property_grid):
    """
    Plots a 3D surface of a thermodynamic property.

    Parameters
    ----------
    property_name : str
        Name for axis labels and title.
    h_vals : array-like
        Enthalpy values (J/kg).
    P_vals_log : array-like
        Log of pressure values if needed.
    property_grid : 2D array
        Property values on the grid.
    """
    H, P = np.meshgrid(h_vals, np.exp(P_vals_log), indexing='ij')  # Convert log(P) if needed

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(H, P / 1e5, property_grid, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Enthalpy [J/kg]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_zlabel(property_name)
    ax.set_title(f'{property_name} Surface', pad=20)

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(property_grid)
    fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label=property_name)

    # Add a legend proxy
    legend_proxy = [Line2D([0], [0], marker='s', color='w', label='Values',
                           markerfacecolor='green', markersize=10)]
    ax.legend(handles=legend_proxy, loc='upper right')

    plt.tight_layout()
    plt.show()

# ----------------- Example usage for raw grids -----------------
# If raw_property_grids is already in memory, comment these lines
with open('raw_property_grids.pkl', 'rb') as f:
    raw_property_grids = pickle.load(f)

# Define your h_vals and P_vals_log arrays as needed
# Example (replace with your actual arrays if needed):
# h_vals = np.linspace(200000, 600000, 50)
# P_vals_log = np.linspace(np.log(20e5), np.log(200e5), 50)

# plot_property_surface('Temperature [K]', h_vals, P_vals_log, raw_property_grids['T'])
# plot_property_surface('Viscosity [Pa·s]', h_vals, P_vals_log, raw_property_grids['mu'])

# ----------------- Example usage for interpolated table -----------------
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

plot_property_surface('Interpolated Temperature [K]', interp_table['h'], np.log(interp_table['P']), interp_table['T']['value'])
plot_property_surface('Interpolated Viscosity [Pa·s]', interp_table['h'], np.log(interp_table['P']), interp_table['mu']['value'])

# =========================================================
