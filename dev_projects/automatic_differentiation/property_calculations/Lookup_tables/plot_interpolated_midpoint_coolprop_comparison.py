import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import CoolProp.CoolProp as cp
from jax_bicubic_HEOS_interpolation_1 import bicubic_interpolant  # Update path if needed
from matplotlib.colors import LogNorm
import os

SAVE_FIGURES = True  # Set to True to save, False to show

if SAVE_FIGURES:
    fig_dir = "Verification_figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

# Load table
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']

Nh = len(h_vals)
Np = len(P_vals)
hmin, hmax = h_vals[0], h_vals[-1]
Lmin, Lmax = np.log(P_vals[0]), np.log(P_vals[-1])

properties_to_test = ['T', 'd', 's', 'mu', 'k']
properties = {
    'T': 'T',
    'd': 'D',
    's': 'S',
    'mu': 'V',
    'k': 'L',
}

fluid = 'CO2'

print("\nChecking midpoint interpolation accuracy")

for prop in properties_to_test:
    print(f"\n🔍 Testing midpoints for property: {prop}")

    symbol = properties[prop]
    abs_error_grid = np.zeros((Nh - 1, Np - 1))
    percent_error_grid = np.zeros((Nh - 1, Np - 1))

    for i in range(Nh - 1):
        for j in range(Np - 1):
            h_mid = 0.5 * (h_vals[i] + h_vals[i + 1])
            P_mid = 0.5 * (P_vals[j] + P_vals[j + 1])
            try:
                val_true = cp.PropsSI(symbol, 'H', h_mid, 'P', P_mid, fluid)
                val_interp = bicubic_interpolant(
                    h_mid, P_mid,
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

    # ---------- Print stats ----------
    valid_mask = ~np.isnan(abs_error_grid)
    print(f"   → Absolute error: min={np.nanmin(abs_error_grid):.3e}, max={np.nanmax(abs_error_grid):.3e}, mean={np.nanmean(abs_error_grid):.3e}, std={np.nanstd(abs_error_grid):.3e}")
    print(f"   → Percentage error: min={np.nanmin(percent_error_grid):.3e}%, max={np.nanmax(percent_error_grid):.3e}%, mean={np.nanmean(percent_error_grid):.3e}%, std={np.nanstd(percent_error_grid):.3e}%")

    # ---------- Plot percentage error contour ----------
    plt.figure(figsize=(8, 5))

    h_mid_vals = 0.5 * (h_vals[:-1] + h_vals[1:])
    P_mid_vals = 0.5 * (P_vals[:-1] + P_vals[1:])
    H_mesh, P_mesh = np.meshgrid(h_mid_vals, P_mid_vals, indexing='ij')

    levels = np.logspace(-10, 2, 13)
    
    contour = plt.contourf(
        H_mesh, 
        P_mesh / 1e5, 
        percent_error_grid, 
        levels=levels, 
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]), 
        cmap='viridis', 
        extend='both'
    )
 
    cbar = plt.colorbar(contour)
    cbar.set_ticks(levels)  # Set ticks to match contour levels
    cbar.set_ticklabels([f"$10^{{{int(np.log10(l))}}}$" for l in levels])  # Optional: cleaner log labels
    cbar.set_label('Percentage error [%] (order of magnitude)')

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

    if SAVE_FIGURES:
        filename = os.path.join(fig_dir, f"{prop}_inter_CP_midpoint_error_contour.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Figure saved to: {filename}")
    else:
        plt.show()

