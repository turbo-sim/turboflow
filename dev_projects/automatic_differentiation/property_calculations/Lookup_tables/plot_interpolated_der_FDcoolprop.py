import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import CoolProp.CoolProp as cp
import jax.numpy as jnp
from matplotlib.colors import LogNorm

# ------------------- Load interpolation table -------------------
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']
L = jnp.log(P_vals)
Pmin=20*1e5 # units - Pa

Nh = len(h_vals)
Np = len(P_vals)

properties_to_test = ['T', 'd', 's', 'mu', 'k']
properties = {
    'T': 'T',
    'd': 'D',
    's': 'S',
    'mu': 'V',
    'k': 'L',
}
deltah = h_vals[1]-h_vals[0]
deltaL = L[1]-L[0]
fluid = 'CO2'

delta_h =  0.01*deltah
delta_P =  1e-6*Pmin

# ------------------- Loop over each property -------------------
for prop in properties_to_test:
    print(f"\n🔬 Verifying gradients for property: {prop}")

    d_dh_table = interp_table[prop]['d_dh']
    d_dP_table = interp_table[prop]['d_dP']
    d2_dhdP_table = interp_table[prop]['d2_dhdP']

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

    # ---------- Percentage errors ----------
    rel_error_dh = np.abs((d_dh_table - d_dh_fd) / d_dh_fd) * 100
    rel_error_dP = np.abs((d_dP_table - d_dP_fd) / d_dP_fd) * 100
    rel_error_dhdP = np.abs((d2_dhdP_table - d2_dhdP_fd) / d2_dhdP_fd) * 100

    def print_stats(label, arr):
        valid = ~np.isnan(arr)
        if np.any(valid):
            print(f"   → {label}: min={np.nanmin(arr):.3e}%, max={np.nanmax(arr):.3e}%, mean={np.nanmean(arr):.3e}%, std={np.nanstd(arr):.3e}%")
        else:
            print(f"   → {label}: All NaN")

    print_stats("Percentage error ∂f/∂h", rel_error_dh)
    print_stats("Percentage error ∂f/∂P", rel_error_dP)
    print_stats("Percentage error ∂²f/∂h∂P", rel_error_dhdP)

    # ---------- Contour plots ----------
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals, indexing='ij')
    levels = np.logspace(-10, 2, 13)

    for err, name in zip([rel_error_dh, rel_error_dP, rel_error_dhdP], ['∂f/∂h', '∂f/∂P', '∂²f/∂h∂P']):
        plt.figure(figsize=(8, 5))
        contour = plt.contourf(H_mesh, P_mesh / 1e5, err, levels=levels,norm=LogNorm(vmin=levels[0], vmax=levels[-1]), cmap='viridis', extend='both')

        cbar = plt.colorbar(contour)
        # cbar.set_label('Percentage error [%]')
        # cbar.formatter = ticker.LogFormatterExponent(base=10)
        # cbar.update_ticks()

        cbar.set_ticks(levels)  # Set ticks to match contour levels
        cbar.set_ticklabels([f"$10^{{{int(np.log10(l))}}}$" for l in levels])  # Optional: cleaner log labels
        cbar.set_label('Percentage error [%] (order of magnitude)')

        plt.xlabel('Enthalpy [J/kg]')
        plt.ylabel('Pressure [bar]')
        plt.title(f'Percentage error vs CoolProp FD {name} ({prop})')

        # ---------- Saturation curve ----------
        P_sats = np.logspace(np.log10(P_vals[0]), np.log10(P_vals[-1]), 200)
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