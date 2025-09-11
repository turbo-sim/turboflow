import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import CoolProp.CoolProp as cp

# Load property grids
with open("raw_property_grids.pkl", "rb") as f:
    raw_property_grids = pickle.load(f)

# Load table
with open("interpolation_table.pkl", "rb") as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']

properties_to_test = ['T', 'd', 's', 'mu', 'k']

fluid = 'CO2'

print("\nChecking interpolation table against raw property grids data")

for prop in properties_to_test:
    print(f"\n🔍 Property: {prop}")

    raw_vals = raw_property_grids[prop]
    table_vals = interp_table[prop]['value']

    abs_error = np.abs(table_vals - raw_vals)
    rel_error = np.abs((table_vals - raw_vals) / raw_vals)
    percent_error = rel_error * 100

    # Mask NaNs if needed
    valid_mask = ~np.isnan(raw_vals)
    abs_error = abs_error[valid_mask]
    rel_error = rel_error[valid_mask]

    print(f"   → Absolute error: max={np.nanmax(abs_error):.3e}, mean={np.nanmean(abs_error):.3e}")
    print(f"   → Relative error: max={np.nanmax(rel_error):.3e}, mean={np.nanmean(rel_error):.3e}")

    # --------- Contour plot ---------
    plt.figure(figsize=(8, 5))
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals, indexing='ij')
    levels = np.logspace(-12, 2, 15)  # percentage error levels

    contour = plt.contourf(H_mesh, P_mesh / 1e5, percent_error, levels=levels, cmap='viridis', extend='both')
    cbar = plt.colorbar(contour)
    cbar.formatter = ticker.LogFormatterMathtext(base=10)
    cbar.update_ticks()
    cbar.set_label('Percentage error [%] (order of magnitude)')

    plt.xlabel('Enthalpy [J/kg]')
    plt.ylabel('Pressure [bar]')
    plt.title(f'Percentage error contour for {prop}')

    # --------- Saturation curve overlay ---------
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
