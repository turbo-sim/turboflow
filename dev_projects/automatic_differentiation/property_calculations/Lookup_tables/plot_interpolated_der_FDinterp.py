import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import CoolProp.CoolProp as cp
from jax_bicubic_HEOS_interpolation_1 import bicubic_interpolant  # <-- adjust to your module name
import jax.numpy as jnp

# ------------------- Load interpolation table -------------------
with open('interpolation_table.pkl', 'rb') as f:
    interp_table = pickle.load(f)

h_vals = interp_table['h']
P_vals = interp_table['P']
L = jnp.log(P_vals)
Pmin=20*1e5 # units - Pa

Nh = len(h_vals)
Np = len(P_vals)
hmin, hmax = h_vals[0], h_vals[-1]
Lmin, Lmax = np.log(P_vals[0]), np.log(P_vals[-1])
fluid = 'CO2'
deltah = h_vals[1]-h_vals[0]
deltaL = L[1]-L[0]

properties_to_test = ['T', 'd', 's', 'mu', 'k']
properties = {
    'T': 'T',
    'd': 'D',
    's': 'S',
    'mu': 'V',
    'k': 'L',
}

# ------------------- Define finite diff function -------------------
def finite_diff(f_p, f_m, delta):
    return (f_p - f_m) / (2 * delta)

# ------------------- Loop over each property -------------------
for prop in properties_to_test:
    print(f"\n🔬 Verifying numerical derivatives on interpolated values: {prop}")

    d_dh_table = interp_table[prop]['d_dh']
    d_dP_table = interp_table[prop]['d_dP']
    d2_dhdP_table = interp_table[prop]['d2_dhdP']

    fd_dh_num = np.zeros_like(d_dh_table)
    fd_dP_num = np.zeros_like(d_dP_table)
    fd_dhdP_num = np.zeros_like(d2_dhdP_table)

    coeffs = interp_table[prop]['coeffs']

    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            eps_h =  0.01*deltah
            eps_P =  1e-6*Pmin


            # Evaluate function at shifted points
            f_hp = bicubic_interpolant(h + eps_h, (P + eps_P), h_vals, jnp.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            f_h  = bicubic_interpolant(h + eps_h, P, h_vals, jnp.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            f_p  = bicubic_interpolant(h, (P + eps_P), h_vals, jnp.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            f_0  = bicubic_interpolant(h, P, h_vals, jnp.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)

            # First derivatives
            fd_dh_num[i, j] = (f_h - f_0) / eps_h
            fd_dP_num[i, j] = (f_p - f_0) / eps_P

            # Mixed derivative
            fd_dhdP_num[i, j] = (f_hp - f_h - f_p + f_0) / (eps_h * eps_P)

            # try:
            #     # ∂f/∂h
            #     f_p_h = bicubic_interpolant(h + delta_h, np.log(P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     f_m_h = bicubic_interpolant(h - delta_h, np.log(P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     fd_dh_num[i, j] = finite_diff(f_p_h, f_m_h, delta_h)

            #     # ∂f/∂P
            #     f_p_P = bicubic_interpolant(h, np.log(P + delta_P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     f_m_P = bicubic_interpolant(h, np.log(P - delta_P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     fd_dP_num[i, j] = finite_diff(f_p_P, f_m_P, delta_P)

            #     # ∂²f/∂h∂P
            #     f_pp = bicubic_interpolant(h + delta_h, np.log(P + delta_P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     f_pm = bicubic_interpolant(h + delta_h, np.log(P - delta_P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     f_mp = bicubic_interpolant(h - delta_h, np.log(P + delta_P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     f_mm = bicubic_interpolant(h - delta_h, np.log(P - delta_P), h_vals, np.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax)
            #     fd_dhdP_num[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * delta_h * delta_P)

            # except:
            #     fd_dh_num[i, j] = np.nan
            #     fd_dP_num[i, j] = np.nan
            #     fd_dhdP_num[i, j] = np.nan

    # ---------- Percentage errors ----------
    rel_error_dh = np.abs((d_dh_table - fd_dh_num) / fd_dh_num) * 100
    rel_error_dP = np.abs((d_dP_table - fd_dP_num) / fd_dP_num) * 100
    rel_error_dhdP = np.abs((d2_dhdP_table - fd_dhdP_num) / fd_dhdP_num) * 100

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
    levels = np.logspace(-12, 2, 15)

    for err, name in zip([rel_error_dh, rel_error_dP, rel_error_dhdP], ['∂f/∂h', '∂f/∂P', '∂²f/∂h∂P']):
        plt.figure(figsize=(8, 5))
        contour = plt.contourf(H_mesh, P_mesh / 1e5, err, levels=levels, cmap='viridis', extend='both')

        cbar = plt.colorbar(contour)
        cbar.set_label('Percentage error [%]')
        cbar.formatter = ticker.LogFormatterExponent(base=10)
        cbar.update_ticks()

        plt.xlabel('Enthalpy [J/kg]')
        plt.ylabel('Pressure [bar]')
        plt.title(f'Percentage error on numerical derivative {name} ({prop})')

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
