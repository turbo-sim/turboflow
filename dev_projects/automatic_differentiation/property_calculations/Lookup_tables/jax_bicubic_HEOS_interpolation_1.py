# ======================== Imports ========================
import os
import psutil
import time
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import CoolProp.CoolProp as cp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# ======================== Config ========================
NCORES = psutil.cpu_count(logical=False)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NCORES}"
jax.config.update("jax_enable_x64", True)

# Global precision
float64 = jnp.dtype("float64")
complex128 = jnp.dtype("complex128")

# =================== Functions to Export ===================
@jax.jit
def compute_bicubic_coefficients_of_ij(i, j, f, fx, fy, fxy):
    #xx=f(0,0)&f(1,0)&f(0,1)&f(1,1)&f_x(0,0)&f_x(1,0)&f_x(0,1)&f_x(1,1)&f_y(0,0)&f_y(1,0)&f_y(0,1)&f_y(1,1)&f_{xy}(0,0)&f_{xy}(1,0)&f_{xy}(0,1)&f_{xy}(1,1)
    xx=[f[i,j],f[i+1,j],f[i,j+1],f[i+1,j+1],fx[i,j],fx[i+1,j],fx[i,j+1],fx[i+1,j+1],fy[i,j],fy[i+1,j],fy[i,j+1],fy[i+1,j+1],fxy[i,j],fxy[i+1,j],fxy[i,j+1],fxy[i+1,j+1]]
    A=[ [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ -3., 3., 0., 0., -2., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ 2., -2., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., -3., 3., 0., 0., -2., -1., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 2., -2., 0., 0., 1., 1., 0., 0 ],
        [ -3., 0., 3., 0., 0., 0., 0., 0., -2., 0., -1., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., -3., 0., 3., 0., 0., 0., 0., 0., -2., 0., -1., 0 ],
        [ 9., -9., -9., 9., 6., 3., -6., -3., 6., -6., 3., -3., 4., 2., 2., 1 ],
        [ -6., 6., 6., -6., -3., -3., 3., 3., -4., 4., -2., 2., -2., -2., -1., -1 ],
        [ 2., 0., -2., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 2., 0., -2., 0., 0., 0., 0., 0., 1., 0., 1., 0 ],
        [ -6., 6., 6., -6., -4., -2., 4., 2., -3., 3., -3., 3., -2., -1., -2., -1 ],
        [ 4., -4., -4., 4., 2., 2., -2., -2., 2., -2., 2., -2., 1., 1., 1., 1]]
    return jnp.matmul(jnp.array(A,dtype=f.dtype),jnp.array(xx,dtype=f.dtype))


# def bicubic_interpolant(h_vals, P_vals, coeffs, hmin, hmax, Lmin, Lmax, Nh, Np):
#     """
#     Create a bicubic interpolant function using precomputed coefficients.
#     """
#     # Ensure that coeffs is in the right shape (Nh, Np, 16)
#     assert coeffs.shape == (Nh, Np, 16), f"Expected coeffs to have shape (Nh, Np, 16), but got {coeffs.shape}"
    
#     # Normalized grid values
#     def normalize(value, min_val, max_val):
#         return (value - min_val) / (max_val - min_val)

#     # The actual interpolant function, it takes h and P as arguments
#     @jit
#     def interpolant_fn(h, P):
#         # Ensure h_vals and P_vals are 1D arrays
#         h_vals_flat = jnp.ravel(h_vals)  # Flatten the h_vals to 1D
#         P_vals_flat = jnp.ravel(P_vals)  # Flatten the P_vals to 1D
        
#         # Normalize h and P
#         norm_h = normalize(h, hmin, hmax)
#         norm_P = normalize(P, Lmin, Lmax)

#         # Identify the grid points surrounding h, P
#         i = jnp.clip(jnp.searchsorted(h_vals_flat, h) - 1, 0, Nh - 2)
#         j = jnp.clip(jnp.searchsorted(P_vals_flat, P) - 1, 0, Np - 2)

#         # Extract the coefficients for the surrounding grid points
#         coeff = coeffs[i, j]

#         # Interpolate in both directions
#         h_diff = norm_h - h_vals_flat[i]
#         P_diff = norm_P - P_vals_flat[j]

#         # Calculate the interpolant using the bicubic coefficients
#         result = (
#             coeff[0] + coeff[1] * h_diff + coeff[2] * P_diff + coeff[3] * h_diff * P_diff +
#             coeff[4] * h_diff**2 + coeff[5] * h_diff * P_diff**2 +
#             coeff[6] * P_diff**2 + coeff[7] * h_diff**2 * P_diff + 
#             coeff[8] * h_diff**3 + coeff[9] * h_diff**2 * P_diff +
#             coeff[10] * h_diff * P_diff**2 + coeff[11] * h_diff**3 * P_diff + 
#             coeff[12] * P_diff**3 + coeff[13] * h_diff * P_diff**3 +
#             coeff[14] * h_diff**3 * P_diff**2 + coeff[15] * h_diff**2 * P_diff**3
#         )

#         return result

#     return interpolant_fn

# @partial(jit, static_argnums=(5, 6))  # Nh and Np are static # static arguments are all except h, P
# def bicubic_interpolant(h, P, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax):
#     """
#     Evaluate the bicubic interpolant at (h, P) using precomputed coefficients.
#     """
#     # Normalize
#     norm_h = (h - hmin) / (hmax - hmin)
#     norm_P = (P - Lmin) / (Lmax - Lmin)

#     # Flatten grid arrays
#     h_vals_flat = jnp.ravel(h_vals)
#     P_vals_flat = jnp.ravel(P_vals)

#     # Find surrounding grid indices
#     i = jnp.clip(jnp.searchsorted(h_vals_flat, h) - 1, 0, Nh - 2)
#     j = jnp.clip(jnp.searchsorted(P_vals_flat, P) - 1, 0, Np - 2)

#     # Relative differences in normalized space
#     h_base = (h_vals_flat[i] - hmin) / (hmax - hmin)
#     P_base = (P_vals_flat[j] - Lmin) / (Lmax - Lmin)
#     h_diff = norm_h - h_base
#     P_diff = norm_P - P_base

#     # Fetch bicubic coefficients
#     coeff = coeffs[i, j]

#     # Evaluate bicubic polynomial
#     result = (
#         coeff[0] + coeff[1] * h_diff + coeff[2] * P_diff + coeff[3] * h_diff * P_diff +
#         coeff[4] * h_diff**2 + coeff[5] * h_diff * P_diff**2 +
#         coeff[6] * P_diff**2 + coeff[7] * h_diff**2 * P_diff +
#         coeff[8] * h_diff**3 + coeff[9] * h_diff**2 * P_diff +
#         coeff[10] * h_diff * P_diff**2 + coeff[11] * h_diff**3 * P_diff +
#         coeff[12] * P_diff**3 + coeff[13] * h_diff * P_diff**3 +
#         coeff[14] * h_diff**3 * P_diff**2 + coeff[15] * h_diff**2 * P_diff**3
#     )

#     return result

@partial(jit, static_argnums=(5, 6))
def bicubic_interpolant(h, P, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax):
    """
    Evaluate the bicubic interpolant at (h, P) using precomputed coefficients.
    """
    # Log-transform P
    L = jnp.log(P)

    # Normalize positions to [0, 1] cell coordinates
    ii = ((h - hmin) / (hmax - hmin) * (Nh - 1))
    i = ii.astype(int)
    x = ii - i

    jj = ((L - Lmin) / (Lmax - Lmin) * (Np - 1))
    j = jj.astype(int)
    y = jj - j


    # Evaluate bicubic polynomial
    result = jnp.zeros_like(h)  # use h shape
    x_pow = jnp.ones_like(h)    # x^0

    for m in range(4):  # m = x power
        y_pow = jnp.ones_like(h)  # y^0 initially
        for n in range(4):  # n = y power
            c = coeffs[i, j, 4 * n + m]
            result += c * x_pow * y_pow
            y_pow = y_pow * y
        x_pow = x_pow * x


    return result



@jax.jit
def inverse_interpolant_scalar_hD(h, D):
    #Find the real(float) index
    ii=((h-hmin)/(hmax-hmin)*(N-1))
    #The integer part is the cell index
    i=ii.astype(int)
    #The remainder (for numerical stability better to use the difference)
    #is instead the position within our interpolation cell.
    x=ii-i
    #find interval that contains the solution
    xth=jnp.ones_like(h) #initialize x to the 0th power
    #First we compute the nodal values, that is the values of D(h,P) where
    #h is the actual enthalpy and P are grid values.
    #TODO: instead of computing all the nodal values and then use sortedsearch
    #to find the correct interval, we could do a binary search. This would
    #constraint M to be a power of 2.
    #Possible example (to be refined) to compute the node. Start with the node
    #corresponding to j=M/2, then compute new index j=j+M/4*(2*(Dj>D)-1)
    #then j=j+M/8*(2*(Dj>D)-1) and so on ...
    #after log2(M) iteration we converged to the index j.
    D_nodal=jnp.zeros(M)
    for m in range(4):
        D_nodal+=bicubic_coefficients[i,:,m]*xth
        xth=xth*x
    #We search more efficiently in which interval we have the solution
    #if we assume a sorted vector.
    #TODO: This assumes that P has a monotonic trend with respect to D
    #at fixed h. This causes some problems and needs further investigation
    if iD==cp.iSmass:
        j=jax.numpy.searchsorted(-D_nodal,-D).astype(int)-1
    else:
        j=jax.numpy.searchsorted(D_nodal,D).astype(int)-1

    #After we are in the unit square, that is for known i and j
    #compute 1D cubic coefficients (as complex numbers to avoid promotion)
    #Each coefficient is bj=sum(aij*x**i)
    #Leading to the equation D=b0 + b1*y + b2*y**2 + b3*y**3
    xth=jnp.ones_like(h)
    b0 =jnp.zeros_like(h,dtype=complex128)
    b1 =jnp.zeros_like(h,dtype=complex128)
    b2 =jnp.zeros_like(h,dtype=complex128)
    b3 =jnp.zeros_like(h,dtype=complex128)
    for m in range(4):
        b0 +=bicubic_coefficients[i,j,4*0+m]*xth
        b1 +=bicubic_coefficients[i,j,4*1+m]*xth
        b2 +=bicubic_coefficients[i,j,4*2+m]*xth
        b3 +=bicubic_coefficients[i,j,4*3+m]*xth
        xth=xth*x
    #solve cubic equation - all three solutions
    #TODO: if necessary, add solution for degenerate (quadratic and linear)
    #For more information:https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    D0=b2*b2-3*b3*b1
    D1=2*b2*b2*b2-9*b3*b2*b1+27*b3*b3*(b0-D)
    C=((D1+(D1*D1-4*D0*D0*D0)**0.5)/2)**(1/3)
    D0C=jax.lax.select(C==(0+0j),0+0j,D0/C)
    z=jnp.array([1,-0.5+0.8660254037844386j,-0.5-0.8660254037844386j])
    y=-1/(3*b3)*(b2+C*z+D0C/z)
    #To find our solution we have two criteria:
    #   -0 imaginary part
    #   -real part between 0 and 1, that are the bounds of our cell
    # We define a "badness" as the deviation from these critera, and pick the
    # solution with the lowest badness
    badness=jax.nn.relu(4*(jnp.real(y)-0.5)**2-1)+jnp.imag(y)**2
    yreal=jnp.real(y[jnp.argmin(badness)])
    jj=j+yreal
    L=Lmin+jj*(Lmax-Lmin)/(M-1)
    P=jnp.exp(L)
    return P

@jax.jit
def inverse_interpolant_scalar_DP(D, P):
    # Convert pressure to log space
    L = jnp.log(P)
    
    # Compute index along pressure grid
    jj = ((L - Lmin) / (Lmax - Lmin) * (M - 1))
    j = jj.astype(int)
    y = jj - j  # fractional position in pressure direction

    # Compute nodal D(h) values at fixed pressure (we'll search h index now)
    yth = jnp.ones_like(D)
    D_nodal = jnp.zeros(N)
    for m in range(4):
        D_nodal += bicubic_coefficients[:, j, m] * yth
        yth = yth * y

    # Search h-direction to find which cell to use
    if iD == cp.iSmass:
        i = jnp.searchsorted(-D_nodal, -D).astype(int) - 1
    else:
        i = jnp.searchsorted(D_nodal, D).astype(int) - 1

    # Now build 1D cubic in x (h-direction) at fixed j
    yth = jnp.ones_like(D)
    b0 = jnp.zeros_like(D, dtype=complex128)
    b1 = jnp.zeros_like(D, dtype=complex128)
    b2 = jnp.zeros_like(D, dtype=complex128)
    b3 = jnp.zeros_like(D, dtype=complex128)
    for m in range(4):
        b0 += bicubic_coefficients[i, j, m + 4*0] * yth
        b1 += bicubic_coefficients[i, j, m + 4*1] * yth
        b2 += bicubic_coefficients[i, j, m + 4*2] * yth
        b3 += bicubic_coefficients[i, j, m + 4*3] * yth
        yth = yth * y

    # Solve cubic: D = b0 + b1*x + b2*x^2 + b3*x^3
    D0 = b2*b2 - 3*b3*b1
    D1 = 2*b2**3 - 9*b3*b2*b1 + 27*b3**2*(b0 - D)
    C = ((D1 + jnp.sqrt(D1**2 - 4*D0**3)) / 2)**(1/3)
    D0C = jax.lax.select(C == 0, 0 + 0j, D0 / C)
    z = jnp.array([1, -0.5 + 0.8660254037844386j, -0.5 - 0.8660254037844386j])
    x = -1/(3*b3)*(b2 + C*z + D0C/z)

    # Pick root with lowest badness
    badness = jax.nn.relu(4*(jnp.real(x)-0.5)**2 - 1) + jnp.imag(x)**2
    xreal = jnp.real(x[jnp.argmin(badness)])

    # Final result: compute h from i + x
    ii = i + xreal
    h = hmin + ii * (hmax - hmin) / (N - 1)
    return h
