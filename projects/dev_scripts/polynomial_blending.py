
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml

# Define case parameters
radius_curvature = np.inf
pitch = 1.00
opening = 0.40
Ma_crit = 1.00
Ma_exit = np.linspace(0.00, 1.00, 200)


from scipy.linalg import solve

# def polynomial_blending(y, alpha):

#     mid = 0.3
#     A = np.asarray([[1, 1, 1], 
#                     [2, 3, 4],
#                     [mid**2, mid**3, mid**4]])

#     b = [1, 0, 0.5]

#     sol = solve(A, b)

#     sigma = sol[0]*y**2 + sol[1]*y**3 + sol[2]*y**4

#     return sigma


def polynomial_blending(x, x0):

    y = 0.5
    A = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [1, x0, x0**2, x0**3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, x0, x0**2, x0**3],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 2, 3],
                    [0, 1, 2*x0, 3*x0**2, 0, -1, -2*x0, -3*x0**2],
                    [0, 0, 2, 6*x0, 0, 0, -2, -6*x0]])

    b = [0, 0, y, y, 1.0, 0, 0, 0]

    sol = solve(A, b)

    sigma = (sol[0] + sol[1]*x + sol[2]*x**2 + sol[3]*x**3) * (x < x0) + \
    (sol[4] + sol[5]*x + sol[6]*x**2 + sol[7]*x**3) * (x > x0)

    cap = np.full_like(sigma, 1)
    # # print(cap)
    # # sigma = np.min([sigma, cap], axis=0)#, method="boltzmann", alpha=5, axis=0)
    # x = np.asarray([sigma, cap])
    # sigma = ml.smooth_min(x, axis=0, method="logsumexp", alpha=5)
    # print(sigma)

    return sigma


# def polynomial_blending6(x, x0):



#     y = 0.2
#     A = np.asarray([[1, 0, 0, 0, 0, 0, 0],
#                     [1, 1, 1, 1, 1, 1, 1],
#                     [1, x0, x0**2, x0**3, x0**4, x0**5, x0**6],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 1, 2, 3, 4, 5, 6],
#                     [0, 0, 2, 0, 0, 0, 0],
#                     [0, 0, 2, 6, 12, 20, 30]])

#     b = [0, 1, y, 0, 0, 0, 0]

#     sol = solve(A, b)

#     sigma = (sol[0] + sol[1]*x + sol[2]*x**2 + sol[3]*x**3 + sol[4]*x**4 + sol[5]*x**5 + sol[6]*x**6)

#     # cap = np.full_like(sigma, 1)
#     # # print(cap)
#     # # sigma = np.min([sigma, cap], axis=0)#, method="boltzmann", alpha=5, axis=0)
#     # x = np.asarray([sigma, cap])
#     # sigma = ml.smooth_min(x, axis=0, method="logsumexp", alpha=5)
#     # print(sigma)

#     return sigma

# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
# ax.set_title("Deviation model testing")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xscale("linear")
ax.set_yscale("linear")
# ax.set_ylim([0, 1])
x = np.linspace(0, 1, 100)
sigma = polynomial_blending(x, 0.75)
ax.plot(x, sigma, linewidth=1.25)
# ax.plot(Ma_exit, beta_aungier, linewidth=1.25, label="Aungier")
# ax.plot(Ma_exit, beta_metal, linewidth=1.25, label="Metal")
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
plt.show()
