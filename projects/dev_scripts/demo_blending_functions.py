import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import turbo_flow as tf


def sigmoid_hyperbolic(x, x0=0.5, alpha=1):
    """
    Compute the sigmoid hyperbolic function.

    This function calculates a sigmoid function based on the hyperbolic tangent.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{1 + \tanh\left(\frac{x - x_0}{\alpha}\right)}{2}

    Parameters
    ----------
    x : array_like
        Input data.
    x0 : float, optional
        Center of the sigmoid function. Default is 0.5.
    alpha : float, optional
        Scale parameter. Default is 0.1.

    Returns
    -------
    array_like
        Output of the sigmoid hyperbolic function.

    """
    sigma = (1 + np.tanh((x-x0)/alpha))/2
    return sigma


def sigmoid_rational(x, n, m):
    """
    Compute the sigmoid rational function.

    This function calculates a sigmoid function using an algebraic approach based on a rational function.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{x^n}{x^n + (1-x)^m}

    Parameters
    ----------
    x : array_like
        Input data.
    n : int
        Power of the numerator.
    m : int
        Power of the denominator.

    Returns
    -------
    array_like
        Output of the sigmoid algebraic function.

    """
    sigma = x ** n / (x ** n + (1 - x) ** m)
    return sigma


def sigmoid_smoothstep(x):
    """
    Compute the smooth step function.

    This function calculates a smooth step function with zero first-order 
    derivatives at the endpoints. More information available at:
    https://resources.wolframcloud.com/FunctionRepository/resources/SmoothStep/ 

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smoothstep function.

    """
    return x ** 2 * (3 - 2 * x)


def sigmoid_smootherstep(x):
    """
    Compute the smoother step function.

    This function calculates a smoother step function with zero second-order 
    derivatives at the endpoints. It is a modification of the smoothstep 
    function to provide smoother transitions.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smootherstep function.

    """
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3






# Define sampling vector
x = np.linspace(0, 1, 500)

# Comparison of methods
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Comparison of sigmoid blending functions")
ax.set_xlabel(r"$x$ value")
ax.set_ylabel(r"$y$ value")
ax.set_xscale("linear")
ax.set_yscale("linear")

sigma = sigmoid_rational(x, n=3, m=3)
ax.plot(x, sigma, linewidth=1.25, label="Rational function")

sigma = sigmoid_smoothstep(x)
ax.plot(x, sigma, linewidth=1.25, label="Smooth step function")

sigma = sigmoid_smootherstep(x)
ax.plot(x, sigma, linewidth=1.25, label="Smoother step function")

sigma = sigmoid_hyperbolic(x, alpha=0.1)
ax.plot(x, sigma, linewidth=1.25, label="Hyperbolic tangent function")

leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)
# plt.show()


# Hyperbolic tangent function
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Comparison of hyperbolic sigmoid functions")
ax.set_xlabel(r"$x$ value")
ax.set_ylabel(r"$y$ value")
ax.set_xscale("linear")
ax.set_yscale("linear")
for i, alpha in enumerate([0.01, 0.05, 0.1, 0.2, 0.3]):
    sigma = sigmoid_hyperbolic(x, x0=0.5, alpha=alpha)
    ax.plot(x, sigma, linewidth=1.25, label=rf"$\alpha={alpha:0.2f}$")
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)

# Plot results
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_title("Comparison of rational sigmoid functions")
ax.set_xlabel(r"$x$ value")
ax.set_ylabel(r"$y$ value")
ax.set_xscale("linear")
ax.set_yscale("linear")
linestyles = ['-', '--', ':']
colors = tf.utils.COLORS_MATLAB
for i, n in enumerate([2, 3, 4]):
    for j, m in enumerate([2, 3, 4]):
        sigma = sigmoid_rational(x, n=n, m=m)
        ax.plot(x, sigma, linewidth=1.25, label=rf"$n={n}, m={m}$", color=colors[i], linestyle=linestyles[j])

# ax.plot(Ma_exit, beta_aungier, linewidth=1.25, label="Aungier")
# ax.plot(Ma_exit, beta_metal, linewidth=1.25, label="Metal")
leg = ax.legend(loc="best")
fig.tight_layout(pad=1, w_pad=None, h_pad=None)

plt.show()
