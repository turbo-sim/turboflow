import turboflow as tf
import jax
import jax.numpy as jnp
from scipy.optimize import rosen_der
from jax import grad, jit

# Define the Rosenbrock function in N dimensions
def rosenbrock_function(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Example input: a vector of N elements
x = jnp.array([1.0, 1.5, 2.0, 2.5])

# Compute the gradient of the Rosenbrock function
rosenbrock_grad = jax.grad(rosenbrock_function)

# Calculate the gradient for the given input
gradient = rosenbrock_grad(x)

gradient_approx = tf.approx_gradient(rosenbrock_function, x, method="3-point", abs_step=1e-4)


# Print the result
print("Rosenbrock gradient (analytic):", rosen_der(x))
print("Rosenbrock gradient (AD):", gradient)
print("Rosenbrock gradient (FD):", gradient_approx)