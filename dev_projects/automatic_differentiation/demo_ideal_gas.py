
import jax.numpy as jnp
from jax import grad
import turboflow as tf

# Define the specific gas constant, for air R is around 287 J/(kg*K)
R = 287.0

# Define the ideal gas equation of state: p = rho * R * T
def ideal_gas_equation(rho, T):
    return rho * R * T

# Example input: density (rho) and temperature (T)
# rho_T = jnp.array([1.225, 300.0])  # rho = 1.225 kg/m^3, T = 300 K
rho, T = 1.225, 300.0

# Compute the partial derivatives
dp_drho = grad(ideal_gas_equation, argnums=0)  # Partial derivative with respect to rho
dp_dT = grad(ideal_gas_equation, argnums=1)    # Partial derivative with respect to T

# Calculate the partial derivatives for the given input
partial_rho = dp_drho(rho, T)
partial_T = dp_dT(rho, T)

# Now compare using turboflow's approx_gradient for finite differences
def wrapped_ideal_gas(rho_T):
    rho, T = rho_T  # rho and T are the inputs
    return ideal_gas_equation(rho, T)

gradient_approx = tf.approx_gradient(wrapped_ideal_gas, x=jnp.asarray([rho, T]), method="3-point")

# Print the results
print("Partial derivative wrt rho (AD):", partial_rho)
print("Partial derivative wrt rho (FD):", gradient_approx[0])
print("Partial derivative wrt T (AD):", partial_T)
print("Partial derivative wrt T (DF):", gradient_approx[1])