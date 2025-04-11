# import jax
# import jax.numpy as jnp  # Import JAX's NumPy for automatic differentiation

# # Define a function with multiple variables: x and y
# @jax.custom_vjp  # Mark this function for custom gradients
# def poly_function(a, b, c, d, x, y):
#     """
#     This function computes:
#         f(x, y) = a * x^3 + b * x * y + c * y^2 + d
#     where a, b, c, and d are constants.
#     """
#     return a * x**3 + b * x * y + c * y**2 + d

# # Define the forward pass for custom differentiation
# def fwd(a, b, c, d, x, y):
#     """
#     Forward pass: Compute the function value and save necessary variables
#     for use in the backward pass.
#     """
#     return poly_function(a, b, c, d, x, y), (a, b, c, d, x, y)  # Save inputs

# # Define the backward pass (custom gradient computation)
# def bwd(res, g):
#     """
#     Backward pass: Compute the gradients using finite difference approximation.
#     """
#     a, b, c, d, x, y = res  # Retrieve saved inputs
#     delta = 1e-3  # Small step size for finite difference

#     # Compute approximate partial derivatives using finite differences
#     df_dx = (poly_function(a, b, c, d, x + delta, y) - poly_function(a, b, c, d, x, y)) / delta
#     df_dy = (poly_function(a, b, c, d, x, y + delta) - poly_function(a, b, c, d, x, y)) / delta

#     # Return None for constants (a, b, c, d) and gradients for (x, y)
#     return None, None, None, None, g * df_dx, g * df_dy  

# # Register the custom forward and backward passes
# poly_function.defvjp(fwd, bwd)

# # Test the gradient computation
# grad_func = jax.grad(poly_function, argnums=(4, 5))  # Compute gradients w.r.t x and y

# # Evaluate the gradient at a specific point
# print("Gradients:", grad_func(2.0, 3.0, 4.0, 5.0, 1.0, 2.0))

################################### Another Example ##############################

import jax
import jax.numpy as jnp

# Function to encode string case into a numerical value
def encode_case(case):
    if case == "Example_Case1":
        return 0.0  # First case
    elif case == "Example_Case2":
        return 1.0  # Second case
    else:
        raise ValueError(f"Unknown case: {case}")

@jax.custom_vjp
def poly_function(case_numeric, a, b, c, d, x, y):
    """
    Compute a polynomial function with different behavior for different cases:
    
    Case 1 (0.0): f(x, y) = a * x^3 + b * x * y + c * y^2 + d
    Case 2 (1.0): f(x, y) = a * x^4 + b * x^2 * y + c * y^2 + d * y
    """
    if case_numeric == 0.0:
        return a * x**3 + b * x * y + c * y**2 + d  # First case
    elif case_numeric == 1.0:
        return a * x**4 + b * x**2 * y + c * y**2 + d * y  # Second case
    else:
        raise ValueError("Invalid case_numeric")

# Forward pass
def fwd(case_numeric, a, b, c, d, x, y):
    return poly_function(case_numeric, a, b, c, d, x, y), (case_numeric, a, b, c, d, x, y)

# Backward pass (custom gradient)
def bwd(res, g):
    case_numeric, a, b, c, d, x, y = res
    delta = 1e-3  # Small step for finite difference

    # Compute gradients using finite differences
    df_dx = (poly_function(case_numeric, a, b, c, d, x + delta, y) - poly_function(case_numeric, a, b, c, d, x, y)) / delta
    df_dy = (poly_function(case_numeric, a, b, c, d, x, y + delta) - poly_function(case_numeric, a, b, c, d, x, y)) / delta

    # Return None for case_numeric and constants (a, b, c, d), gradients for (x, y)
    return None, None, None, None, None, g * df_dx, g * df_dy  

# Register custom differentiation
poly_function.defvjp(fwd, bwd)

# Test the gradient computation for one case
case = "Example_Case2"  # Choose only one case
case_numeric = encode_case(case)

# Compute gradient w.r.t x and y
grad_func = jax.grad(poly_function, argnums=(5, 6))
gradients = grad_func(case_numeric, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0)

print(f"Gradients for {case}: {gradients}")
