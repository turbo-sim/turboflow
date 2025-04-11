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
        return {"T": a * x**3 + b * x * y + c * y**2 + d, 
                "p": a * x**4 + b * x**2 * y + c * y**2 + d * y}  # First case
    elif case_numeric == 1.0:
        return {"T": a * x**4 + b * x**2 * y + c * y**2 + d * y, 
                "p": a * x**3 + b * x * y + c * y**2 + d}  # Second case
    else:
        raise ValueError("Invalid case_numeric")

# Forward pass
def fwd(case_numeric, a, b, c, d, x, y):
    return poly_function(case_numeric, a, b, c, d, x, y), (case_numeric, a, b, c, d, x, y)

# Backward pass (custom gradient)
def bwd(res, g):
    case_numeric, a, b, c, d, x, y = res
    delta = 1e-3  # Small step for finite difference

    # Initialize gradient storage
    grad_x_total = 0.0
    grad_y_total = 0.0

    # Iterate over the dictionary keys in the function output
    for key in g:  # g is a dictionary
        # Compute partial derivatives for each key with respect to x and y
        df_dx = (poly_function(case_numeric, a, b, c, d, x + delta, y)[key] - poly_function(case_numeric, a, b, c, d, x, y)[key]) / delta
        df_dy = (poly_function(case_numeric, a, b, c, d, x, y + delta)[key] - poly_function(case_numeric, a, b, c, d, x, y)[key]) / delta

        # Apply the gradient to the partial derivatives for this key
        grad_x_total += g[key] * df_dx
        grad_y_total += g[key] * df_dy

    # Return total gradients w.r.t. x and y
    return None, None, None, None, None, grad_x_total, grad_y_total



# Register custom differentiation   
poly_function.defvjp(fwd, bwd)    

# Test the gradient computation for one case
case = "Example_Case2"  # Choose only one case
case_numeric = encode_case(case)

# Compute gradient w.r.t x and y for both 'T' and 'p'
grad_func = jax.jacrev(poly_function, argnums=(5, 6))
gradients = grad_func(case_numeric, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0)

print(f"Gradients for {case}: {gradients}")
