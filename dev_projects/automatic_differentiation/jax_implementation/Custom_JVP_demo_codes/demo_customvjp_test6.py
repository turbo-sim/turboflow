import jax
import jax.numpy as jnp

# Function to encode string case into a numerical value
def encode_case(case):
    if case == "Example_Case1":
        return 0.0
    elif case == "Example_Case2":
        return 1.0
    else:
        raise ValueError(f"Unknown case: {case}")

class PolynomialModel:
    def __init__(self, case_numeric):
        self.case_numeric = case_numeric  # already numeric
    
    def poly_function(self, a, b, c, d, x, y):
        if self.case_numeric == 0.0:
            return {"T": a * x**3 + b * x * y + c * y**2 + d, 
                    "p": a * x**4 + b * x**2 * y + c * y**2 + d * y}
        elif self.case_numeric == 1.0:
            return {"T": a * x**4 + b * x**2 * y + c * y**2 + d * y, 
                    "p": a * x**3 + b * x * y + c * y**2 + d}
        else:
            raise ValueError("Invalid case_numeric")

# Function to initialize class and return the method
@jax.custom_vjp
def get_poly_function_numeric(case_numeric, a, b, c, d, x, y):
    model = PolynomialModel(case_numeric)
    return model.poly_function(a, b, c, d, x, y)

# Forward function
def fwd(case_numeric, a, b, c, d, x, y):
    return get_poly_function_numeric(case_numeric, a, b, c, d, x, y), (case_numeric, a, b, c, d, x, y)

# Backward function
def bwd(res, g):
    case_numeric, a, b, c, d, x, y = res
    delta = 1e-3
    grad_x_total, grad_y_total = 0.0, 0.0

    for key in g:
        df_dx = (get_poly_function_numeric(case_numeric, a, b, c, d, x + delta, y)[key] - get_poly_function_numeric(case_numeric, a, b, c, d, x, y)[key]) / delta
        df_dy = (get_poly_function_numeric(case_numeric, a, b, c, d, x, y + delta)[key] - get_poly_function_numeric(case_numeric, a, b, c, d, x, y)[key]) / delta
        grad_x_total += g[key] * df_dx
        grad_y_total += g[key] * df_dy

    return None, None, None, None, None, grad_x_total, grad_y_total

# Register custom differentiation
get_poly_function_numeric.defvjp(fwd, bwd)

# Test the logic
case = "Example_Case2"  # string input
case_numeric = encode_case(case)  # convert before calling JAX-traced function

grad_func = jax.jacrev(get_poly_function_numeric, argnums=(5, 6))
gradients = grad_func(case_numeric, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0)

print(f"Gradients: {gradients}")
