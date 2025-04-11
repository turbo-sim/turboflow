
import jax
import jax.numpy as jnp
from functools import partial

# Define the class
class MyMath:
    def __init__(self):
        self.function_repository = {
            0.0: lambda x, y: x**3 + 2*x*y + y**2,
            1.0: lambda x, y: x**2 + 2*x*y + y**3
        }

    def compute(self, case_numeric, x, y):
        func = self.function_repository.get(case_numeric)
        if func is None:
            raise ValueError(f"Invalid case_numeric: {case_numeric}")
        return func(x, y)

# Define the outer function that creates the object and calls the method
@partial(jax.custom_jvp, nondiff_argnums=(0,))
def compute_wrapper(case_numeric, x, y):
    obj = MyMath()
    return obj.compute(case_numeric, x, y)

# Define the custom JVP rule
@compute_wrapper.defjvp
def compute_wrapper_jvp(case_numeric, primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    obj = MyMath()

    delta = 1e-3
    f = obj.compute

    df_dx = (f(case_numeric, x + delta, y) - f(case_numeric, x, y)) / delta
    df_dy = (f(case_numeric, x, y + delta) - f(case_numeric, x, y)) / delta

    jvp = df_dx * x_dot + df_dy * y_dot

    return f(case_numeric, x, y), jvp

# Test the setup
def encode_case(case):
    return {"First_Case": 0.0, "Second_Case": 1.0}[case]

case = "First_Case"
case_numeric = encode_case(case)
x_val = jnp.array(1.0)
y_val = jnp.array(5.0)

# Compute gradients
grad_func = jax.grad(compute_wrapper, argnums=(1, 2))  # x and y
grad_x, grad_y = grad_func(case_numeric, x_val, y_val)

print("grad_x:", grad_x)
print("grad_y:", grad_y)



