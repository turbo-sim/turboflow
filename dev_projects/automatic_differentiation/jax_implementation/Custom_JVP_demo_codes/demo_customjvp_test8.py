import jax
import jax.numpy as jnp
from functools import partial

# Class with method using two configuration parameters
class MyMathExtended:
    def __init__(self):
        self.repo = {
            (0.0, 0.0): lambda x, y: x**3 + y**2,
            (0.0, 1.0): lambda x, y: x**2 + y**3,
            (1.0, 0.0): lambda x, y: jnp.sin(x) + jnp.cos(y),
            (1.0, 1.0): lambda x, y: jnp.exp(x * y),
        }

    def compute(self, case1, case2, x, y):
        func = self.repo.get((case1, case2))
        if func is None:
            raise ValueError(f"Invalid case: {(case1, case2)}")
        return func(x, y)

# Outer function using custom_jvp with TWO nondiff arguments
@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def compute_wrapper(case1, case2, x, y):
    obj = MyMathExtended()
    return obj.compute(case1, case2, x, y)

# JVP rule for the wrapper
@compute_wrapper.defjvp
def compute_wrapper_jvp(case1, case2, primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents

    obj = MyMathExtended()
    f = lambda a, b: obj.compute(case1, case2, a, b)

    delta = 1e-3
    df_dx = (f(x + delta, y) - f(x, y)) / delta
    df_dy = (f(x, y + delta) - f(x, y)) / delta

    jvp = df_dx * x_dot + df_dy * y_dot
    return f(x, y), jvp

# Encode functions
def encode_case(c):
    return {"First": 0.0, "Second": 1.0}[c]

# Test the function
case1 = encode_case("First")
case2 = encode_case("Second")

x_val = jnp.array(1.0)
y_val = jnp.array(2.0)

# Get gradients w.r.t. x and y
grad_func = jax.grad(compute_wrapper, argnums=(2, 3))
grad_x, grad_y = grad_func(case1, case2, x_val, y_val)

print("grad_x:", grad_x)
print("grad_y:", grad_y)
