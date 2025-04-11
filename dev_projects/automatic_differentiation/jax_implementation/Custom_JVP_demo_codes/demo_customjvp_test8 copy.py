import jax
import jax.numpy as jnp
from functools import partial

import turboflow as tf

# Class with method using two configuration parameters
class MyMathExtended:
    def __init__(self, case1, case2):
        self.case1 = case1
        self.case2 = case2
        # self.repo = {
        #     (0.0, 0.0): lambda x, y: x**3 + y**2,
        #     (0.0, 1.0): lambda x, y: x**2 + y**3,
        #     (1.0, 0.0): lambda x, y: jnp.sin(x) + jnp.cos(y),
        #     (1.0, 1.0): lambda x, y: jnp.exp(x * y),
        # }

    def compute(self, x, y):
        if self.case1 == "First" and self.case2 == "First":
            return x**3 + y**2
        elif self.case1 == "First" and self.case2 == "Second":
            return x**2 + y**3
        elif self.case1 == "Second" and self.case2 == "First":
            return jnp.sin(x) + jnp.cos(y)
        elif self.case1 == "Second" and self.case2 == "Second":
            return jnp.exp(x * y)
        else:
            raise ValueError(f"Invalid case inputs: {self.case1} {self.case2}")

# Outer function using custom_jvp with TWO nondiff arguments
@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def compute_wrapper(case1, case2, x, y):
    obj = MyMathExtended(case1, case2)
    return obj.compute(x, y)

# JVP rule for the wrapper
@compute_wrapper.defjvp
def compute_wrapper_jvp(case1, case2, primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents

    obj = MyMathExtended(case1, case2)
    f = lambda a, b: obj.compute(a, b)

    delta = 1e-3
    df_dx = (f(x + delta, y) - f(x, y)) / delta
    df_dy = (f(x, y + delta) - f(x, y)) / delta

    jvp = df_dx * x_dot + df_dy * y_dot
    return f(x, y), jvp

# Encode functions
# def encode_case(c):
#     return {"First": 0.0, "Second": 1.0}[c]

# Test the function
case1 = "First"
case2 = "First"

x_val = jnp.array(1.0)
y_val = jnp.array(2.0)

# Get gradients w.r.t. x and y
grad_func = jax.jacfwd(compute_wrapper, argnums=(2, 3))
grad_x, grad_y = grad_func(case1, case2, x_val, y_val)

print("grad_x:", grad_x)
print("grad_y:", grad_y)
