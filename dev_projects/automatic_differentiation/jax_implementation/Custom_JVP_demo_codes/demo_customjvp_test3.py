
import jax
import jax.numpy as jnp
from functools import partial

# Define the function repository for each case
function_repository = {
    0.0: lambda x, y: x**3 + 2*x*y + y**2,  # "First_Case"
    1.0: lambda x, y: x**2 + 2*x*y + y**3   # "Second_Case"
}

def encode_case(case):
    if case == "First_Case":
        return 0.0
    elif case == "Second_Case":
        return 1.0
    else:
        raise ValueError(f"Unknown case: {case}")

def inner_func(case_numeric, x, y):
    case_func = function_repository.get(case_numeric)
    if case_func is None:
        raise ValueError(f"Unknown case: {case_numeric}")
    return case_func(x, y)

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def func_custom_jvp(case_numeric, x, y):
    return inner_func(case_numeric, x, y)

# Correct: include case_numeric in the defjvp
@func_custom_jvp.defjvp
def func_custom_jvp_jvp(case_numeric, primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents

    delta = 1e-3

    df_dx = (inner_func(case_numeric, x + delta, y) - inner_func(case_numeric, x, y)) / delta
    df_dy = (inner_func(case_numeric, x, y + delta) - inner_func(case_numeric, x, y)) / delta

    # Compute the JVP (directional derivative) ignoring case_numeric
    jvp = df_dx * x_dot + df_dy * y_dot
    
    return inner_func(case_numeric, x, y), jvp




# Testing the implementation

case = "First_Case"
case_numeric = encode_case(case)

x_val = jnp.array(1.0)
y_val = jnp.array(5.0)

# Compute gradient of the output w.r.t x and y
grad_func = jax.grad(func_custom_jvp, argnums=(1, 2))  # x is arg 1, y is arg 2
grad_x, grad_y = grad_func(case_numeric, x_val, y_val)

print("grad_x:", grad_x)
print("grad_y:", grad_y)


