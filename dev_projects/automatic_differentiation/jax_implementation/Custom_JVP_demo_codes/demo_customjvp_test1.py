import jax
import jax.numpy as jnp

# # Define a function of two variables: f(x, y) = x^2 + 2xy + y^2
# def func(x, y):
#     return x**2 + 2*x*y + y**3

# # Define the custom JVP for the function
# @jax.custom_jvp
# def func_custom_jvp(x, y):
#     return func(x, y)

# # Define the forward-mode JVP function
# @func_custom_jvp.defjvp
# def func_custom_jvp_jvp(primals, tangents):
#     x, y = primals
#     x_dot, y_dot = tangents  # Directional derivatives
    
#     # Small step for finite difference
#     delta = 1e-3

#     # Compute finite difference approximations for partial derivatives
#     df_dx = (func(x + delta, y) - func(x, y)) / delta
#     df_dy = (func(x, y + delta) - func(x, y)) / delta

#     # Compute the JVP (directional derivative)
#     jvp = df_dx * x_dot + df_dy * y_dot
    
#     return func(x, y), jvp

# # Compute gradient using jax.grad (which will use our custom JVP)
# grad_func = jax.grad(func_custom_jvp, argnums=(0, 1))  # Gradient w.r.t x and y

# # Gradient Descent Optimization
# def gradient_descent(learning_rate=0.01, num_iterations=100):
#     # Initialize variables (random start point)
#     x = jnp.array(1.0)
#     y = jnp.array(5.0)
    
#     for i in range(num_iterations):
#         # Compute gradients
#         dx, dy = grad_func(x, y)
        
#         # Update step
#         x -= learning_rate * dx
#         y -= learning_rate * dy

#         # Print progress
#         if i % 10 == 0:
#             print(f"Iteration {i}: x={x:.4f}, y={y:.4f}, f(x,y)={func(x, y):.4f}")

#     return x, y

# # Run optimization
# if __name__ == "__main__":
#     opt_x, opt_y = gradient_descent()

#     # Print final results
#     print("\nOptimized Values:")
#     print(f"x = {opt_x}, y = {opt_y}, f(x, y) = {func(opt_x, opt_y)}")

# # Test the function with forward-mode differentiation
# x_value = jnp.array(1.0)
# y_value = jnp.array(5.0)

# # List of input variables
# input_vars = ["x", "y"]
# values = (x_value, y_value)

# # Dictionary to store partial derivatives
# partial_derivatives = {}

# # Loop over each input variable
# for i, var in enumerate(input_vars):
#     # Create tangent vector with 1.0 for the current variable, 0.0 for others
#     tangents = [0.0] * len(input_vars)
#     tangents[i] = 1.0  # Set current variable's tangent to 1.0

#     # Compute JVP to get partial derivative with respect to the current variable
#     _, derivs = jax.jvp(func_custom_jvp, values, tuple(tangents))


#     # Store the results in dictionary with proper keys
#     partial_derivatives[f"d_f/d_{var}"] = derivs

# # Print the dictionary of partial derivatives
# print("Partial Derivatives:")
# for key, value in partial_derivatives.items():
#     print(f"{key}: {value}")


import jax
import jax.numpy as jnp

# Define the function repository for each case
function_repository = {
    0.0: lambda x, y: x**3 + 2*x*y + y**2,  # "First_Case"
    1.0: lambda x, y: x**2 + 2*x*y + y**3  # "Second_Case"
}

# Function to encode the case string to a numeric value
def encode_case(case):
    if case == "First_Case":
        return 0.0  # Encode as 0.0
    elif case == "Second_Case":
        return 1.0  # Encode as 1.0
    else:
        raise ValueError(f"Unknown case: {case}")

# Define the func that takes the encoded case (numeric value) and x, y values
def func(case_numeric, x, y):
    # Extract the corresponding function from the repository
    case_func = function_repository.get(case_numeric)
    
    if case_func is None:
        raise ValueError(f"Unknown case: {case_numeric}")
    
    # Return the result of the function evaluation for the selected case
    return case_func(x, y)

# Define the custom JVP for the function
@jax.custom_jvp
def func_custom_jvp(x, y, case_numeric):
    return func(case_numeric, x, y)

# Define the forward-mode JVP function
@func_custom_jvp.defjvp
def func_custom_jvp_jvp(primals, tangents):
    x, y, case_numeric = primals
    x_dot, y_dot, _ = tangents  # Directional derivatives for x and y, but case_numeric is constant

    # Small step for finite difference
    delta = 1e-3

    # Compute finite difference approximations for partial derivatives (ignoring case_numeric)
    df_dx = (func(case_numeric, x + delta, y) - func(case_numeric, x, y)) / delta
    df_dy = (func(case_numeric, x, y + delta) - func(case_numeric, x, y)) / delta

    # Compute the JVP (directional derivative) but ignore the tangent for case_numeric
    jvp = df_dx * x_dot + df_dy * y_dot
    
    return func(case_numeric, x, y), jvp

# Compute gradient using jax.grad (we will pass the encoded case as input)
# grad_func_first_case = jax.grad(lambda x, y: func_custom_jvp(x, y, 0.0), argnums=(0, 1))  # Gradient w.r.t x and y for "First_Case"
# grad_func_second_case = jax.grad(lambda x, y: func_custom_jvp(x, y, 1.0), argnums=(0, 1))  # Gradient w.r.t x and y for "Second_Case"

# grad_func = jax.grad(lambda x, y, case_numeric: func_custom_jvp(x, y, case_numeric), argnums=(0, 1))  # Gradient w.r.t x and y
grad_func = jax.grad(func_custom_jvp, argnums=(0,1)) 


# Gradient Descent Optimization
def gradient_descent(case, learning_rate=0.01, num_iterations=100):
    # Encode the case string as numeric value
    case_numeric = encode_case(case)
    
    # Initialize variables (random start point)
    x = jnp.array(1.0)
    y = jnp.array(5.0)

    # Select the gradient function based on the encoded case
    # if case_numeric == 0.0:
    #     grad_func = grad_func_first_case
    # elif case_numeric == 1.0:
    #     grad_func = grad_func_second_case
    # else:
    #     raise ValueError(f"Unknown case: {case_numeric}")
    
    for i in range(num_iterations):
        # Compute gradients
        # dx, dy = grad_func(x, y)  # Call func_custom_jvp with the fixed 'case'
        dx, dy = grad_func(x, y, case_numeric)  # Call func_custom_jvp with the fixed 'case_numeric'
        
        # Update step
        x -= learning_rate * dx
        y -= learning_rate * dy

        # Print progress
        if i % 10 == 0:
            print(f"Iteration {i}: x={x:.4f}, y={y:.4f}, f(x,y)={func(case_numeric, x, y):.4f}")

    return x, y

# Run optimization for a specific case
if __name__ == "__main__":
    case = "Second_Case"  # Change this to "Second_Case" to test the other case
    opt_x, opt_y = gradient_descent(case)

    # Print final results
    print("\nOptimized Values:")
    print(f"x = {opt_x}, y = {opt_y}, f(x, y) = {func(encode_case(case), opt_x, opt_y)}")




