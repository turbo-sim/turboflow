import jax
import jax.numpy as jnp

# Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

# Define the custom JVP using finite differences
@jax.custom_jvp
def rosenbrock_custom_jvp(x, y):
    return rosenbrock(x, y)

@rosenbrock_custom_jvp.defjvp
def rosenbrock_custom_jvp_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents  

    delta = 1e-3  # Small perturbation for finite difference

    # Finite difference approximation for gradients
    df_dx = (rosenbrock(x + delta, y) - rosenbrock(x, y)) / delta
    df_dy = (rosenbrock(x, y + delta) - rosenbrock(x, y)) / delta

    # Compute JVP (directional derivative)
    jvp = df_dx * x_dot + df_dy * y_dot
    
    return rosenbrock(x, y), jvp

# Compute gradients using jax.grad
grad_rosenbrock_custom = jax.grad(rosenbrock_custom_jvp, argnums=(0, 1))  # Custom JVP
grad_rosenbrock_auto = jax.grad(rosenbrock, argnums=(0, 1))  # Pure JAX automatic differentiation

# Gradient Descent Optimization Function
def gradient_descent(grad_fn, method_name, learning_rate=0.001, num_iterations=10000):
    x = jnp.array(-1.5)
    y = jnp.array(1.5)
    
    print(f"\nStarting {method_name} Optimization:")

    for i in range(num_iterations):
        # Compute gradients
        dx, dy = grad_fn(x, y)
        
        # Update step
        x -= learning_rate * dx
        y -= learning_rate * dy

        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i}: x={x:.4f}, y={y:.4f}, f(x,y)={rosenbrock(x, y):.6f}")

    print(f"\nFinal {method_name} Optimized Values:")
    print(f"x = {x:.4f}, y = {y:.4f}, f(x, y) = {rosenbrock(x, y):.6f}\n")
    return x, y

# Run both optimizations
if __name__ == "__main__":
    # Custom JVP (finite difference method)
    opt_x1, opt_y1 = gradient_descent(grad_rosenbrock_custom, "Custom JVP")

    # JAX Automatic Differentiation
    opt_x2, opt_y2 = gradient_descent(grad_rosenbrock_auto, "JAX Autodiff")



