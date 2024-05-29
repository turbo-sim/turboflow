import os
import numpy as np
import matplotlib.pyplot as plt
import turbo_flow as tf

if __name__ == "__main__":

    # Example data
    x = np.linspace(-0.1, 0.1, 2001)

    # Calculate softmin using various methods
    f_exact = np.abs(x)
    f_quadratic = tf.smooth_abs(x, method='quadratic', epsilon=1e-4)
    f_logarithmic = tf.smooth_abs(x, method='logarithmic', epsilon=1e-3)
    f_hyperbolic = tf.smooth_abs(x, method='hyperbolic', epsilon=1e-3)
 
    # Create the folder to save figures
    fig_dir = "figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Plot the data
    fig, ax = plt.subplots()
    ax.set_xlabel('Exit Mach number')
    ax.set_ylabel('Throat Mach number')
    ax.plot(x, f_exact, label='Exact abs', color='black')
    ax.plot(x, f_quadratic, label='Quadratic abs')
    ax.plot(x, f_logarithmic, label='Logarithmic abs')
    ax.plot(x, f_hyperbolic, label='Hyperbolic abs')
    ax.legend(fontsize=10)
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)
    filename = os.path.join(fig_dir, "smooth_abs_function")
    fig.savefig(filename + ".png", bbox_inches="tight")
    fig.savefig(filename + ".svg", bbox_inches="tight")

    # Show figure
    plt.show()