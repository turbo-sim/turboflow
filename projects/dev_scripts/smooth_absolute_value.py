import os
import numpy as np
import matplotlib.pyplot as plt
import meanline_axial as ml

if __name__ == "__main__":

    # Example data
    x = np.linspace(-0.5, 0.5, 501)

    # Calculate softmin using various methods
    f_exact = np.abs(x)
    f_quadratic = ml.smooth_abs(x, method='quadratic', epsilon=1e-4)
    f_logarithmic = ml.smooth_abs(x, method='logarithmic', epsilon=1e-2)
    # f_softplus = ml.smooth_abs(x, method='softplus', epsilon=1e-3)
 
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
    # ax.plot(x, f_softplus, label='Softplus abs', linestyle=":")
    ax.legend(fontsize=10)
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)
    filename = os.path.join(fig_dir, "smooth_abs_function")
    fig.savefig(filename + ".png", bbox_inches="tight")
    fig.savefig(filename + ".svg", bbox_inches="tight")

    # Show figure
    plt.show()