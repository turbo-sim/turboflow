import os
import numpy as np
import matplotlib.pyplot as plt
import turboflow as tf

if __name__ == "__main__":

    fig_dir = "figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Example data
    Ma_crit = 1
    x = np.linspace(0.5, 1.5, 51)

    # Calculate softmin using various methods
    f_max = np.minimum(x, Ma_crit)
    f_logsumexp = tf.smooth_minimum(x, Ma_crit, method="logsumexp", alpha=25)
    f_boltzmann = tf.smooth_minimum(x, Ma_crit, method="boltzmann", alpha=25)
    f_pnorm = tf.smooth_minimum(x, Ma_crit, method="p-norm", alpha=25)

    # Plot the data
    fig, ax = plt.subplots()
    ax.set_xlabel("Exit Mach number")
    ax.set_ylabel("Throat Mach number")
    ax.plot(x, f_max, label="Exact maximum", color="black")
    ax.plot(x, f_boltzmann, label="Boltzmann maximum")
    ax.plot(x, f_logsumexp, label="LogSumExp maximum")
    ax.plot(x, f_pnorm, label="p-norm maximum", linestyle=":")
    ax.legend(fontsize=10)
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)
    filename = os.path.join(fig_dir, "smooth_throat_mach")
    fig.savefig(filename + ".png", bbox_inches="tight")
    fig.savefig(filename + ".svg", bbox_inches="tight")

    # Show figure
    plt.show()
