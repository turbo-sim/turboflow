import os
import numpy as np
import matplotlib.pyplot as plt
import meanline_axial as ml

if __name__ == "__main__":

    # Example data
    x = np.linspace(0.5, 1.5, 51)
    inputs = np.asarray([x, 1 + 0*x])  # f1(x) = x and f2(x) = 1

    # Calculate softmin using various methods
    f_max = np.min(inputs, axis=0)
    f_pnorm = ml.softmin(inputs, method="p-norm", alpha=25, axis=0)
    f_logsumexp = ml.softmin(inputs, method="logsumexp", alpha=25, axis=0)
    f_boltzmann = ml.softmin(inputs, method="boltzmann", alpha=25, axis=0)

    # Create the folder to save figures
    fig_dir = "figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Plot the data
    fig, ax = plt.subplots()
    ax.set_xlabel('Exit Mach number')
    ax.set_ylabel('Throat Mach number')
    ax.plot(x, f_max, label='Exact maximum', color='black')
    ax.plot(x, f_boltzmann, label='Boltzmann maximum')
    ax.plot(x, f_logsumexp, label='LogSumExp maximum')
    ax.plot(x, f_pnorm, label='p-norm maximum', linestyle=":")
    ax.legend(fontsize=10)
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)
    filename = os.path.join(fig_dir, "smooth_throat_mach")
    fig.savefig(filename + ".png", bbox_inches="tight")
    fig.savefig(filename + ".svg", bbox_inches="tight")

    # Show figure
    plt.show()