import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler


def set_plot_options(
    fontsize=13,
    grid=True,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
):
    """Set plot options for publication-quality figures"""
	
    if isinstance(color_order, str):
        if color_order.lower() == "default":
            color_order = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

        elif color_order.lower() == "matlab":
            color_order = [
                "#0072BD",
                "#D95319",
                "#EDB120",
                "#7E2F8E",
                "#77AC30",
                "#4DBEEE",
                "#A2142F",
            ]

    # Define dictionary of custom settings
    rcParams = {
        "text.usetex": False,
        "font.size": fontsize,
        "font.style": "normal",
        "font.family": "serif",  # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
        "font.serif": ["Times New Roman"],  # ['times new roman', 'cmr10']
        "mathtext.fontset": "stix",  # ["stix", 'cm']
        "axes.edgecolor": "black",
        "axes.linewidth": 1.25,
        "axes.titlesize": fontsize,
        "axes.titleweight": "normal",
        "axes.titlepad": fontsize * 1.4,
        "axes.labelsize": fontsize,
        "axes.labelweight": "normal",
        "axes.labelpad": fontsize,
        "axes.xmargin": margin,
        "axes.ymargin": margin,
        # "axes.zmargin": margin,
        "axes.grid": grid,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.prop_cycle": cycler(color=color_order),
        "grid.alpha": 0.5,
        "grid.color": "black",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "legend.borderaxespad": 1,
        "legend.borderpad": 0.6,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.labelcolor": "black",
        "legend.labelspacing": 0.3,
        "legend.fancybox": True,
        "legend.fontsize": fontsize - 2,
        "legend.framealpha": 1.00,
        "legend.handleheight": 0.7,
        "legend.handlelength": 1.25,
        "legend.handletextpad": 0.8,
        "legend.markerscale": 1.0,
        "legend.numpoints": 1,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "lines.markeredgewidth": 1.25,
        "lines.markerfacecolor": "white",
        "xtick.direction": "in",
        "xtick.labelsize": fontsize - 1,
        "xtick.bottom": major_ticks,
        "xtick.top": major_ticks,
        "xtick.major.size": 6,
        "xtick.major.width": 1.25,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.75,
        "xtick.minor.visible": minor_ticks,
        "ytick.direction": "in",
        "ytick.labelsize": fontsize - 1,
        "ytick.left": major_ticks,
        "ytick.right": major_ticks,
        "ytick.major.size": 6,
        "ytick.major.width": 1.25,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.75,
        "ytick.minor.visible": minor_ticks,
        "savefig.dpi": 500
    }

    # Update the internal Matplotlib settings dictionary
    mpl.rcParams.update(rcParams)


def print_installed_fonts():
    """Print the list of fonts installed in your computer"""
    fonts = mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in sorted(fonts):
        print(font)


def print_rc_parameters():
    """Print the current rcParams used by Matplotlib"""
    params = mpl.rcParams
    for key, value in params.items():
        print(f"{key}: {value}")
    # with open("rcParams_default.txt", 'w') as file:
    #     for key, value in params.items():
    #         file.write(f"{key}: {value}\n")


def create_sample_plot():
    # Create figure
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.gca()
    ax.set_title(r"$f(x) = \cos(2\pi x + \phi)$")
    ax.set_xlabel("$x$ axis")
    ax.set_ylabel("$y$ axis")
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.15, 1.15])
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Plot the function
    x = np.linspace(0, 1, 200)
    phi_array = np.arange(0, 1, 0.2) * np.pi
    for phi in phi_array:
        ax.plot(x, np.cos(2 * np.pi * x + phi), label=f"$\phi={phi/np.pi:0.2f}\pi$")

    # Create legend
    ax.legend(loc="lower right", ncol=1)

    # Adjust PAD
    fig.tight_layout(pad=1.00, w_pad=None, h_pad=None)

    # Display figure
    plt.show()


if __name__ == "__main__":
    set_plot_options()
    create_sample_plot()

