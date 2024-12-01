import os
import sys

# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()


# Import submodules
from .math import *
from .plot_functions import *
from .config_validation import *

# Import subpackages
from .pysolver_view import *
from .properties import *
from .cycles import *
from .utilities import *

from .axial_turbine import *
from .axial_turbine.performance_analysis import *
from .axial_turbine.design_optimization import *

# Set plot options
set_plot_options()


# Package info
__version__ = "0.1.12"
PACKAGE_NAME = "turboflow"
URL_GITHUB = "https://github.com/turbo-sim/turboflow"
URL_DOCS = "https://turbo-sim.github.io/turboflow/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"


# def print_banner():
#     """Prints a banner."""
#     banner = """
#         ______ __  __ ____   ____   ____   ______ __    ____  _       __
#        /_  __// / / // __ \ / __ ) / __ \ / ____// /   / __ \| |     / /
#         / /  / / / // /_/ // __  |/ / / // /_   / /   / / / /| | /| / / 
#        / /  / /_/ // _, _// /_/ // /_/ // __/  / /___/ /_/ / | |/ |/ /   
#       /_/   \____//_/ |_|/_____/ \____//_/    /_____/\____/  |__/|__/  
#     """
#     print(BREAKLINE)
#     print(banner)
#     print(BREAKLINE)
#     # https://manytools.org/hacker-tools/ascii-banner/
#     # Style: Sland


# def print_package_info():
#     """Prints package information with predefined values."""

#     info = f""" Version:       {__version__}
#  Repository:    {URL_GITHUB}
#  Documentation: {URL_DOCS}"""
#     print_banner()
#     print(BREAKLINE)
#     print(info)
#     print(BREAKLINE)



def print_banner(logger=None):
    """Prints or logs a banner."""
    banner = """
        ______ __  __ ____   ____   ____   ______ __    ____  _       __
       /_  __// / / // __ \ / __ ) / __ \ / ____// /   / __ \| |     / /
        / /  / / / // /_/ // __  |/ / / // /_   / /   / / / /| | /| / / 
       / /  / /_/ // _, _// /_/ // /_/ // __/  / /___/ /_/ / | |/ |/ /   
      /_/   \____//_/ |_|/_____/ \____//_/    /_____/\____/  |__/|__/  
    """
    # https://manytools.org/hacker-tools/ascii-banner/
    # Style: Sland
    output = f"{BREAKLINE}\n{banner}\n{BREAKLINE}"
    if logger:
        log_line_by_line(logger, output)
    else:
        print(output)


def print_package_info(logger=None):
    """Prints or logs package information with predefined values."""

    info = f""" Version:       {__version__}
 Repository:    {URL_GITHUB}
 Documentation: {URL_DOCS}"""
    
    if logger:
        print_banner(logger=logger)
        log_line_by_line(logger, f"{BREAKLINE}\n{info}\n{BREAKLINE}")
    else:
        print_banner()
        print(f"{BREAKLINE}\n{info}\n{BREAKLINE}")