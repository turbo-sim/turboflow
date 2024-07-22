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


def print_banner():
    """Prints a banner."""
    banner = """
        ______ __  __ ____   ____   ____   ______ __    ____  _       __
       /_  __// / / // __ \ / __ ) / __ \ / ____// /   / __ \| |     / /
        / /  / / / // /_/ // __  |/ / / // /_   / /   / / / /| | /| / / 
       / /  / /_/ // _, _// /_/ // /_/ // __/  / /___/ /_/ / | |/ |/ /   
      /_/   \____//_/ |_|/_____/ \____//_/    /_____/\____/  |__/|__/  
    """
    print(BREAKLINE)
    print(banner)
    print(BREAKLINE)
    # https://manytools.org/hacker-tools/ascii-banner/
    # Style: Sland


def print_package_info():
    """Prints package information with predefined values."""

    info = f""" Version:       {__version__}
 Repository:    {URL_GITHUB}
 Documentation: {URL_DOCS}"""
    print_banner()
    print(BREAKLINE)
    print(info)
    print(BREAKLINE)


def add_package_to_pythonpath():
    """Append package containing this __file__ to the PYTHONPATH environment variable"""

    # Define bash command to append cwd to PYTHONPATH
    # PACKAGE_PATH = os.getcwd()
    # PACKAGE_PATH = os.path.dirname(os.getcwd())
    PACKAGE_PATH = os.path.dirname(__file__)
    bashrc_header = f"# Append {PACKAGE_NAME} package to PYTHONPATH"
    if sys.platform == "win32":  # Windows
        bashrc_line = f'export PYTHONPATH=$PYTHONPATH\;"{PACKAGE_PATH}"'
    else:  # Linux or MacOS
        bashrc_line = f'export PYTHONPATH=$PYTHONPATH:"{PACKAGE_PATH}"'

    # Locate the .bashrc file
    bashrc_path = os.path.expanduser("~/.bashrc")

    # Ask for user confirmation with default set to no
    response = input(
        f"Do you want to add the {PACKAGE_NAME} path to your .bashrc? [yes/NO]: "
    )
    if response.lower() in ["y", "yes"]:
        try:
            # Check if the line already exists in the .bashrc
            with open(bashrc_path, "r") as file:
                if bashrc_line in file.read():
                    print(f".bashrc already contains the {PACKAGE_NAME} path.")
                else:
                    with open(bashrc_path, "a") as file_append:
                        file_append.write(f"\n{bashrc_header}")
                        file_append.write(f"\n{bashrc_line}\n")
                    print(f"Path to {PACKAGE_NAME} package added to .bashrc")
                    print(
                        f"Restart the terminal or run 'source ~/.bashrc' for the changes to take effect."
                    )
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Operation aborted by user.")
