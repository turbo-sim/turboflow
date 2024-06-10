# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()

import toml

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


import os
import toml

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

__version__ = "0.1.10"

def print_package_info():
    """Prints package information with predefined values."""

    info = f""" Version:       {__version__}"""
    
    # Assuming print_banner and BREAKLINE are defined elsewhere in your code
    print_banner()
    print(BREAKLINE)
    print(info)

# def print_package_info():
#     """Prints package information."""
#     try:
#         pyproject_path = find_pyproject_toml()
#         with open(pyproject_path, "r") as f:
#             pyproject = toml.load(f)

#         poetry_config = pyproject.get("tool", {}).get("poetry", {})
#         info = f""" Version:       {poetry_config.get("version")}
#  Repository:    {poetry_config.get("repository")}
#  Documentation: {poetry_config.get("documentation")}"""
#         print_banner()
#         print(BREAKLINE)
#         print(info)
#     except FileNotFoundError:
#         print("Error: pyproject.toml file not found.")


# def find_pyproject_toml():
#     """Finds the path to the pyproject.toml file."""
#     current_script_path = os.path.abspath(__file__)
#     current_script_dir = os.path.dirname(current_script_path)
#     project_root_dir = os.path.dirname(current_script_dir)
#     pyproject_path = os.path.join(project_root_dir, "pyproject.toml")
#     return pyproject_path


