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


# Import submodules
from .math import *
from .plot_functions import *
from .config_validation import *

# Import subpackages
from .pysolver_view import *
from .properties import *
from .thermodynamic_cycles import *
from .utilities import *

from .axial_turbine import *
from .axial_turbine.performance_analysis import *
from .axial_turbine.design_optimization import *

# Set plot options
set_plot_options()
