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


from .utilities import *
from .numerical_differentiation import *
from .optimization import *
from .optimization_problems import *
from .optimization_wrappers import *
from .nonlinear_system import *
from .nonlinear_system_problems import *
