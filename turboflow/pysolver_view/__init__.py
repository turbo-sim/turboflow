# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython
    import IPython.core.ultratb
    from packaging import version
except ImportError:
    # IPython not available. Use default exception printing.
    pass
else:
    import sys
    ipython_version = version.parse(IPython.__version__)
    if ipython_version >= version.parse("9.0.0"):
        sys.excepthook = IPython.core.ultratb.FormattedTB(theme_name='linux', call_pdb=False)
    else:
        sys.excepthook = IPython.core.ultratb.FormattedTB(color_scheme='linux', call_pdb=False)

# Import JAX in a flexible way
try:
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

import logging
logger = logging.getLogger("jax")
logger.setLevel(logging.WARNING)

# Import submodules
from .pysolver_utilities import *
from .numerical_differentiation import *
from .optimization import *
from .optimization_problems import *
from .optimization_wrappers import *
from .nonlinear_system import *
from .nonlinear_system_problems import *


__version__ = "0.6.8"
URL_GITHUB = "https://github.com/turbo-sim/pysolver_view"
URL_DOCS = "https://turbo-sim.github.io/pysolver_view/"
