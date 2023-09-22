from glob import glob
from os.path import basename, dirname, isfile

# Get a list of all Python files in the current directory (excluding __init__.py)
modules = glob(dirname(__file__) + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# Import all modules specified in __all__
for module in __all__:
    exec(f"from .{module} import *")







