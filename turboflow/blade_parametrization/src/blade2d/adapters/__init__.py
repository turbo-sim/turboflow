"""Adapters from geometry-model dictionaries to Blade2D core APIs."""

from .axial import (
    curvature_camberline_from_axial_geometry,
    section_from_axial_geometry,
    thickness_profile_from_axial_geometry,
)
from .radial import (
    curvature_camberline_from_radial_geometry,
    section_from_radial_geometry,
    thickness_profile_from_radial_geometry,
)
