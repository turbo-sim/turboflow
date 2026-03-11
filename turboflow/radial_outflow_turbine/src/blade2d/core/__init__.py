"""Core numerical kernels for 2D blade parameterization."""

from .bspline import (
    basis_derivative_at_u,
    basis_derivative_matrix,
    basis_functions_at_u,
    basis_matrix,
    curve_derivative_points,
    curve_points,
    open_uniform_knot_vector,
)
from .camber_curvature import (
    compute_curvature_camberline,
    cumulative_trapezoid,
    default_curvature_control_points,
)
from .chord import (
    chord_from_theta,
    resolve_axial_chord,
    resolve_radial_chord,
)
from .thickness_quartic import (
    quartic_bspline_thickness,
    thickness_from_geometry,
)
from .thickness_naca import (
    naca_modified_thickness,
)
from .section_builder import (
    build_section_from_camber_and_thickness,
    build_section_from_results,
    dimensionalize_section,
)
