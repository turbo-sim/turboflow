"""Blade2D package for curvature and thickness development."""

from .core import (
    basis_derivative_at_u,
    basis_derivative_matrix,
    basis_functions_at_u,
    basis_matrix,
    build_section_from_camber_and_thickness,
    build_section_from_results,
    compute_curvature_camberline,
    chord_from_theta,
    cumulative_trapezoid,
    dimensionalize_section,
    curve_derivative_points,
    curve_points,
    default_curvature_control_points,
    open_uniform_knot_vector,
    naca_modified_thickness,
    quartic_bspline_thickness,
    resolve_axial_chord,
    resolve_radial_chord,
    thickness_from_geometry,
)
from .adapters import (
    curvature_camberline_from_axial_geometry,
    curvature_camberline_from_radial_geometry,
    section_from_axial_geometry,
    section_from_radial_geometry,
    thickness_profile_from_axial_geometry,
    thickness_profile_from_radial_geometry,
)
from .integration import (
    VALID_CAMBERLINE_TYPES,
    VALID_THICKNESS_MODELS,
    build_axial_section_from_geometry,
    build_radial_section_from_geometry,
    build_sections_from_yaml,
    load_components_from_yaml,
    validate_cascade_geometry_models,
)
