"""YAML-driven blade section parametrization entrypoints.

This module is the integration layer for:
- camberline model selection (`camberline_type`)
- thickness model selection (`thickness_model`)
- section generation for radial and axial cascades

Current Blade2D implementation supports:
- `camberline_type`: `curvature_based`
- `thickness_model`: `NACA` or `B_spline`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..adapters.axial import section_from_axial_geometry
from ..adapters.radial import section_from_radial_geometry


VALID_CAMBERLINE_TYPES = ("curvature_based",)
VALID_THICKNESS_MODELS = ("NACA", "B_spline")


def _normalize_model(value: str) -> str:
    return str(value).strip()


def validate_cascade_geometry_models(geometry: dict[str, Any]) -> None:
    """Validate model selectors for one cascade geometry block."""
    if "camberline_type" not in geometry:
        raise KeyError("geometry must include 'camberline_type'.")
    if "thickness_model" not in geometry:
        raise KeyError("geometry must include 'thickness_model' ('NACA' or 'B_spline').")

    camberline_type = _normalize_model(geometry["camberline_type"])
    thickness_model = _normalize_model(geometry["thickness_model"])

    if camberline_type not in VALID_CAMBERLINE_TYPES:
        raise ValueError(
            f"Unsupported camberline_type='{camberline_type}'. "
            f"Supported in this integration: {VALID_CAMBERLINE_TYPES}."
        )
    if thickness_model not in VALID_THICKNESS_MODELS:
        raise ValueError(
            f"Unsupported thickness_model='{thickness_model}'. "
            f"Supported: {VALID_THICKNESS_MODELS}."
        )


def build_radial_section_from_geometry(geometry: dict[str, Any], *, n_points: int = 161) -> dict[str, Any]:
    """Generate one radial section from a validated geometry dictionary."""
    validate_cascade_geometry_models(geometry)
    return section_from_radial_geometry(
        geometry,
        n_points=n_points,
        thickness_n_control=11,
        enable_le_blend=False,
        thickness_lambda_smooth_d2=2.0e-3,
        thickness_lambda_smooth_d1=5.0e-5,
    )


def build_axial_section_from_geometry(geometry: dict[str, Any], *, n_points: int = 161) -> dict[str, Any]:
    """Generate one axial section from a validated geometry dictionary."""
    validate_cascade_geometry_models(geometry)
    return section_from_axial_geometry(
        geometry,
        n_points=n_points,
        thickness_n_control=11,
        enable_le_blend=False,
        thickness_lambda_smooth_d2=2.0e-3,
        thickness_lambda_smooth_d1=5.0e-5,
    )


def load_components_from_yaml(path: str | Path) -> list[dict[str, Any]]:
    """Load component list from YAML file."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "components" not in data:
        raise ValueError("YAML must be a dictionary with a top-level 'components' key.")
    components = data["components"]
    if not isinstance(components, list):
        raise ValueError("'components' must be a list.")
    return components


def build_sections_from_yaml(path: str | Path, *, n_points: int = 161) -> dict[str, dict[str, Any]]:
    """Build sections for all cascade components in YAML.

    Returns a dict keyed by component name.
    """
    components = load_components_from_yaml(path)
    out: dict[str, dict[str, Any]] = {}

    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            continue
        ctype = comp.get("component_type", None)
        geom = comp.get("geometry", None)
        if ctype not in ("radial_cascade", "axial_cascade"):
            continue
        if not isinstance(geom, dict):
            raise ValueError(f"Component index {i} has invalid or missing 'geometry' block.")

        name = str(comp.get("name", f"component_{i+1}"))
        if ctype == "radial_cascade":
            out[name] = build_radial_section_from_geometry(geom, n_points=n_points)
        else:
            out[name] = build_axial_section_from_geometry(geom, n_points=n_points)

    return out
