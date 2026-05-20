#!/usr/bin/env python3
"""
One-shot comparator for:
  1) axial_turbine YAML (reference)
  2) radial_outflow_turbine component-wise YAML

It runs one operating point in each solver stack and exports deltas for:
  - boundary conditions
  - overall KPIs
  - plane-wise quantities
  - cascade-wise quantities
  - geometry (reference full-geometry vs component-wise axial-cascade full-geometry)

Usage:
  python compare_axial_vs_compwise.py ^
    --axial-yaml "C:\\path\\to\\one_stage_config_reference.yaml" ^
    --comp-yaml "C:\\path\\to\\one_stage_comp_wise.yaml" ^
    --output-dir "C:\\path\\to\\output" ^
    --prefix "one_stage_compare"

Or run with no YAML flags and let the script auto-discover:
  paper 2 - Radial outflow single-phase/simulations/development/config/
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import os
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml


# Make local repository importable when running this file directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from turboflow.axial_turbine import geometry_model as axial_geom  # noqa: E402
from turboflow.axial_turbine import performance_analysis as axial_pa  # noqa: E402
from turboflow.radial_outflow_turbine import (  # noqa: E402
    geometry_model_axial as comp_ax_geom,
)
from turboflow.radial_outflow_turbine import (  # noqa: E402
    performance_analysis_update as comp_pa,
)


class SimpleLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = bool(verbose)

    def _print(self, level: str, msg: str):
        if self.verbose:
            print(f"[{level}] {msg}")

    def info(self, msg: str):
        self._print("INFO", str(msg))

    def warning(self, msg: str):
        self._print("WARN", str(msg))

    def error(self, msg: str):
        self._print("ERROR", str(msg))


def _is_numeric_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)


def _safe_eval_string(s: str):
    try:
        return eval(s, {"__builtins__": {}}, {"np": np})
    except Exception:
        return s


def _evaluate_expressions(obj: Any):
    if isinstance(obj, dict):
        return {k: _evaluate_expressions(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_evaluate_expressions(v) for v in obj]
    if isinstance(obj, str):
        return _safe_eval_string(obj)
    return obj


def _ensure_numpy_like_for_axial_geometry(geom: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in geom.items():
        if isinstance(v, list):
            try:
                out[k] = np.array(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _pick_first_from_value(v: Any):
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return v.flat[0].item() if v.size > 0 else v
    if isinstance(v, list):
        return v[0] if len(v) > 0 else v
    if isinstance(v, tuple):
        return v[0] if len(v) > 0 else v
    return v


def _extract_operation_point_from_operation_points(op: Any) -> Dict[str, Any]:
    if isinstance(op, list):
        if not op:
            raise ValueError("operation_points is an empty list.")
        if not isinstance(op[0], dict):
            raise TypeError("operation_points list must contain dictionaries.")
        return dict(op[0])
    if isinstance(op, dict):
        return {k: _pick_first_from_value(v) for k, v in op.items()}
    raise TypeError("operation_points must be dict or list of dicts.")


def _extract_first_operation_point(
    cfg: Dict[str, Any],
    solver_kind: str,
    op_index: int = 0,
) -> Dict[str, Any]:
    if "operation_points" in cfg and cfg["operation_points"] is not None:
        op = _extract_operation_point_from_operation_points(cfg["operation_points"])
    else:
        perf_map = cfg.get("performance_analysis", {}).get("performance_map", None)
        if perf_map is None:
            raise ValueError(
                "Cannot find operation point source. Need 'operation_points' or "
                "'performance_analysis.performance_map'."
            )
        if solver_kind == "axial":
            op_list = axial_pa.generate_operation_points(perf_map)
        elif solver_kind == "comp":
            op_list = comp_pa.generate_operation_points(perf_map)
        else:
            raise ValueError(f"Unknown solver_kind: {solver_kind}")

        if op_index < 0 or op_index >= len(op_list):
            raise IndexError(
                f"op_index={op_index} outside generated operation points range [0, {len(op_list)-1}]"
            )
        op = dict(op_list[op_index])

    if solver_kind == "axial":
        if "fluid_name" not in op:
            fluid_cfg = cfg.get("fluid", {}) or {}
            if "name" in fluid_cfg:
                op["fluid_name"] = fluid_cfg["name"]
    else:
        # component-wise validator requires exact keys
        allowed = {"p0_in", "T0_in", "p_out", "alpha_in", "omega"}
        op = {k: v for k, v in op.items() if k in allowed}

    return op


def _scalar(x: Any):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.size == 1:
            return x.reshape(-1)[0].item()
        return x
    if isinstance(x, np.generic):
        return x.item()
    return x


def _to_float_or_none(x: Any):
    x = _scalar(x)
    if _is_numeric_scalar(x):
        return float(x)
    return None


def _relative_delta(a: float, b: float, eps: float = 1e-12) -> float:
    return (b - a) / max(abs(a), eps)


def _common_numeric_keys(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    out = []
    for k in keys:
        if _to_float_or_none(a[k]) is not None and _to_float_or_none(b[k]) is not None:
            out.append(k)
    return out


def _as_array_1d(x: Any) -> np.ndarray:
    try:
        if isinstance(x, np.ndarray):
            arr = x.reshape(-1)
            return arr.astype(float)
        if isinstance(x, (list, tuple)):
            arr = np.asarray(x).reshape(-1)
            return arr.astype(float)
        if _is_numeric_scalar(x):
            return np.asarray([x], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    return np.asarray([], dtype=float)


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_overall_rows(overall_ax: Dict[str, Any], overall_comp: Dict[str, Any]):
    rows = []
    for key in _common_numeric_keys(overall_ax, overall_comp):
        a = _to_float_or_none(overall_ax[key])
        b = _to_float_or_none(overall_comp[key])
        rows.append(
            {
                "key": key,
                "axial_value": a,
                "comp_value": b,
                "abs_delta_comp_minus_axial": b - a,
                "rel_delta_comp_minus_axial": _relative_delta(a, b),
            }
        )
    return rows


def _build_series_rows(
    series_ax: Dict[str, Any],
    series_comp: Dict[str, Any],
    block_name: str,
):
    rows: List[Dict[str, Any]] = []
    common = sorted(set(series_ax.keys()) & set(series_comp.keys()))
    for key in common:
        arr_a = _as_array_1d(series_ax[key])
        arr_b = _as_array_1d(series_comp[key])
        if arr_a.size == 0 or arr_b.size == 0:
            continue
        n = min(arr_a.size, arr_b.size)
        for i in range(n):
            va = float(arr_a[i])
            vb = float(arr_b[i])
            rows.append(
                {
                    "block": block_name,
                    "index": i,
                    "key": key,
                    "axial_value": va,
                    "comp_value": vb,
                    "abs_delta_comp_minus_axial": vb - va,
                    "rel_delta_comp_minus_axial": _relative_delta(va, vb),
                }
            )
    return rows


def _axial_geometry_rows(full_geom: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = int(_scalar(full_geom.get("number_of_cascades", 0)))
    rows = []
    for i in range(n):
        row = {"row_index": i}
        for k, v in full_geom.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.size > i:
                row[k] = _scalar(v[i])
            else:
                row[k] = _scalar(v)
        rows.append(row)
    return rows


def _comp_geometry_rows(cfg_comp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    components = cfg_comp.get("components", [])
    if not isinstance(components, list):
        return rows

    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            continue
        ctype = str(comp.get("component_type", "")).lower()
        if ctype != "axial_cascade":
            continue
        raw = comp.get("geometry", {})
        if not isinstance(raw, dict):
            continue
        full = comp_ax_geom.calculate_full_geometry_for_axial_cascade(
            raw,
            name=comp.get("name", f"row_{i+1}"),
        )
        row = {"row_index": len(rows)}
        row.update({k: _scalar(v) for k, v in dict(full).items()})
        rows.append(row)
    return rows


def _build_geometry_rows(
    rows_ax: List[Dict[str, Any]],
    rows_comp: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = min(len(rows_ax), len(rows_comp))
    for i in range(n):
        a = rows_ax[i]
        b = rows_comp[i]
        common = sorted(set(a.keys()) & set(b.keys()))
        for key in common:
            va = _to_float_or_none(a[key])
            vb = _to_float_or_none(b[key])
            if va is None or vb is None:
                continue
            out.append(
                {
                    "row_index": i,
                    "key": key,
                    "axial_value": va,
                    "comp_value": vb,
                    "abs_delta_comp_minus_axial": vb - va,
                    "rel_delta_comp_minus_axial": _relative_delta(va, vb),
                }
            )
    return out


def _solve_axial_once(
    cfg_axial: Dict[str, Any],
    op_axial: Dict[str, Any],
    logger: SimpleLogger,
):
    cfg = copy.deepcopy(cfg_axial)
    if "geometry" not in cfg:
        raise ValueError("Axial config must contain 'geometry'.")
    cfg["geometry"] = _ensure_numpy_like_for_axial_geometry(cfg["geometry"])

    solvers = axial_pa.compute_performance(
        operation_points=[op_axial],
        config=cfg,
        export_results=False,
        stop_on_failure=False,
        logger=logger,
    )
    if not solvers:
        raise RuntimeError("Axial solver returned empty solver list.")
    solver = solvers[0]
    if getattr(solver, "problem", None) is None:
        raise RuntimeError("Axial solver has no attached problem/results.")
    return solver, solver.problem.results


def _solve_comp_once(
    cfg_comp: Dict[str, Any],
    op_comp: Dict[str, Any],
    logger: SimpleLogger,
):
    cfg = copy.deepcopy(cfg_comp)
    if "components" not in cfg:
        raise ValueError("Component-wise config must contain 'components'.")
    if "fluid" not in cfg:
        raise ValueError(
            "Component-wise config must contain 'fluid' section "
            "(name/model/model_options)."
        )

    solvers = comp_pa.compute_performance(
        operation_points=[op_comp],
        config=cfg,
        export_results=False,
        stop_on_failure=False,
        logger=logger,
        use_previous_solution=False,
        explore_initial_guess=False,
    )
    if not solvers:
        raise RuntimeError("Component-wise solver returned empty solver list.")
    solver = solvers[0]
    if getattr(solver, "problem", None) is None:
        raise RuntimeError("Component-wise solver has no attached problem/results.")
    return solver, solver.problem.results


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be mapping/dict: {path}")
    return _evaluate_expressions(data)


def run_comparison(
    axial_yaml: str,
    comp_yaml: str,
    output_dir: str,
    prefix: str,
    op_index: int,
    quiet: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    logger = SimpleLogger(verbose=not quiet)

    logger.info(f"Loading axial YAML: {axial_yaml}")
    cfg_ax = _read_yaml(axial_yaml)
    logger.info(f"Loading component-wise YAML: {comp_yaml}")
    cfg_cp = _read_yaml(comp_yaml)

    op_ax = _extract_first_operation_point(cfg_ax, solver_kind="axial", op_index=op_index)
    op_cp = _extract_first_operation_point(cfg_cp, solver_kind="comp", op_index=op_index)

    logger.info("Solving axial reference (1 op point)...")
    solver_ax, results_ax = _solve_axial_once(cfg_ax, op_ax, logger)
    logger.info("Solving component-wise case (1 op point)...")
    solver_cp, results_cp = _solve_comp_once(cfg_cp, op_cp, logger)

    status_rows = [
        {
            "solver": "axial_reference",
            "success": bool(getattr(solver_ax, "success", False)),
            "message": str(getattr(solver_ax, "message", "")),
        },
        {
            "solver": "component_wise",
            "success": bool(getattr(solver_cp, "success", False)),
            "message": str(getattr(solver_cp, "message", "")),
        },
    ]

    bc_rows = []
    bc_keys = sorted(set(op_ax.keys()) | set(op_cp.keys()))
    for k in bc_keys:
        va = op_ax.get(k, None)
        vb = op_cp.get(k, None)
        fa = _to_float_or_none(va)
        fb = _to_float_or_none(vb)
        bc_rows.append(
            {
                "key": k,
                "axial_value": va,
                "comp_value": vb,
                "abs_delta_comp_minus_axial": (fb - fa) if (fa is not None and fb is not None) else "",
                "rel_delta_comp_minus_axial": _relative_delta(fa, fb)
                if (fa is not None and fb is not None)
                else "",
            }
        )

    overall_ax = results_ax.get("overall", {}) or {}
    overall_cp = results_cp.get("overall", {}) or {}
    overall_rows = _build_overall_rows(overall_ax, overall_cp)

    planes_ax = results_ax.get("planes", {}) or {}
    planes_cp = results_cp.get("planes", {}) or {}
    plane_rows = _build_series_rows(planes_ax, planes_cp, block_name="planes")

    casc_ax = results_ax.get("cascades", {}) or {}
    casc_cp = results_cp.get("cascades", {}) or {}
    cascade_rows = _build_series_rows(casc_ax, casc_cp, block_name="cascades")

    geom_ax_full = axial_geom.calculate_full_geometry(
        _ensure_numpy_like_for_axial_geometry(copy.deepcopy(cfg_ax["geometry"]))
    )
    geom_ax_rows = _axial_geometry_rows(geom_ax_full)
    geom_cp_rows = _comp_geometry_rows(cfg_cp)
    geom_rows = _build_geometry_rows(geom_ax_rows, geom_cp_rows)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{timestamp}"
    paths = {
        "status": os.path.join(output_dir, f"{base}_status.csv"),
        "boundary": os.path.join(output_dir, f"{base}_boundary_conditions.csv"),
        "overall": os.path.join(output_dir, f"{base}_overall.csv"),
        "planes": os.path.join(output_dir, f"{base}_planes.csv"),
        "cascades": os.path.join(output_dir, f"{base}_cascades.csv"),
        "geometry": os.path.join(output_dir, f"{base}_geometry.csv"),
        "summary": os.path.join(output_dir, f"{base}_summary.txt"),
    }

    _write_csv(paths["status"], status_rows, ["solver", "success", "message"])
    _write_csv(
        paths["boundary"],
        bc_rows,
        ["key", "axial_value", "comp_value", "abs_delta_comp_minus_axial", "rel_delta_comp_minus_axial"],
    )
    _write_csv(
        paths["overall"],
        overall_rows,
        ["key", "axial_value", "comp_value", "abs_delta_comp_minus_axial", "rel_delta_comp_minus_axial"],
    )
    _write_csv(
        paths["planes"],
        plane_rows,
        ["block", "index", "key", "axial_value", "comp_value", "abs_delta_comp_minus_axial", "rel_delta_comp_minus_axial"],
    )
    _write_csv(
        paths["cascades"],
        cascade_rows,
        ["block", "index", "key", "axial_value", "comp_value", "abs_delta_comp_minus_axial", "rel_delta_comp_minus_axial"],
    )
    _write_csv(
        paths["geometry"],
        geom_rows,
        ["row_index", "key", "axial_value", "comp_value", "abs_delta_comp_minus_axial", "rel_delta_comp_minus_axial"],
    )

    kpi_keys = ["mass_flow_rate", "efficiency_ts", "efficiency_tt", "torque", "power"]
    kpi_map = {r["key"]: r for r in overall_rows}
    summary_lines = [
        "Axial vs Component-wise one-shot comparison",
        f"Axial YAML: {axial_yaml}",
        f"Comp YAML:  {comp_yaml}",
        "",
        f"Axial success: {status_rows[0]['success']} | message: {status_rows[0]['message']}",
        f"Comp success:  {status_rows[1]['success']} | message: {status_rows[1]['message']}",
        "",
        "KPI deltas (component-wise minus axial):",
    ]
    for k in kpi_keys:
        row = kpi_map.get(k)
        if row is None:
            summary_lines.append(f"  - {k}: not present in both outputs")
        else:
            summary_lines.append(
                "  - {k}: axial={a:.8g}, comp={b:.8g}, abs_delta={d:.8g}, rel_delta={r:.8g}".format(
                    k=k,
                    a=float(row["axial_value"]),
                    b=float(row["comp_value"]),
                    d=float(row["abs_delta_comp_minus_axial"]),
                    r=float(row["rel_delta_comp_minus_axial"]),
                )
            )

    summary_lines += [
        "",
        "Exported files:",
        f"  - {paths['status']}",
        f"  - {paths['boundary']}",
        f"  - {paths['overall']}",
        f"  - {paths['planes']}",
        f"  - {paths['cascades']}",
        f"  - {paths['geometry']}",
    ]

    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info("Comparison complete.")
    for p in paths.values():
        logger.info(f"  {p}")

    return paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare one axial reference YAML vs one component-wise YAML at one operating point."
    )
    p.add_argument(
        "--axial-yaml",
        default=None,
        help=(
            "Path to reference (axial_turbine style) YAML. "
            "If omitted, auto-discovery is attempted."
        ),
    )
    p.add_argument(
        "--comp-yaml",
        default=None,
        help=(
            "Path to component-wise (radial_outflow_turbine style) YAML. "
            "If omitted, auto-discovery is attempted."
        ),
    )
    p.add_argument("--output-dir", default="output", help="Directory for CSV/text reports.")
    p.add_argument("--prefix", default="axial_vs_comp", help="Output filename prefix.")
    p.add_argument(
        "--op-index",
        type=int,
        default=0,
        help="Index when operation point is selected from generated performance map.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress console logs.")
    return p.parse_args()


def _default_yaml_candidates(filename: str) -> List[str]:
    rel = os.path.join(
        "paper 2 - Radial outflow single-phase",
        "simulations",
        "development",
        "config",
        filename,
    )

    roots: List[str] = []
    for root in [os.getcwd(), THIS_DIR, REPO_ROOT]:
        if root and root not in roots:
            roots.append(root)

    repo_parent = os.path.dirname(REPO_ROOT)
    if repo_parent and repo_parent not in roots:
        roots.append(repo_parent)

    repo_grandparent = os.path.dirname(repo_parent)
    if repo_grandparent and repo_grandparent not in roots:
        roots.append(repo_grandparent)

    return [os.path.abspath(os.path.join(root, rel)) for root in roots]


def _resolve_yaml_path(user_path: str | None, default_filename: str, arg_name: str) -> str:
    if user_path:
        resolved = os.path.abspath(os.path.expanduser(user_path))
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"{arg_name} points to a file that does not exist:\n"
                f"  {resolved}"
            )
        return resolved

    candidates = _default_yaml_candidates(default_filename)
    for path in candidates:
        if os.path.exists(path):
            return path

    tried = "\n".join([f"  - {c}" for c in candidates])
    raise FileNotFoundError(
        f"Could not auto-discover {arg_name}.\n"
        f"Tried:\n{tried}\n"
        f"Provide the path explicitly with {arg_name}."
    )


if __name__ == "__main__":
    args = parse_args()
    axial_yaml = _resolve_yaml_path(
        args.axial_yaml,
        "one_stage_config_reference.yaml",
        "--axial-yaml",
    )
    comp_yaml = _resolve_yaml_path(
        args.comp_yaml,
        "one_stage_comp_wise.yaml",
        "--comp-yaml",
    )
    if not args.quiet:
        print(f"[INFO] Using axial YAML: {axial_yaml}")
        print(f"[INFO] Using comp YAML:  {comp_yaml}")

    run_comparison(
        axial_yaml=axial_yaml,
        comp_yaml=comp_yaml,
        output_dir=os.path.abspath(args.output_dir),
        prefix=args.prefix,
        op_index=args.op_index,
        quiet=args.quiet,
    )
