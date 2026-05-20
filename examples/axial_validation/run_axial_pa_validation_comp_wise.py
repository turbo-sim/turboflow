"""

Edit YAML_PATH to point at your config file.
Set USE_MAP=True to sweep using performance_analysis.performance_map.
"""

import os
import sys
import logging
import yaml
import numpy as np
from pathlib import Path

from turboflow.radial_outflow_turbine.performance_analysis_update import compute_performance


# -----------------------
# USER SETTINGS 
# -----------------------
YAML_PATH = r"one_stage_comp_wise.yaml"   
# SCRIPT_DIR = Path(__file__).resolve().parent
# YAML_PATH = SCRIPT_DIR / "config" / "one_stage_comp_wise.yaml"
USE_MAP   =True                          # True = use performance map, False = single point
OUT_DIR   = "output_axial"                         # output folder
EXPORT    = True                             # write Excel/YAML/PKL artifacts
VERBOSE   = 2                                # 0=warnings, 1=info, 2=debug
# -----------------------


def _eval_np_expressions(obj):
    """
    Recursively evaluate simple NumPy expressions in strings, e.g.
    "13.8e4/np.linspace(1.1, 5.0, 100)". Safe scope: only exposes numpy as 'np'.
    """
    if isinstance(obj, dict):
        return {k: _eval_np_expressions(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_eval_np_expressions(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if ("np." in s) or any(op in s for op in ("/", "*", "+", "-", "e")):
            try:
                return eval(s, {"__builtins__": {}}, {"np": np})
            except Exception:
                return obj
        return obj
    return obj


def _choose_operation_points(cfg, use_map: bool):
    if use_map:
        pa = cfg.get("performance_analysis", {}) or {}
        perf_map = pa.get("performance_map") or {}
        if not perf_map:
            raise ValueError("USE_MAP=True but 'performance_analysis.performance_map' is missing in YAML.")
        return perf_map
    op = cfg.get("operation_points") or cfg.get("operation_point")
    if not op:
        raise ValueError("Single-point run requires a top-level 'operation_points' block in YAML.")
    return op


def build_logger(verbosity: int):
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logger = logging.getLogger("performance")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def main():
    # Load YAML
    if not os.path.exists(YAML_PATH):
        print(f"YAML not found: {YAML_PATH}", file=sys.stderr)
        sys.exit(2)

    with open(YAML_PATH, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Evaluate NumPy-style expressions inside YAML
    cfg = _eval_np_expressions(raw_cfg)

    # Ensure solver prints convergence to terminal
    pa = cfg.setdefault("performance_analysis", {})
    solver_opts = pa.setdefault("solver_options", {})
    solver_opts.setdefault("print_convergence", True)

    # Choose operation points
    operation_points = _choose_operation_points(cfg, use_map=USE_MAP)

    # Logger
    logger = build_logger(VERBOSE)
    logger.info("Starting performance analysis...")
    logger.info(f"Config file: {YAML_PATH}")
    logger.info(f"Output dir : {OUT_DIR}")
    logger.info(f"Run mode   : {'map (sweep)' if USE_MAP else 'single point'}")

    # Run
    solvers = compute_performance(
        operation_points=operation_points,
        config=cfg,
        out_filename=os.path.splitext(os.path.basename(YAML_PATH))[0],
        out_dir=OUT_DIR,
        export_results=EXPORT,
        logger=logger,
    )

    # Exit code based on convergence
    success_count = sum(1 for s in solvers if getattr(s, "success", False))
    if success_count == len(solvers):
        logger.info("All operation points converged.")
        sys.exit(0)
    else:
        logger.warning(f"{len(solvers) - success_count} operation point(s) failed to converge.")
        sys.exit(1)


if __name__ == "__main__":
    main()
