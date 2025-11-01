#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run performance analysis for a radial-outflow turbine.

👉 How to use:
1. Place this script in the same folder as your YAML file (or adjust the path below).
2. Edit the variable `YAML_FILE` near the top to point to your YAML file name or path.
3. Run:
       python run_radial_outflow_performance.py
4. Results (Excel, YAML snapshot, and pickle) will appear in the 'output' folder.
"""

import os
import sys
import yaml
import numpy as np
import logging

# -------------------------------------------------------------------------
# 💾 >>>>>>>>>>>>  INPUT: specify your YAML filename or path here  <<<<<<<<<<<<<
# -------------------------------------------------------------------------
# YAML_FILE = "C:\Users\sprdi\OneDrive - Danmarks Tekniske Universitet\Roberto Agromayor's files - 2026 ASME Turbo Expo\paper 2 - Radial outflow single-phase\simulations\development\radial_config_trial1.yaml"  # change this to your YAML file path if needed
YAML_FILE = r"C:\Users\sprdi\OneDrive - Danmarks Tekniske Universitet\Roberto Agromayor's files - 2026 ASME Turbo Expo\paper 2 - Radial outflow single-phase\simulations\development\radial_config_trial1.yaml"
OUTPUT_DIR = "output"                       # Folder where results are saved
OUTPUT_NAME = "radial_outflow_perf"         # Base name for result files (without timestamp)
# -------------------------------------------------------------------------


# ---- Project imports (adjust paths if your modules are in another folder) ----
from turboflow.radial_outflow_turbine import performance_analysis as perf
from turboflow.radial_outflow_turbine import geometry_model as geom


# ---------------- Logging ----------------
def make_logger():
    logger = logging.getLogger("radial_outflow_perf")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


# --------------- Helpers -----------------
def _maybe_eval_numpy_expression(val):
    """Allow simple 'np.*' expressions in YAML values (like np.linspace(...))."""
    if isinstance(val, str) and ("np." in val or "numpy." in val):
        safe_env = {"np": np, "numpy": np}
        return eval(val, {"__builtins__": {}}, safe_env)
    return val


def _walk_and_eval(obj):
    """Recursively evaluate numpy expressions inside dict/lists."""
    if isinstance(obj, dict):
        return {k: _walk_and_eval(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_eval(v) for v in obj]
    return _maybe_eval_numpy_expression(obj)


def _normalize_simulation_options(sim_opts):
    """Clean up and normalize keys for simulation options."""
    sim_opts = dict(sim_opts or {})

    # Fix typo
    if "choking_citerion" in sim_opts and "choking_criterion" not in sim_opts:
        sim_opts["choking_criterion"] = sim_opts.pop("choking_citerion")

    # Normalize loss model
    lm = sim_opts.get("loss_model", None)
    if isinstance(lm, dict):
        model_name = lm.get("model", None)
        if model_name is None:
            raise ValueError("loss_model dict must include a 'model' key (e.g., {model: benner})")
    elif isinstance(lm, str) or lm is None:
        pass
    else:
        raise ValueError("loss_model must be a string or a dict with a 'model' key.")

    # Normalize choking criterion
    cc = sim_opts.get("choking_criterion", None)
    if cc is None:
        sim_opts["choking_criterion"] = "critical_mach_number"
    else:
        mapping = {
            "evaluate_cascade_throat": "critical_mach_number",
            "critical_mach": "critical_mach_number",
            "critical_mass_flow": "critical_mass_flow_rate",
            "critical_isentropic": "critical_isentropic_throat",
        }
        sim_opts["choking_criterion"] = mapping.get(cc, cc)

    sim_opts.setdefault("blockage_model", "flat_plate_turbulent")
    return sim_opts


def _prepare_config(cfg, logger):
    """Prepare configuration dictionary for compute_performance."""
    cfg = _walk_and_eval(cfg)

    sim_opts = _normalize_simulation_options(cfg.get("simulation_options", {}))
    perf_section = cfg.get("performance_analysis", {})
    perf_map = perf_section.get("performance_map") or cfg.get("operation_points") or {}
    solver_opts = perf_section.get("solver_options", {})
    initial_guess = perf_section.get("initial_guess", None)

    consolidated_geometry = cfg  # Let performance engine call calculate_full_geometry internally

    final_config = {
        "geometry": consolidated_geometry,
        "simulation_options": sim_opts,
        "performance_analysis": {
            "solver_options": solver_opts,
            "initial_guess": initial_guess,
            "performance_map": perf_map,
        },
    }

    logger.info("Simulation options:")
    for k, v in sim_opts.items():
        logger.info(f"  - {k}: {v}")

    return final_config, perf_map


# -------------------- Main --------------------
def run(yaml_path, out_dir="output", out_filename=None):
    logger = make_logger()

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    config, op_points = _prepare_config(cfg, logger)
    operation_points = op_points

    solvers = perf.compute_performance(
        operation_points=operation_points,
        config=config,
        out_filename=out_filename,
        out_dir=out_dir,
        export_results=True,
        stop_on_failure=False,
        logger=logger,
    )

    logger.info("\n✅ Performance analysis completed successfully!")
    return solvers


# -------------------- Script entry --------------------
if __name__ == "__main__":
    run(YAML_FILE, out_dir=OUTPUT_DIR, out_filename=OUTPUT_NAME)
