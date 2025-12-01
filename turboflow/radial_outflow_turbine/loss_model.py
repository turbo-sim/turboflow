from . import loss_model_benner as br
from . import loss_model_kacker_okapuu as ko
from . import loss_model_moustapha as mo
from .. import utilities as utils


import jax.numpy as jnp

LOSS_MODELS = ["kacker_okapuu", "moustapha", "benner", "isentropic", "custom"]
LOSS_COEFFICIENTS = ["stagnation_pressure", "kinetic_energy"]

KEYS_LOSSES = [
    "loss_definition",
    "loss_error",
    "loss_total",
    "loss_profile",
    "loss_clearance",
    "loss_secondary",
    "loss_trailing",
    "loss_incidence",
]


def _normalize_loss_options(loss_model_options):
    """
    Accept string or dict; return a dict with defaults.
    """
    if isinstance(loss_model_options, str):
        opts = {"model": loss_model_options}
    elif isinstance(loss_model_options, dict):
        opts = dict(loss_model_options)  # shallow copy
    else:
        raise ValueError(
            f"loss_model_options must be str or dict, got {type(loss_model_options).__name__}"
        )

    # Defaults
    opts.setdefault("loss_coefficient", "stagnation_pressure")
    opts.setdefault("inlet_displacement_thickness_height_ratio", 0.011)
    opts.setdefault("tuning_factors", {})

    # Custom model must have a value
    if opts["model"] == "custom" and "custom_value" not in opts:
        raise ValueError(
            "custom loss model requires 'custom_value' in loss_model_options"
        )

    # Validate loss coefficient
    if opts["loss_coefficient"] not in LOSS_COEFFICIENTS:
        raise ValueError(
            f"Invalid loss_coefficient '{opts['loss_coefficient']}'. "
            f"Available: {', '.join(LOSS_COEFFICIENTS)}"
        )

    return opts


def evaluate_loss_model(loss_model_options, input_parameters):
    """
    Calculate loss coefficient based on the selected loss model.

    `loss_model_options` may be a string (e.g. 'benner') or a dict like:
      {"model": "benner", "loss_coefficient": "stagnation_pressure",
       "inlet_displacement_thickness_height_ratio": 0.011, "tuning_factors": {...}}
    """
    # Normalize options and inject into input_parameters so model code can read it
    opts = _normalize_loss_options(loss_model_options)
    model = opts["model"]

    # Ensure input_parameters has the options under the expected key
    input_parameters = dict(input_parameters)  # shallow copy
    input_parameters["loss_model"] = opts

    # Function mappings
    model_funcs = {
        "kacker_okapuu": ko.compute_losses,
        "moustapha": mo.compute_losses,
        "benner": br.compute_losses,
    }

    if model in model_funcs:
        loss_dict = model_funcs[model](input_parameters)
    elif model == "isentropic":
        loss_dict = {
            "loss_profile": 0.0,
            "loss_incidence": 0.0,
            "loss_trailing": 0.0,
            "loss_secondary": 0.0,
            "loss_clearance": 0.0,
            "loss_total": 0.0,
        }
    elif model == "custom":
        val = float(opts["custom_value"])
        loss_dict = {
            "loss_profile": 0.0,
            "loss_incidence": 0.0,
            "loss_trailing": 0.0,
            "loss_secondary": 0.0,
            "loss_clearance": 0.0,
            "loss_total": val,
        }
    else:
        raise ValueError(
            f"Invalid loss model '{model}'. Available: {', '.join(LOSS_MODELS)}"
        )

    # Apply tuning factors if any
    tuning = {f"loss_{k}": v for k, v in opts.get("tuning_factors", {}).items()}
    apply_tuning_factors(loss_dict, tuning)

    # Compute loss coefficient definition
    loss_coeff = opts["loss_coefficient"]
    if loss_coeff == "stagnation_pressure":
        p0rel_in = input_parameters["flow"]["p0_rel_in"]
        p0_rel_is = input_parameters["flow"]["p0_rel_is"]
        p0rel_out = input_parameters["flow"]["p0_rel_out"]
        p_out = input_parameters["flow"]["p_out"]
        # definition using isentropic reference:
        Y_definition = (p0_rel_is - p0rel_out) / (p0rel_out - p_out)
    elif loss_coeff == "kinetic_energy":
        w = input_parameters["flow"]["w_out"]
        h = input_parameters["flow"]["h_out"]
        h_is = input_parameters["flow"]["h_is"]
        Y_definition = (h - h_is) / (0.5 * w**2)
    else:
        # guarded above, but keep safety
        raise ValueError(
            f"Invalid loss coefficient '{loss_coeff}'. "
            f"Available: {', '.join(LOSS_COEFFICIENTS)}"
        )

    # Loss error vs. model sum
    #  TODO clipping trick to prevent residual blow up in the first iteration
    # Y_definition = jnp.clip(Y_definition, 0.0, 1.0)
    loss_dict["loss_definition"] = Y_definition
    loss_dict["loss_error"] = Y_definition - loss_dict["loss_total"]

    return loss_dict


def apply_tuning_factors(loss_dict, tuning_factors):
    """
    Multiply components by provided tuning factors (keys like 'loss_profile', ...).
    """
    for key, factor in tuning_factors.items():
        if key not in loss_dict:
            raise KeyError(f"Tuning factor key '{key}' not found in loss dictionary.")
        loss_dict[key] *= factor
    return loss_dict
