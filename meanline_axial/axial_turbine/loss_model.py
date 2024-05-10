from . import loss_model_benner as br
from . import loss_model_kacker_okapuu as ko
from . import loss_model_moustapha as mo
from .. import utilities as utils

# List of valid options
LOSS_MODELS = [
    "kacker_okapuu",
    "moustapha",
    "benner",
    "isentropic",
    "custom",
]
"""
Available loss models.
"""

LOSS_COEFFICIENTS = [
    "stagnation_pressure",
]
"""
Available loss coefficients.
"""

# Keys that the output loss dictionary must have
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


def evaluate_loss_model(loss_model_options, input_parameters):
    """
    Calculate loss coefficient based on the selected loss model.

    The avaialble loss models are:

    - `kacker_okapuu` : loss model according to :cite:`kacker_mean_1982`.
    - `moustapha` : loss model according to :cite:`moustapha_improved_1990`.
    - `benner` : loss model according to :cite:`benner_influence_1997`, :cite:`benner_empirical_2006-1` and :cite:`benner_empirical_2006`. 
    - `isentropic` : Loss coefficient set to zero (isentropic).
    - `custom` :  Constant loss coefficent according to user. Require `loss_model_options["custom_value"]`.

    Parameters
    ----------
    loss_model_options : dict 
        Options for the loss calculation.
    input_parameters : dict
        Input parameters required for loss model calculation.

    Returns:
    dict
        A dictionary containing loss components.
    """

    # Function mappings for each loss model
    loss_model_functions = {
        LOSS_MODELS[0]: ko.compute_losses,
        LOSS_MODELS[1]: mo.compute_losses,
        LOSS_MODELS[2]: br.compute_losses,
        LOSS_MODELS[3]: lambda _: {
            "loss_profile": 0.0,
            "loss_incidence": 0.0,
            "loss_trailing": 0.0,
            "loss_secondary": 0.0,
            "loss_clearance": 0.0,
            "loss_total": 0.0,
        },
        LOSS_MODELS[4]: lambda input_parameters: 
                {
                "loss_profile": 0.0,
                "loss_incidence": 0.0,
                "loss_trailing": 0.0,
                "loss_secondary": 0.0,
                "loss_clearance": 0.0,
                "loss_total": loss_model_options["custom_value"],
            },
    }

    # Evaluate loss model
    model = loss_model_options["model"]
    if model in loss_model_functions:
        loss_dict = loss_model_functions[model](input_parameters)
    else:
        options = ", ".join(f"'{k}'" for k in LOSS_MODELS)
        raise ValueError(f"Invalid loss model '{model}'. Available options: {options}")

    # Apply tuning factors (empty dict if not provided)
    tuning_factors = loss_model_options.get("tuning_factors", {})
    tuning_factors = {f"loss_{key}": value for key, value in tuning_factors.items()}
    apply_tuning_factors(loss_dict, tuning_factors)

    # Compute the loss coefficient according to definition
    loss_coeff = loss_model_options["loss_coefficient"]
    if  loss_coeff == "stagnation_pressure":
        # TODO add other loss coefficient definitions
        p0rel_in = input_parameters["flow"]["p0_rel_in"]
        p0rel_out = input_parameters["flow"]["p0_rel_out"]
        p_out = input_parameters["flow"]["p_out"]
        Y_definition = (p0rel_in - p0rel_out) / (p0rel_out - p_out)

    else:
        options = ", ".join(f"'{k}'" for k in LOSS_COEFFICIENTS)
        raise ValueError(f"Invalid loss coefficient '{loss_coeff}'. Available options: {options}")

    # Compute loss coefficient error
    loss_dict["loss_error"] = (Y_definition - loss_dict["loss_total"])

    # Save the definition of the loss coefficient
    loss_dict["loss_definition"] = loss_model_options["loss_coefficient"]

    return loss_dict


def apply_tuning_factors(loss_dict, tuning_factors):
    """
    Apply tuning factors to the loss model

    The tuning factors are multiplied with their associated loss component.

    Parameters
    ----------
    loss_dict : dict 
        A dictionary containing loss components.
    tuning_factors : dict 
        A dictionary containing the multiplicative tuning factors.

    Raises:
        KeyError: If a key from `tuning_factors` is not found in `loss_dict`.
    """

    for key, factor in tuning_factors.items():
        if key not in loss_dict:
            raise KeyError(f"Tuning factor key '{key}' not found in loss dictionary.")
        loss_dict[key] *= factor

    return loss_dict

