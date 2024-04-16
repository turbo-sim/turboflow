from . import loss_model_benner as br
from . import loss_model_kacker_okapuu as ko
from . import loss_model_moustapha as mo
from . import loss_model_benner_moustapha as bm
from .. import utilities as utils

# List of valid options
LOSS_MODELS = [
    "kacker_okapuu",
    "moustapha",
    "benner",
    "benner_moustapha",
    "isentropic",
    "custom",
]

LOSS_COEFFICIENTS = [
    "stagnation_pressure",
]

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

    Args:
        loss_model_options (dict): Options for the loss model.
        input_parameters (dict): Input parameters required for loss model calculation.

    Returns:
        tuple: (Y, loss_dict), where Y is the loss coefficient and loss_dict is a dictionary of loss components.
    """
    # TODO improve docstring and add citations to the papers of each loss model

    # Function mappings for each loss model
    loss_model_functions = {
        LOSS_MODELS[0]: ko.compute_losses,
        LOSS_MODELS[1]: mo.compute_losses,
        LOSS_MODELS[2]: br.compute_losses,
        LOSS_MODELS[3]: bm.compute_losses,
        LOSS_MODELS[4]: lambda _: {
            "loss_profile": 0.0,
            "loss_incidence": 0.0,
            "loss_trailing": 0.0,
            "loss_secondary": 0.0,
            "loss_clearance": 0.0,
            "loss_total": 0.0,
        },
        LOSS_MODELS[5]: lambda input_parameters: 
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

    # Validate the output losses dictionary
    utils.validate_keys(loss_dict, KEYS_LOSSES, KEYS_LOSSES)

    return loss_dict


def apply_tuning_factors(loss_dict, tuning_factors):
    """
    Applies tuning factors to the loss model

    Args:
        loss_dict (dict): Dictionary containing loss components.
        tuning_factors (dict): Dictionary containing the multiplicative tuning factors.

    Raises:
        KeyError: If a key from tuning_factors is not found in loss_dict.
    """

    for key, factor in tuning_factors.items():
        if key not in loss_dict:
            raise KeyError(f"Tuning factor key '{key}' not found in loss dictionary.")
        loss_dict[key] *= factor

    return loss_dict


def validate_loss_dictionary(loss_dict):
    """
    Validates that the loss dictionary contains exactly the expected keys

    Parameters:
    - loss_dict (dict): The dictionary to validate.

    Raises:
    - ValueError: If the dictionary does not contain the required keys or contains extra keys.
    """

    missing_keys = set(KEYS_LOSSES) - loss_dict.keys()
    extra_keys = loss_dict.keys() - set(KEYS_LOSSES)

    if missing_keys:
        raise ValueError(
            f"Missing required keys in loss dictionary: {', '.join(missing_keys)}"
        )

    if extra_keys:
        raise ValueError(
            f"Extra keys found in loss dictionary: {', '.join(extra_keys)}"
        )

    return loss_dict
