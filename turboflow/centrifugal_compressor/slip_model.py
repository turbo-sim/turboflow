
import numpy as np

# List of valid options
SLIP_MODELS = [
    "no-slip",
    "wiesner",
]

def evaluate_slip(slip_model, input, geometry):

    # Function mappings for each slip model
    slip_model_functions = {
        SLIP_MODELS[0] : lambda x, y: 0,
        SLIP_MODELS[1] : get_slip_weisner,
    }

    # Evaluate deviation model
    if slip_model in SLIP_MODELS:
        slip_velocity = slip_model_functions[slip_model](input, geometry)
        return slip_velocity
    else:
        options = ", ".join(f"'{k}'" for k in SLIP_MODELS)
        raise ValueError(
            f"Invalid deviation model: '{slip_model}'. Available options: {options}"
        )

def get_slip_weisner(input, geometry):

    # Load variables
    u_out = input["u_out"]
    theta_out = geometry["trailing_edge_angle"]
    z = geometry["number_of_blades"]

    return u_out*np.sqrt(np.cos(theta_out))/(z**0.7)