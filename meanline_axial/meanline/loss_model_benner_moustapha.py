import numpy as np
from .. import math

# TODO: Add smoothing to the min/max/abs/piecewise functions
# TODO: we can create generic smooth versions of these functions and call them when necessary

# TODO: Create test scripts to plot how the different loss model functions behave when smoothing is applied / verify that the behavior is reasonable beyond their intended range
# TODO: Raise warninings to a log file to be aware when the loss model is outside its range of validity and because of which variables
# TODO: the log object can be passed as an additional variable to the functions with a default parameter Logger=None
# TODO: Whenever something is out of range we can do "if Logger: " to check if the logger was actually passed as an input and log an descriptive message

# TODO: Write detailed docstring in .rst format explaning the equations of the loss model?
# TODO: This could be prepared in latex and then transformed into .rst with chatGPT

# TODO: For all functions:
# TODO: add docstring explaning the equations and the original paper
# TODO: possibly include the equation number (or figure number) of the original paper
# TODO: explain smoothing/blending tricks
# We can sit together one day and prepare a draft of the explanation of each function with help of chatGPT
# In this way I also get the chance to understand how the loss model works in detail


def compute_losses(input_parameters):
    """Evaluate loss coefficient according to Benner (2007) model,
      but using the incidence loss from Moustapha (1997)"""

    # Extract input parameters
    flow_parameters = input_parameters["flow"]
    geometry = input_parameters["geometry"]
    options = input_parameters["loss_model"]

    # Assume angle of minimum incidence loss is equal to metal angle
    # TODO: Digitize Figure 2 from :cite:`moustapha_improved_1990`?
    beta_des = geometry["metal_angle_le"]

    # Calculate inlet displacement thickness to height ratio
    delta_height = options["inlet_displacement_thickness_height_ratio"]
    delta_height = delta_height * (flow_parameters["Re_in"] / 3e5) ** (-1 / 7)

    # Profile loss coefficient
    Y_p = get_profile_loss(flow_parameters, geometry)

    # Trailing edge coefficient
    Y_te = get_trailing_edge_loss(flow_parameters, geometry)

    # Secondary loss coefficient
    Y_s = get_secondary_loss(flow_parameters, geometry, delta_height)

    # Tip clearance loss coefficient
    Y_cl = get_tip_clearance_loss(flow_parameters, geometry)

    # Incidence loss coefficient
    Y_inc = get_incidence_loss(flow_parameters, geometry, beta_des)

    # Penetration depth to blade height ratio
    ZTE = get_penetration_depth(flow_parameters, geometry, delta_height)

    # Correct profile losses according to penetration depth
    Y_p *= 1 - ZTE
    Y_te *= 1 - ZTE
    Y_inc *= 1 - ZTE

    # Return a dictionary of loss components
    losses = {
        "loss_profile": Y_p,
        "loss_incidence": Y_inc,
        "loss_trailing": Y_te,
        "loss_secondary": Y_s,
        "loss_clearance": Y_cl,
        "loss_total": Y_p + Y_te + Y_inc + Y_s + Y_cl,
    }

    return losses


def get_profile_loss(flow_parameters, geometry):
    r"""
    Calculate the profile loss coefficient for the current cascade using the Kacker and Okapuu loss model.
    The equation for :math:`Y_p` is given by:

    .. math::

        Y_p = Yp_{reaction} - \left|\frac{\theta_{in}}{\beta_{out}}\right| \cdot
        \left(\frac{\theta_{in}}{\beta_{out}}\right) \cdot (Yp_{impulse} - Yp_{reaction})

    where:
        - :math:`Yp_{reaction}` is the reaction loss coefficient computed using Aungier correlation.
        - :math:`Yp_{impulse}` is the impulse loss coefficient computed using Aungier correlation.
        - :math:`\\theta_{in}` is the inlet metal angle.
        - :math:`\\beta_{out}` is the exit flow angle.

    The function also applies various corrections based on flow parameters and geometry factors.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
            - "Re_out" (float) : Reynolds number at the outlet.
            - "Ma_rel_out" (float) : Exit relative Mach number.
            - "Ma_rel_in" (float) : Inlet relative Mach number.
            - "p0_rel_in" (float) : Inlet total relative pressure.
            - "p_in" (float) : Inlet static pressure.
            - "p0_rel_out" (float) : Exit total relative pressure.
            - "p_out" (float) : Exit static pressure.
            - "beta_out" (float) : Exit relative flow angle.

    geometry : dict
        Dictionary with geometric parameters.
            - "hub_tip_ratio_in" (float) : Hub to tip radius ratio at inlet.
            - "pitch" (float) : Pitch.
            - "chord" (float) : Chord length.
            - "metal_angle_le" (float) : Inlet metal angle.
            - "thickness_max" (float) : Maximum thickness.

    cascade_type : str
        Type of cascade ('stator' or 'rotor').

    Returns
    -------
    float
        Profile loss coefficient (:math:`Y_p`).

    """

    # TODO: explain smoothing/blending tricks

    # Load data
    Re = flow_parameters["Re_out"]
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    Ma_rel_in = flow_parameters["Ma_rel_in"]
    p0rel_in = flow_parameters["p0_rel_in"]
    p_in = flow_parameters["p_in"]
    p0rel_out = flow_parameters["p0_rel_out"]
    p_out = flow_parameters["p_out"]
    beta_out = flow_parameters["beta_out"]

    r_ht_in = geometry["hub_tip_ratio_in"]
    s = geometry["pitch"]
    c = geometry["chord"]
    theta_in = geometry["metal_angle_le"]
    t_max = geometry["thickness_max"]
    cascade_type = geometry["cascade_type"]

    # Reynolds number correction factor
    f_Re = (
        (Re / 2e5) ** (-0.4) * (Re < 2e5)
        + 1 * (Re >= 2e5 and Re <= 1e6)
        + (Re / 1e6) ** (-0.2) * (Re > 1e6)
    )

    # Mach number correction factor
    f_Ma = 1 + 60 * (Ma_rel_out - 1) ** 2 * (
        Ma_rel_out > 1
    )  ## TODO smoothing / mach crit

    # Compute losses related to shock effects at the inlet of the cascade
    f_hub = get_hub_to_mean_mach_ratio(r_ht_in, cascade_type)
    a = max(0, f_hub * Ma_rel_in - 0.4)  # TODO: smoothing
    Y_shock = 0.75 * a**1.75 * r_ht_in * (p0rel_in - p_in) / (p0rel_out - p_out)
    Y_shock = max(0, Y_shock)  # TODO: smoothing

    # Compute compressible flow correction factors
    Kp, K2, K1 = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)

    # Yp_reaction and Yp_impulse according to Aungier correlation
    # These formulas are valid for 40<abs(angle_out)<80
    # Extrapolating outside of this limits might give completely wrong results
    # If the optimization algorithm has upper and lower bounds for the outlet
    # angle there is no need to worry about this problem
    # angle_out_bis keeps the 40deg-losses for outlet angles lower than 40deg
    angle_out_bis = max(abs(beta_out), 40)
    Yp_reaction = nozzle_blades(s / c, angle_out_bis)
    Yp_impulse = impulse_blades(s / c, angle_out_bis)

    # Formula according to Kacker-Okapuu
    Y_p = Yp_reaction - abs(theta_in / beta_out) * (theta_in / beta_out) * (
        Yp_impulse - Yp_reaction
    )

    # Limit the extrapolation of the profile loss to avoid negative values for
    # blade profiles with little deflection
    # Low limit to 80% of the axial entry nozzle profile loss
    # This value is completely arbitrary
    Y_p = max(Y_p, 0.8 * Yp_reaction)  # TODO: smoothing

    # Avoid unphysical effect on the thickness by defining the variable aa
    aa = max(0, -theta_in / beta_out)  # TODO: smoothing
    Y_p = Y_p * ((t_max / c) / 0.2) ** aa
    Y_p = 0.914 * (2 / 3 * Y_p * Kp + Y_shock)

    # Corrected profile loss coefficient
    Y_p = f_Re * f_Ma * Y_p

    return Y_p


def get_trailing_edge_loss(flow_parameters, geometry):
    r"""
    Calculate the trailing edge loss coefficient using the Kacker-Okapuu model.
    The main equation for the kinetic-energy coefficient is given by:

    .. math::

        d_{\phi^2} = d_{\phi^2_{reaction}} - \left|\frac{\theta_{in}}{\beta_{out}}\right| \cdot
        \left(\frac{\theta_{in}}{\beta_{out}}\right) \cdot (d_{\phi^2_{impulse}} - d_{\phi^2_{reaction}})

    The kinetic-energy coefficient is converted to the total pressure loss coefficient by:

    .. math::

        Y_{te} = \frac{1}{{1 - \phi^2}} - 1

    where:
        - :math:`d_{\phi^2_{reaction}}` and :math:`d_{\phi^2_{impulse}}` are coefficients related to kinetic energy loss for reaction and impulse blades respectively, and are interpolated based on trailing edge to throat opening ratio (r_to).
        - :math:`\beta_{out}` is the exit flow angle.
        - :math:`\theta_{in}` is the inlet metal angle.

    The function also applies various interpolations and computations based on flow parameters and geometry factors.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
            - "beta_out" (float) : Exit flow angle.

    geometry : dict
        Dictionary with geometric parameters.
            - "thickness_te" (float) : Trailing edge thickness.
            - "opening" (float) : Throat width.
            - "metal_angle_le" (float) : Inlet metal angle.

    Returns
    -------
    float
        Trailing edge loss coefficient (:math:`Y_{te}`).


    """

    t_te = geometry["thickness_te"]
    o = geometry["opening"]
    angle_in = geometry["metal_angle_le"]
    angle_out = flow_parameters["beta_out"]

    # Range of trailing edge to throat opening ratio
    r_to_data = [0, 0.2, 0.4]

    # Reacting blading
    phi_data_reaction = [0, 0.045, 0.15]

    # Impulse blading
    phi_data_impulse = [0, 0.025, 0.075]

    # Numerical trick to avoid too big r_to's
    r_to = min(0.4, t_te / o)  # TODO: smoothing

    # Interpolate data
    d_phi2_reaction = np.interp(r_to, r_to_data, phi_data_reaction)
    d_phi2_impulse = np.interp(r_to, r_to_data, phi_data_impulse)

    # Compute kinetic energy loss coefficient
    d_phi2 = d_phi2_reaction - abs(angle_in / angle_out) * (angle_in / angle_out) * (
        d_phi2_impulse - d_phi2_reaction
    )

    # Limit the extrapolation of the trailing edge loss
    d_phi2 = max(d_phi2, d_phi2_impulse / 2)  # TODO: smoothing
    Y_te = 1 / (1 - d_phi2) - 1

    return Y_te


def get_secondary_loss(flow_parameters, geometry, delta_height):
    # TODO: add docstring explaning the equations and the original paper
    # TODO: possibly include the equation number (or figure number) of the original paper
    # TODO: explain smoothing/blending tricks

    beta_in = flow_parameters["beta_in"]
    beta_out = flow_parameters["beta_out"]
    height = geometry["height"]
    chord = geometry["chord"]
    stagger = geometry["stagger_angle"]

    AR = height / chord  # Aspect ratio
    CR = math.cosd(beta_in) / math.cosd(
        beta_out
    )  # Convergence ratio from Benner et al.[2006]

    if AR <= 2:  # TODO: sigmoid blending to convert to smooth piecewise function
        denom = (
            np.sqrt(math.cosd(stagger))
            * CR
            * AR**0.55
            * (math.cosd(beta_out) / (math.cosd(stagger))) ** 0.55
        )
        Y_sec = (0.038 + 0.41 * np.tanh(1.2 * delta_height)) / denom
    else:
        denom = (
            np.sqrt(math.cosd(stagger))
            * CR
            * AR
            * (math.cosd(beta_out) / (math.cosd(stagger))) ** 0.55
        )
        Y_sec = (0.052 + 0.56 * np.tanh(1.2 * delta_height)) / denom

    return Y_sec


def get_tip_clearance_loss(flow_parameters, geometry):
    r"""
    Calculate the tip clearance loss coefficient for the current cascade using the Kacker and Okapuu loss model.
    The equation for the tip clearance loss coefficent is given by:

    .. math::

        Y_{cl} = B \cdot Z \cdot \frac{c}{H} \cdot \left(\frac{t_{cl}}{H}\right)^{0.78}

    where:
        - :math:`B` is an empirical parameter that depends on the type of cascade (0 for stator, 0.37 for shrouded rotor).
        - :math:`Z` is a blade loading parameter
        - :math:`c` is the chord length.
        - :math:`H` is the blade height.
        - :math:`t_{cl}` is the tip clearance.

    The function also applies various computations and corrections based on flow parameters, geometry, and cascade type.


    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
            - "beta_out" (float) : Exit flow angle.
            - "beta_in" (float) : Inlet flow angle.

    geometry : dict
        Dictionary with geometric parameters.
            - "height" (float) : Blade height.
            - "chord" (float) : Chord length.
            - "tip_clearance" (float) : Tip clearance.

    cascade_type : str
        Type of cascade ('stator' or 'rotor').

    Returns
    -------
    float
        Tip clearance loss coefficient (Y_cl).


    """

    beta_out = flow_parameters["beta_out"]
    beta_in = flow_parameters["beta_in"]

    H = geometry["height"]
    c = geometry["chord"]
    t_cl = geometry["tip_clearance"]
    cascade_type = geometry["cascade_type"]

    # Calculate blade loading parameter Z
    angle_m = math.arctand((math.tand(beta_in) + math.tand(beta_out)) / 2)
    Z = (
        4
        * (math.tand(beta_in) - math.tand(beta_out)) ** 2
        * math.cosd(beta_out) ** 2
        / math.cosd(angle_m)
    )

    # Empirical parameter (0 for stator, 0.37 for shrouded rotor)
    if cascade_type == "stator":
        B = 0
    elif cascade_type == "rotor":
        B = 0.37
    else:
        print("Specify the type of cascade")

    # Tip clearance loss coefficient
    Y_cl = B * Z * c / H * (t_cl / H) ** 0.78

    return Y_cl


def get_incidence_loss(flow_parameters, geometry, beta_des):
    r"""
    Calculate the total pressure loss coefficient due to incidence in a cascade according to :cite'moustapha_incidence_1990'

    The model first computes the incidence parameter, :math:`\chi`. Based on this parameter, the kinetic-energy incidence
    loss coefficient is calculated to match experimental data.

    For 0 <= :math:`\chi` <= 800:

    .. math::

        \Delta\phi^2_p = 0.778e^{-5}*\chi + 0.56e^{-7}*\chi^2 + 0.4e^{-10}*\chi^3 + 2.054e^{-19}*\chi^6

    For -800 <= :math:`\chi` < 0:

    .. math::

        \Delta\phi^2_p = -5.1734e^{-6}*\chi + 7.6902e^{-9}*\chi^2

    The kinetic-energy coefficient is converted to total pressure loss coefficient (:math:`Y`) using the formula:

    .. math::

        Y = \frac{[1 - \frac{\gamma-1}{2}M_2^2 (\frac{1}{\phi^2} - 1)]^{\frac{-\gamma}{\gamma-1}} - 1}{1 - (1 + \frac{\gamma-1}{2} M_2^2)^{\frac{-\gamma}{\gamma-1}}}


    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing kinetic and thermodynamic variables describing the flow through the current cascade.
            - "Ma_rel_out" (float) : Exit relative Mach number.
            - "beta_in" (float) : Inlet relative flow angle.
            - "gamma" (float) : Specific heat ratio.

    geometry : dict
        Dictionary with the geometrical parameters describing the current cascade.
            - "pitch" (float) : Pitch.
            - "le" (float) : Leading edge diameter.
            - "metal_angle_le" (float) : Inlet metal angle.
            - "metal_angle_te" (float) : Exit metal angle.

    beta_des : float
        Desired beta value.

    Returns
    -------
    float
        Total pressure loss coefficient (:math:`Y_\mathrm{inc}`).

    Raises
    ------
    Warning
        If incidence parameter (:math:`chi`) is out of range.

    """

    Ma_rel_out = flow_parameters["Ma_rel_out"]  # Exit relative mach number
    beta_in = flow_parameters["beta_in"]  # Inlet relativ flow angle
    gamma = flow_parameters["gamma_out"]  # Specific heat ratio

    s = geometry["pitch"]  # Pitch
    le = geometry["diameter_le"]  # Leading edge diameter
    theta_in = geometry["metal_angle_le"]  # Inlet metal angle
    theta_out = geometry["metal_angle_te"]  # Exit metal angle

    # Compute incidence parameter
    chi = get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des)

    # Check if incidence parameter is within the range of the experimental data
    if abs(chi) > 800:
        raise Warning("Incidence parameter out of range: chi = {chi}")

    # Calculate kinetic-energy loss coefficient
    if chi >= 0:
        dPhi = (
            0.778e-5 * chi
            + 0.56e-7 * chi**2
            + 0.4e-10 * chi**3
            + 2.054e-19 * chi**6
        )
    elif chi < 0:
        dPhi = -5.1734e-6 * chi + 7.6902e-9 * chi**2

    # Convert kinetic-energy loss coefficient to total pressure loss coeffcient
    Y_inc = convert_kinetic_energy_coefficient(dPhi, gamma, Ma_rel_out)

    return Y_inc


def get_hub_to_mean_mach_ratio(r_ht, cascade_type):
    # Compute the ratio between mach at hub and mean span at the inlet of the current cascade
    # Due to radial variation in gas conditions, mach at the hub will alway be higher than at mean
    # Thus, shock losses at the hub could occur even when the mach is subsonic at the mean balde span

    # r_ht is the hub to tip ratio at the inlet of the current cascade
    # cascade_type is a string that specify whether the current cascade is a stator or a rotor

    if r_ht < 0.5:  # TODO: add smoothing, this is essentially a max(r_ht, 0.5)
        r_ht = 0.5  # Numerical trick to prevent extrapolation

    r_ht_data = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Stator curve
    f_data_S = [1.4, 1.18, 1.05, 1.0, 1.0, 1.0]

    # Rotor curve
    f_data_R = [2.15, 1.7, 1.35, 1.12, 1.0, 1.0]

    if cascade_type == "stator":
        f = np.interp(r_ht, r_ht_data, f_data_S)
    elif cascade_type == "rotor":
        f = np.interp(r_ht, r_ht_data, f_data_R)
    else:
        print("Specify the type of cascade")

    return f


def get_compressible_correction_factors(Ma_rel_in, Ma_rel_out):
    # Compute compressible flow correction factor according to Kacker and Okapuu loss model
    # The loss correlation proposed by ainley ant Mathieson overpredicts losses at high mach numbers
    # These correction factors reduces losses accordingly at higher mach numbers

    # TODO: add docstring explaning the equations and the original paper
    # TODO: possibly include the equation number (or figure number) of the original paper
    # TODO: explain smoothing/blending tricks

    K1 = 1 * (Ma_rel_out < 0.2) + (1 - 1.25 * (Ma_rel_out - 0.2)) * (
        Ma_rel_out > 0.2 and Ma_rel_out < 1.00
    )  # TODO: this can be converted to a smooth piecewise function (sigmoid blending)
    K2 = (Ma_rel_in / Ma_rel_out) ** 2
    Kp = 1 - K2 * (1 - K1)
    Kp = max(0.1, Kp)  # TODO: smoothing
    return [Kp, K2, K1]


def nozzle_blades(r_sc, angle_out):
    # Use Aungier correlation to compute the pressure loss coefficient
    # This correlation is a formula that reproduces the figures from the Ainley
    # and Mathieson original figures

    phi = 90 - angle_out
    r_sc_min = (0.46 + phi / 77) * (phi < 30) + (0.614 + phi / 130) * (
        phi >= 30
    )  # TODO: sigmoid blending to convert to smooth piecewise function
    X = r_sc - r_sc_min
    A = (0.025 + (27 - phi) / 530) * (phi < 27) + (0.025 + (27 - phi) / 3085) * (
        phi >= 27
    )  # TODO: sigmoid blending to convert to smooth piecewise function
    B = 0.1583 - phi / 1640
    C = 0.08 * ((phi / 30) ** 2 - 1)
    n = 1 + phi / 30
    Yp_reaction = (A + B * X**2 + C * X**3) * (phi < 30) + (A + B * abs(X) ** n) * (
        phi >= 30
    )  # TODO: sigmoid blending to convert to smooth piecewise function
    return Yp_reaction


def impulse_blades(r_sc, angle_out):
    # Use Aungier correlation to compute the pressure loss coefficient
    # This correlation is a formula that reproduces the figures from the Ainley
    # and Mathieson original figures

    phi = 90 - angle_out
    r_sc_min = 0.224 + 1.575 * (phi / 90) - (phi / 90) ** 2
    X = r_sc - r_sc_min
    A = 0.242 - phi / 151 + (phi / 127) ** 2
    B = (0.3 + (30 - phi) / 50) * (phi < 30) + (0.3 + (30 - phi) / 275) * (
        phi >= 30
    )  # TODO: sigmoid blending to convert to smooth piecewise function
    C = 0.88 - phi / 42.4 + (phi / 72.8) ** 2
    Yp_impulse = A + B * X**2 - C * X**3
    return Yp_impulse


def get_penetration_depth(flow_parameters, geometry, delta_height):
    """Return the penetration depth to blade heigh ratio"""

    # TODO: add docstring explaning the equations and the original paper
    # TODO: possibly include the equation number (or figure number) of the original paper
    # TODO: explain smoothing/blending tricks

    beta_in = flow_parameters["beta_in"]
    beta_out = flow_parameters["beta_out"]
    axial_chord = geometry["axial_chord"]
    pitch = geometry["pitch"]
    chord = geometry["chord"]
    height = geometry["height"]

    CR = math.cosd(beta_in) / math.cosd(
        beta_out
    )  # Convergence ratio from Benner et al.[2006]

    BSx = axial_chord / pitch  # Axial blade solidity
    delta_height = (
        delta_height  # Boundary layer displacement thickness relative to blade height
    )
    AR = height / chord  # Aspect ratio

    Ft = F_t(BSx, beta_in, beta_out)

    Z_TE = 0.10 * Ft**0.79 / np.sqrt(CR) / (AR) ** 0.55 + 32.70 * delta_height**2

    Z_TE = min(Z_TE, 0.99)  # TODO: smoothing

    return Z_TE


def get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des):
    # Computes the incidence parameters according to :cite'moustapha_incidence_1990'
    # Depends on the leading edge diamater (le), pitch (s), inelt metal angle (theta_in)
    # exit metal_angle (theta_out), inlet relative flow angle (beta_in) and design inlet relative flow angle (beta_des)

    # The parameter is calculated by the following formula:

    # .. math::

    #    \chi = \left(\frac{d}{s}\right)^{-1.6}  \left(\frac{\cos{\beta_1}}{\cos{\beta_2}} \right)^{-2}  \left(\alpha_1 - \alpha_{1,\mathrm{des}}\right)

    # The leading edge to pitch ratio :math:`\left(\frac{d}{s}\right)` reduced the incidence losses when increased.
    # The term :math:`left(\frac{\cos{\beta_1}}{\cos{\beta_2}} \right)` attempt to describe the degree of reaction through the flow section, and reduces the incidence losses when increased
    # The term :math:`\left(\alpha_1 - \alpha_{1,\mathrm{des}}\right)` is the incidence angle

    # TODO: add docstring explaning the equations and the original paper
    # TODO: possibly include the equation number (or figure number) of the original paper
    # TODO: explain smoothing/blending tricks

    chi = (
        (le / s) ** (-1.6)
        * (math.cosd(theta_in) / math.cosd(theta_out)) ** (-2)
        * (abs(beta_in) - abs(beta_des))
    )
    return chi


def convert_kinetic_energy_coefficient(dPhi, gamma, Ma_rel_out):
    # TODO: this conversion assumes that the fluid is a perfect gas
    # TODO: we can create a more general function to convert between different loss coefficient deffinitions
    #   TODO: stagation pressure loss coefficient
    #   TODO: enthalpy loss coefficient (two-variants)
    #   TODO: kinetic energy loss coefficient
    #   TODO: entropy loss coefficient
    # Given the fluid state, I believe it is possible to convert between all of them

    denom = 1 - (1 + (gamma - 1) / 2 * Ma_rel_out**2) ** (-gamma / (gamma - 1))
    numer = (1 - (gamma - 1) / 2 * Ma_rel_out**2 * (1 / (1 - dPhi) - 1)) ** (
        -gamma / (gamma - 1)
    ) - 1

    Y = numer / denom

    return Y


def F_t(BSx, beta_in, beta_out):
    # TODO: add docstring explaning the equations and the original paper
    # TODO: possibly include the equation number (or figure number) of the original paper
    # TODO: explain smoothing/blending tricks

    a_m = math.arctand(0.5 * (math.tand(beta_in) + math.tand(beta_out)))

    F_t = (
        2
        * 1
        / BSx
        * math.cosd(a_m) ** 2
        * (abs(math.tand(beta_in)) + abs(math.tand(beta_out)))
    )

    return F_t


if __name__ == "__main__":
    We = 31.05
    le = 0.081e-2
    s = 1.524e-2
    theta_in = 29.6
    theta_out = -61.6
    beta_in = 50
    beta_des = theta_in
    chi = get_incidence_parameter(le, s, We, theta_in, theta_out, beta_in, beta_des)
    print(chi)
    print(beta_in)
