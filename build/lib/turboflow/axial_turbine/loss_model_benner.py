import numpy as np
from .. import math

# TODO: Add smoothing to the min/max/abs/piecewise functions

# TODO: Create test scripts to plot how the different loss model functions behave when smoothing is applied / verify that the behavior is reasonable beyond their intended range
# TODO: Raise warninings to a log file to be aware when the loss model is outside its range of validity and because of which variables
# TODO: the log object can be passed as an additional variable to the functions with a default parameter Logger=None
# TODO: Whenever something is out of range we can do "if Logger: " to check if the logger was actually passed as an input and log an descriptive message

# TODO: Write detailed docstring in .rst format explaning the equations of the loss model?
# TODO: This could be prepared in latex and then transformed into .rst with chatGPT

# TODO: For all functions:
# TODO: add docstring explaning the equations and the original paper
# TODO: explain smoothing/blending tricks
# We can sit together one day and prepare a draft of the explanation of each function with help of chatGPT
# In this way I also get the chance to understand how the loss model works in detail


def compute_losses(input_parameters):
    r"""
    
    Evaluate loss coefficient according to :cite:`benner_influence_1997`, :cite:`benner_empirical_2006-1` and :cite:`benner_empirical_2006`. 

    This model split the total loss coefficient into profile, secondary, trailing edge, tip clearance and incidence losses.
    The loss coefficient are combined through the following relation:

    .. math::

        \mathrm{Y_{tot}} = (\mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{inc}})(1-Z/H) + \mathrm{Y_{s}} + \mathrm{Y_{cl}}

    where :math:`Z` is the penetration depth of the passage vortex separation line in spanwise direction, and :math:`H` is the mean blade height. This quantity depend on the displacement thickness of the
    inlet endwall bounary layer, which is approximated from a reference value:

    .. math::

        \frac{\delta}{H} = \frac{\delta}{H}_\mathrm{ref} \frac{\mathrm{Re_{in}}}{3\cdot10^5}^{-1/7}

    The reference value may be specified by the user. 

    The function calls a function for each loss component:

        - :math:`\mathrm{Y_{p}}` : `get_profile_loss`
        - :math:`\mathrm{Y_{te}}` : `get_trailing_edge_loss`
        - :math:`\mathrm{Y_{s}}` : `get_secondary_loss`
        - :math:`\mathrm{Y_{cl}}` : `get_tip_clearance_loss`
        - :math:`\mathrm{Y_{inc}}` : `get_incidence_loss`
        - :math:`Z/H` : `get_penetration_depth`


    Parameters
    ----------
    input_parameters : dict
        A dictionary containing containing the keys, `flow`, `geometry` and `options`.

    Returns
    -------
    dict
        A dictionary with the loss components.
    
    """

    # Extract input parameters
    flow_parameters = input_parameters["flow"]
    geometry = input_parameters["geometry"]
    options = input_parameters["loss_model"]

    # Assume angle of minimum incidence loss is equal to metal angle
    # TODO: Digitize Figure 2 from :cite:`moustapha_improved_1990`?
    beta_des = geometry["leading_edge_angle"]

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
    The equation for :math:`\mathrm{Y_p}` is given by:

    .. math::

        \mathrm{Y_p} = \mathrm{Y_{reaction}} - \left|\frac{\theta_\mathrm{in}}{\beta_\mathrm{out}}\right| \cdot
        \left(\frac{\theta_\mathrm{in}}{\beta_\mathrm{out}}\right) \cdot (\mathrm{Y_{impulse}} - \mathrm{Y_{reaction}})

    where:

        - :math:`\mathrm{Y_{reaction}}` is the reaction loss coefficient computed using Aungier correlation.
        - :math:`\mathrm{Y_{impulse}}` is the impulse loss coefficient computed using Aungier correlation.
        - :math:`\theta_\mathrm{in}` is the inlet metal angle.
        - :math:`\beta_\mathrm{out}` is the exit flow angle.

    The function also applies various corrections based on flow parameters and geometry factors.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.

    Returns
    -------
    float
        Profile loss coefficient.

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
    theta_in = geometry["leading_edge_angle"]
    t_max = geometry["maximum_thickness"]
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

        d_{\phi^2} = d_{\phi^2_\mathrm{reaction}} - \left|\frac{\theta_\mathrm{in}}{\beta_\mathrm{out}}\right| \cdot
        \left(\frac{\theta_\mathrm{in}}{\beta_\mathrm{out}}\right) \cdot (d_{\phi^2_\mathrm{impulse}} - d_{\phi^2_\mathrm{reaction}})

    The kinetic-energy coefficient is converted to the total pressure loss coefficient by:

    .. math::

        \mathrm{Y_{te}} = \frac{1}{{1 - \phi^2}} - 1

    where:

        - :math:`d_{\phi^2_\mathrm{reaction}}` and :math:`d_{\phi^2_\mathrm{impulse}}` are coefficients related to kinetic energy loss for reaction and impulse blades respectively, and are interpolated based on trailing edge to throat opening ratio.
        - :math:`\beta_\mathrm{out}` is the exit flow angle.
        - :math:`\theta_\mathrm{in}` is the inlet metal angle.

    The function also applies various interpolations and computations based on flow parameters and geometry factors.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.

    Returns
    -------
    float
        Trailing edge loss coefficient.

    """

    t_te = geometry["trailing_edge_thickness"]
    o = geometry["opening"]
    angle_in = geometry["leading_edge_angle"]
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
    r"""
    Calculate the secondary loss coefficient.

    The correlation takes two different forms depending on the aspect ratio:

    If aspect ratio :math:`\leq 2.0`

    .. math::

        \mathrm{Y_{s}} = \frac{0.038 + 0.41 \tanh(1.20\delta^*/H)}{\sqrt{\cos(\xi)}CR(H/c)^{0.55}\frac{\cos(\beta_\mathrm{out})}{\cos(\xi)}^{0.55}}

    wheras if aspect ratio :math:`>2.0`

    .. math::

        \mathrm{Y_{s}} = \frac{0.052 + 0.56 \tanh(1.20\delta^*/H)}{\sqrt{\cos(\xi)}CR(H/c)\frac{\cos(\beta_\mathrm{out})}{\cos(\xi)}^{0.55}}

    where:

        - :math:`CR = \frac{\cos(\beta_\mathrm{in})}{\cos(\beta_\mathrm{out})}` is the convergence ratio.  
        - :math:`H` is the mean height.
        - :math:`c` is the chord.
        - :math:`\delta^*` is the inlet endwall boundary layer displacement thickness.
        - :math:`\xi` is the stagger angle. 
        - :math:`\beta_\mathrm{out}` is the exit relative flow angle.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.

    Returns
    -------
    float
        Secondary loss coefficient.

    """

    beta_in = flow_parameters["beta_in"]
    beta_out = flow_parameters["beta_out"]
    height = geometry["height"]
    chord = geometry["chord"]
    stagger = geometry["stagger_angle"]

    AR = height / chord  
    CR = math.cosd(beta_in) / math.cosd(
        beta_out
    )  
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

        \mathrm{Y_{cl}} = B \cdot Z \cdot \frac{c}{H} \cdot \left(\frac{t_\mathrm{cl}}{H}\right)^{0.78}

    where:

        - :math:`B` is an empirical parameter that depends on the type of cascade (0 for stator, 0.37 for shrouded rotor).
        - :math:`Z` is a blade loading parameter
        - :math:`c` is the chord.
        - :math:`H` is the mean blade height.
        - :math:`t_\mathrm{cl}` is the tip clearance.


    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.

    Returns
    -------
    float
        Tip clearance loss coefficient.
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
    Calculate the incidence loss coefficient according to the correlation proposed by :cite:`benner_influence_1997`.
    
    This model calculates the incidence loss parameter through the function `get_incidence_parameter`. The kinetic energu coefficient
    can be obtained from `get_incidence_profile_loss_increment`, which is a function of the incidence parameter. The kinetic energy loss coefficient is converted
    to the total pressure loss coefficient through `convert_kinetic_energy_coefficient`.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.
    beta_des : float
        Inlet flow angle for zero incidence losses (design).

    Returns
    -------
    float
        Incidence loss coefficient.
    """

    beta_in = flow_parameters["beta_in"]
    gamma = flow_parameters["gamma_out"]
    Ma_rel_out = flow_parameters["Ma_rel_out"]

    le = geometry["leading_edge_diameter"]
    s = geometry["pitch"]
    We = geometry["leading_edge_wedge_angle"]
    theta_in = geometry["leading_edge_angle"]
    theta_out = math.arccosd(geometry["A_throat"]/geometry["A_out"])

    chi = get_incidence_parameter(le, s, We, theta_in, theta_out, beta_in, beta_des)

    dPhip = get_incidence_profile_loss_increment(chi)

    Y_inc = convert_kinetic_energy_coefficient(dPhip, gamma, Ma_rel_out)

    return Y_inc


def get_hub_to_mean_mach_ratio(r_ht, cascade_type):

    r"""
    Compute the ratio between Mach at the hub and mean span at the inlet of the current cascade.

    Due to radial variation in gas conditions, Mach at the hub will always be higher than at the mean.
    Thus, shock losses at the hub could occur even when the Mach is subsonic at the mean blade span. 

    Parameters
    ----------
    r_ht : float
        Hub to tip ratio at the inlet of the current cascade.
    cascade_type : str
        Type of the current cascade, either 'stator' or 'rotor'.

    Returns
    -------
    float
        Ratio between Mach at the hub and mean span at the inlet of the current cascade.

    """

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
    r"""

    Compute compressible flow correction factor according to Kacker and Okapuu loss model :cite:`kacker_mean_1982`.

    The correction factors :math:`\mathrm{K_1}`, :math:`\mathrm{K_2}` and :math:`\mathrm{K_p}` was introduced by :cite:`kacker_mean_1982` to correct previous correlation (:cite:`ainley_method_1951`)
    for effect of higher mach number and channel acceleration. The correction factors reduces the losses at higher mach number. Their definition follows: 

    .. math::

        &\mathrm{K_1} = \begin{cases}
                1 & \text{if } \mathrm{Ma_{out}} < 0.2 \\
                1 - 1.25(\mathrm{Ma_{out}} - 0.2) & \text{if } \mathrm{Ma_{out}} \geq 0.2
              \end{cases} \\
        &\mathrm{K_2} = \left(\frac{\mathrm{Ma_{in}}}{\mathrm{Ma_{out}}}\right) \\
        &\mathrm{K_p} = 1 - \mathrm{K_2}(1-\mathrm{K_1})

    where:

        - :math:`\mathrm{Ma_{in}}` is the relative mach number at the cascade inlet.
        -  :math:`\mathrm{Ma_{out}}` is the relative mach number at the cascade exit.
    
    Parameters
    ----------
    Ma_rel_in : float
        Inlet relative mach number.
    Ma_rel_out : float
        Exit relative mach number.

    Returns
    -------
    float
        Correction factor :math:`\mathrm{K_p}`.
    float
        Correction factor :math:`\mathrm{K_2}`.
    float
        Correction factor :math:`\mathrm{K_1}`.
    """

    K1 = 1 * (Ma_rel_out < 0.2) + (1 - 1.25 * (Ma_rel_out - 0.2)) * (
        Ma_rel_out > 0.2 and Ma_rel_out < 1.00
    )  # TODO: this can be converted to a smooth piecewise function (sigmoid blending)
    K2 = (Ma_rel_in / Ma_rel_out) ** 2
    Kp = 1 - K2 * (1 - K1)
    Kp = max(0.1, Kp)  # TODO: smoothing
    return [Kp, K2, K1]


def nozzle_blades(r_sc, angle_out):

    r"""
    Use Aungier correlation to compute the pressure loss coefficient for nozzle blades :cite:`aungier_turbine_2006`.

    This correlation is a formula that reproduces the figures from the Ainley and Mathieson original figures :cite:`ainley_method_1951`,
    and is a function of the pitch-to-chord ratio and exit relative flow angle. 

    The correlation uses the following equations:

    .. math::

        & \beta_\mathrm{tan} = 90 - \beta_\mathrm{ax} \\
        & \left(\frac{s}{c}\right)_\mathrm{min} = \begin{cases}
                                                    0.46 + \frac{\beta_\mathrm{tan}}{77} && \text{if } \beta_\mathrm{tan} < 30 \\
                                                    0.614 + \frac{\beta_\mathrm{tan}}{130} && \text{if } \beta_\mathrm{tan} \geq 30
                                                \end{cases} \\
        & X = \left(\frac{s}{c}\right) - \left(\frac{s}{c}\right)_\mathrm{min} \\
        & A = \begin{cases}
                0.025 + \frac{27 - \beta_\mathrm{tan}}{530} && \text{if } \beta_\mathrm{tan} < 27 \\
                0.025 + \frac{27 - \beta_\mathrm{tan}}{3085} && \text{if } \beta_\mathrm{tan} \geq 27
              \end{cases} \\
        & B = 0.1583 - \frac{\beta_\mathrm{tan}}{1640} \\
        & C = 0.08\left(\frac{\beta_\mathrm{tan}}{30}\right)^2 -1 \\
        & n = 1 + \frac{\beta_\mathrm{tan}}{30} \\
        & \mathrm{Y_{p,reaction}} = \begin{cases}
                A + BX^2 + CX^3 && \text{if } \beta_\mathrm{tan} < 30 \\
                A + B |X|^n && \text{if } \beta_\mathrm{tan} \geq 30
              \end{cases}   

    where:

        - :math:`s` is the pitch
        - :math:`c` is the chord
        - :math:`\beta_\mathrm{tan}` and :math:`\beta_\mathrm{ax}` is the exit relative flow angle with respect to tangential and axial direction. 
        
    Parameters
    ----------
    r_sc : float
        Pitch-to-chord ratio.
    angle_out : float
        Exit relative flow angle (in degrees).

    Returns
    -------
    float
        Pressure loss coefficient for impulse blades.

    """

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

    r"""
    Use Aungier correlation to compute the pressure loss coefficient for impulse blades :cite:`aungier_turbine_2006`.

    This correlation is a formula that reproduces the figures from the Ainley and Mathieson original figures :cite:`ainley_method_1951`,
    and is a function of the pitch-to-chord ratio and exit relative flow angle. 

    The correlation uses the following equations:

    .. math::

        & \beta_\mathrm{tan} = 90 - \beta_\mathrm{ax} \\
        & \left(\frac{s}{c}\right)_\mathrm{min} = 0.224 + 1.575\left(\frac{\beta_\mathrm{tan}}{90}\right) - \left(\frac{\beta_\mathrm{tan}}{90}\right)^2 \\
        & X = \left(\frac{s}{c}\right) - \left(\frac{s}{c}\right)_\mathrm{min} \\
        & A = 0.242 - \frac{\beta_\mathrm{tan}}{151} + \left(\frac{\beta_\mathrm{tan}}{127}\right)^2 \\
        & B = \begin{cases}
                0.3 + \frac{30 - \beta_\mathrm{tan}}{50} && \text{if } \beta_\mathrm{tan} < 30 \\
                0.3 + \frac{30 - \beta_\mathrm{tan}}{275} && \text{if } \beta_\mathrm{tan} \geq 30
              \end{cases} \\
        & C = 0.88 - \frac{\beta_\mathrm{tan}}{42.4} + \left(\frac{\beta_\mathrm{tan}}{72.8}\right)^2 \\
        & \mathrm{Y_{p,impulse}} = A + BX^2 - CX^3 

    where:

        - :math:`s` is the pitch
        - :math:`c` is the chord
        - :math:`\beta_\mathrm{tan}` and :math:`\beta_\mathrm{ax}` is the exit relative flow angle with respect to tangential and axial direction. 
        
    Parameters
    ----------
    r_sc : float
        Pitch-to-chord ratio.
    angle_out : float
        Exit relative flow angle (in degrees).

    Returns
    -------
    float
        Pressure loss coefficient for impulse blades.

    """
        

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
    r"""
    Calculated the penetration depth of the passage vortex separation line relative to the blade span. 
    
    The endwall inlet boundary layer generate a vortex which propagtes through the cascade section, 
    and the spanwise penetration of this vortex affect the magnutude of the secondary loss coefficient. 
    This function approximate the vortex penetration depth to the blade span by the correlation developed by :cite:`benner_empirical_2006-1`.

    The quantity is calculated as:

    .. math::

        \frac{\mathrm{Z_{te}}}{H} = \frac{0.10F_t^{0.79}}{\sqrt{CR}\left(\frac{H}{c}\right)^{0.55}} + 32.70\frac{\delta^*}{H}^2

    where:

        - :math:`CR = \frac{\cos(\beta_\mathrm{in})}{\cos(\beta_\mathrm{out})}` is the convergence ratio.  
        - :math:`H` is the mean height.
        - :math:`c` is the chord.
        - :math:`\delta^*` is the inlet endwall boundary layer displacement thickness.
        - :math:`\beta_\mathrm{out}` is the exit relative flow angle.
        - :math:`F_t` is the tangential loading coefficient.

    The tangiential loading coefficient is calculated by a separate function (`F_t`).

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.
    delta_height : float
        The inlet endwall boundary layer displacement thickness relative to the mean blade height.

    Returns
    -------
    float
        Spanwise penetration depth of the passage vortex relative to the mean blade height.
    
    """

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


def convert_kinetic_energy_coefficient(dPhi, gamma, Ma_rel_out):

    r"""
    
    Convert the kinetic energy coefficient increment due to incidence to the total pressure loss coefficient according to the following correlation:

    .. math::

        \mathrm{Y} = \frac{\left(1-\frac{\gamma -1}{2}\mathrm{Ma_{out}}^2(\frac{1}{(1-\Delta\phi^2_p)}-1)\right)^\frac{-\gamma}{\gamma - 1}-1}{1-\left(1 + \frac{\gamma - 1}{2}\mathrm{Ma_{out}}^2\right)^\frac{-\gamma}{\gamma - 1}}

    where:

        - :math:`\gamma` is the specific heat ratio.
        - :math:`\mathrm{Ma_{out}}` is the cascade exit relative mach number.
        - :math:`\Delta\phi^2_p` is the kinetic energy loss coefficient increment due to incidence.
    
    Parameters
    ----------
    dPhi : float
        Kinetic energy coefficient increment.
    gamma : float
        Heat capacity ratio.
    Ma_rel_out : float
        The cascade exit relative mach number.

    Returns
    -------
    float
        The total pressure loss coefficient.  

    Warnings
    --------
    This conversion assumes that the fluid is a perfect gas.  

    """

    denom = 1 - (1 + (gamma - 1) / 2 * Ma_rel_out**2) ** (-gamma / (gamma - 1))
    numer = (1 - (gamma - 1) / 2 * Ma_rel_out**2 * (1 / (1 - dPhi) - 1)) ** (
        -gamma / (gamma - 1)
    ) - 1

    Y = numer / denom

    return Y


def get_incidence_profile_loss_increment(chi, chi_extrapolation=5, loss_limit=0.5):
    r"""
    Computes the increment in profile losses due to the effect of incidence according to the correlation proposed by :cite:`benner_influence_1997`.

    The increment in profile losses due to incidence is based on the incidence parameter :math:`\chi`.
    The formula used to compute :math:`\chi` is given by:

    .. math::

       \chi = We^{-0.2}\left(\frac{d}{s}\right)^{-0.05}  \left(\frac{\cos{\beta_1}}{\cos{\beta_2}} \right)^{-1.4}  \left(\alpha_1 - \alpha_{1,\mathrm{des}}\right)

    Depending on the value of :math:`\chi`, two equations are used for computing the increment in profile losses:

    1. For :math:`\chi \geq 0`:

    .. math::

        \Delta \phi_{p}^2 = \sum_{i=1}^{8} a_i \, \chi^i

    with coefficients:
    
    .. math::
    
        a_1 = -6.149 \times 10^{-5}  \\
        a_2 = +1.327 \times 10^{-3}  \\
        a_3 = -2.506 \times 10^{-4}  \\
        a_4 = -1.542 \times 10^{-4}  \\
        a_5 = +9.017 \times 10^{-5}  \\
        a_6 = +1.106 \times 10^{-5}  \\
        a_7 = -5.318 \times 10^{-6}  \\
        a_8 = +3.711 \times 10^{-7}

    2. For :math:`\chi < 0`:

    .. math::

        \Delta \phi_{p}^2 = \sum_{i=1}^{2} b_i \, \chi^i

    with coefficients:
    
    .. math::
    
        b_1 = -8.720e-4 \times 10^{-4}  \\
        b_2 = +1.358e-4 \times 10^{-4}
    
    This function also implements two safeguard methods to prevent unrealistic values of the pressure loss increment
    when the value of :math:`\chi` is outside the range of validity of the correlation

    1. Linear extrapolation beyond a certain :math:`\chi`:
        Beyond a certain value of the distance parameter :math:`\hat{\chi}` the function linearly extrapolates based on the slope of the polynomial.

    2. Loss Limiting:
        The profile loss increment is limited to a maximum value :math:`\Delta \hat{\phi}_{p}^2` to prevent potential over-predictions. The limiting mechanism is smooth to ensure that the function is differentiable.

    The final expression for the loss coefficient increment can be summarized as:

    .. math::

        \begin{align}
        \Delta \phi_{p}^2 = 
        \begin{cases}
        \sum_{i=1}^{2} b_i \, \chi^i & \text{for } \chi < 0 \\
        \sum_{i=1}^{8} a_i \, \chi^i & \text{for } 0 \leq \chi  \leq \hat{\chi} \\
        \Delta\phi_{p}^2(\hat{\chi}) + \mathrm{slope}(\hat{\chi}) \cdot(\chi - \hat{\chi}) & \text{for } \chi > \hat{\chi}
        \end{cases}
        \end{align}

    .. math::
        \Delta \phi_{p}^2 = \mathrm{smooth\,min}(\Delta \phi_{p}^2, \Delta \hat{\phi}_{p}^2)


    Parameters
    ----------
    chi : array-like
        The distance parameter, :math:`\chi`, used to compute the profile losses increment.
    chi_extrapolation : float, optional
        The value of :math:`\chi` beyond which linear extrapolation is applied. Default is 5.
    loss_limit : float, optional
        The upper limit to which the profile loss increment is smoothly constrained. Default is 0.5.

    Returns
    -------
    array-like
        Increment in profile losses due to the effect of incidence.
    """

    # Define the polynomial function for positive values of chi
    coeffs_1 = np.array(
        [
            +3.711e-7,
            -5.318e-6,
            +1.106e-5,
            +9.017e-5,
            -1.542e-4,
            -2.506e-4,
            +1.327e-3,
            -6.149e-5,
        ]
    )
    func_1 = lambda x: np.sum(
        np.asarray([a * x**i for (i, a) in enumerate(coeffs_1[::-1], start=1)]),
        axis=0,
    )

    # Define the polynomial function for negative values of chi
    coeffs_2 = np.array([1.358e-4, -8.72e-4])
    func_2 = lambda x: np.sum(
        np.asarray([b * x**i for (i, b) in enumerate(coeffs_2[::-1], start=1)]),
        axis=0,
    )

    # Calculate the profile losses increment based on polynomials
    loss_poly = np.where(chi >= 0, func_1(chi), func_2(chi))

    # Apply linear extrapolation beyond the threshold for chi
    _chi = chi_extrapolation
    loss = func_1(_chi)
    coeffs_slope = np.array([i * a for i, a in enumerate(coeffs_1[::-1], start=1)])
    slope = np.sum([a * _chi ** (i - 1) for i, a in enumerate(coeffs_slope, start=1)])
    loss_extrap = loss + slope * (chi - _chi)
    loss_increment = np.where(chi > _chi, loss_extrap, loss_poly)

    # Limit the loss to the loss_limit
    if loss_limit is not None:
        loss_increment = np.asarray(
            [loss_increment, np.full_like(loss_increment, loss_limit)]
        )
        loss_increment = math.smooth_min(
            loss_increment, method="logsumexp", alpha=25, axis=0
        )

    return loss_increment


def get_incidence_parameter(le, s, We, theta_in, theta_out, beta_in, beta_des):

    r"""
    Calculate the incidence parameter according to the correlation proposed by :cite:`benner_influence_1997`. 
    
    The incidence parameter is used to calculate the increment in profile losses due to the effect of incidence according to function `get_incidence_profile_loss_increment`.

    The quantity is calculated as:

    .. math::

        \chi = \frac{\mathrm{d_{le}}}{s}^{-0.05}\mathrm{We_{le}}^{-0.2}\frac{\cos\theta_\mathrm{in}}{\cos\theta_\mathrm{out}}^{-1.4}(\beta_\mathrm{in} - \beta_\mathrm{des})

    where:

        - :math:`\mathrm{d_{le}}` is the leading edge diameter.  
        - :math:`s` is the pitch.
        - :math:`\mathrm{We_{le}}` is the leading edge wedge angle.
        - :math:`\theta_\mathrm{in}` and :math:`\theta_\mathrm{out}` is the blade metal angle at the inlet and outlet respectively.
        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{des}` is the inlet relative flow angle at given and design conditions respectively.

    Parameters
    ----------
    le : float
        Leading edge diameter.
    s : float
        Pitch.
    We : float
        Leading edge wedge angle (in degrees).
    theta_in : float
        Leading edge metal angle (in degrees).
    theta_out : float
        Trailing edge metal angle (in degrees).
    beta_in : float
        Inlet relative flow angle (in degrees).
    beta_des : float
        Inlet relative flow angle at design condition (in degrees).

    Returns
    -------
    float
        Incidence parameter.
    """

    chi = (
        (le / s) ** (-0.05)
        * (We) ** (-0.2)
        * (math.cosd(theta_in) / math.cosd(theta_out)) ** (-1.4)
        * (abs(beta_in) - abs(beta_des))
    )
    return chi


def F_t(BSx, beta_in, beta_out):

    r"""

    Calculate the tangential loading coefficient according to the correlation proposed by :cite:`benner_empirical_2006-1`.

    .. math::

        F_t = 2\frac{s}{c_\mathrm{ax}}\cos^2(\beta_m)(\tan(\beta_\mathrm{in} - \beta_\mathrm{out}))

    where:

        - :math:`s` is the pitch.
        - :math:`c_\mathrm{ax}` is the axial chord.
        - :math:`\beta_m = \tan^{-1}(0.5(\tan(\beta_\mathrm{in}) + \tan(\beta_\mathrm{out})))` is the mean vector angle.
        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{out}` is the inlet and outlet relative flow angle.

    Parameters
    ----------
    Bsx : float
        Blade solidity based on axial chord.
    beta_in : float
        Inlet relative flow angle (in degrees).
    beta_out : float
        Exit relative flow angle (in degrees).

    Returns
    -------
    float
        Tangetnial loading parameter.
    
    """

    a_m = math.arctand(0.5 * (math.tand(beta_in) + math.tand(beta_out)))

    F_t = (
        2
        * 1
        / BSx
        * math.cosd(a_m) ** 2
        * (abs(math.tand(beta_in)) + abs(math.tand(beta_out)))
    )

    return F_t

