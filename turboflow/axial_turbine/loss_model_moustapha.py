import numpy as np
from .. import math


def compute_losses(input_parameters):
    r"""

    Evaluate loss coefficient according to :cite:`kacker_mean_1982` and :cite:`moustapha_improved_1990`.

    This model split the total loss coefficient into profile, secondary, trailing edge, tip clearance and incidence.
    The model adopt the incidence loss coefficient proposed by  :cite:`moustapha_improved_1990`, but is otherwise similar to the
    model proposed by :cite:`kacker_mean_1982`.
    The loss coefficient are combined through the following relation:

    .. math::

        \mathrm{Y_{tot}} = \mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{corr}}\mathrm{Y_{s}} + \mathrm{Y_{cl}} + \mathrm{Y_{inc}}

    The function calls a function for each loss component:

        - :math:`\mathrm{Y_{p}}` : `get_profile_loss`
        - :math:`\mathrm{Y_{te}}` : `get_trailing_edge_loss`
        - :math:`\mathrm{Y_{s}}` : `get_secondary_loss`
        - :math:`\mathrm{Y_{cl}}` : `get_tip_clearance_loss`
        - :math:`\mathrm{Y_{inc}}` : `get_incidence_loss`
        - :math:`\mathrm{Y_{corr}}` : `get_secondary_loss_correction_factor`

    Parameters
    ----------
    input_parameters : dict
        A dictionary containing containing the keys, `flow`, `geometry` and `options`.

    Returns
    -------
    dict
        A dictionary with the loss components.

    """

    # Load data
    flow_parameters = input_parameters["flow"]
    geometry = input_parameters["geometry"]
    beta_des = geometry["leading_edge_angle"]

    # Profile loss coefficient
    Y_p = get_profile_loss(flow_parameters, geometry)

    # Secondary loss coefficient
    Y_s = get_secondary_loss(flow_parameters, geometry)

    # Tip clearance loss coefficient
    Y_cl = get_tip_clearance_loss(flow_parameters, geometry)

    # Trailing edge loss coefficienct
    Y_te = get_trailing_edge_loss(flow_parameters, geometry)

    # Incidence loss for profile loss
    Y_inc = get_incidence_loss(flow_parameters, geometry, beta_des)

    # Incidence correction factor for secondary loss coefficient
    Y_corr = get_secondary_loss_correction_factor(flow_parameters, geometry)
    Y_s *= Y_corr

    # Calculate total pressure loss coefficient
    Y = Y_p + Y_s + Y_cl + Y_te + Y_inc

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

    # Load data
    Re = flow_parameters["Re_out"]
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    Ma_rel_in = flow_parameters["Ma_rel_in"]
    p0rel_in = flow_parameters["p0_rel_in"]
    p0rel_is = flow_parameters["p0_rel_is"]
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
    f_Ma = 1 + 60 * (Ma_rel_out - 1) ** 2 * (Ma_rel_out > 1)

    # Compute losses related to shock effects at the inlet of the cascade
    f_hub = get_hub_to_mean_mach_ratio(r_ht_in, cascade_type)
    a = max(0, f_hub * Ma_rel_in - 0.4)
    # Y_shock = 0.75 * a**1.75 * r_ht_in * (p0rel_in - p_in) / (p0rel_out - p_out)
    Y_shock = 0.75 * a**1.75 * r_ht_in * (p0rel_is - p_in) / (p0rel_out - p_out)
    Y_shock = max(0, Y_shock)

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
    Y_p = max(Y_p, 0.8 * Yp_reaction)

    # Avoid unphysical effect on the thickness by defining the variable aa
    aa = max(0, -theta_in / beta_out)
    Y_p = Y_p * ((t_max / c) / 0.2) ** aa
    Y_p = 0.914 * (2 / 3 * Y_p * Kp + Y_shock)

    # Corrected profile loss coefficient
    Y_p = f_Re * f_Ma * Y_p

    return Y_p


def get_secondary_loss(flow_parameters, geometry):
    r"""
    The function calculates the secondary loss coefficient using the Kacker-Okapuu model.
    The main equation for :math:`\mathrm{Y_s}` is given by:

    .. math::

        \mathrm{Y_s} = 1.2 \cdot \mathrm{K_s} \cdot 0.0334 \cdot far \cdot Z \cdot \frac{\cos(\beta_{out})}{\cos(\theta_{in})}

    where:

        - :math:`K_s` is a correction factor accounting for compressible flow effects.
        - :math:`far` is a factor that account for the aspect ratio of the current cascade.
        - :math:`Z` is a blade loading parameter.
        - :math:`\beta_\mathrm{out}` is the exit flow angle.
        - :math:`\theta_\mathrm{in}` is the inlet metal angle.

    The correction factor :math:`\mathrm{K_s}` from a separate correction factor given from `get_compressible_correction_factors`.
    `get_compressible_correction_factors` returns :math:`\mathrm{K_s}`, which is used to calculate :math:`\mathrm{K_s}`:

    .. math::

        \mathrm{K_s} = 1 - \left(\frac{b}{H}\right)^2 (1-\mathrm{K_p})

    where:

        - :math:`b` is the axial chord.
        - :math:`H` is the mean blade height.

    The aspect ratio factor is calculated from the following correlation:

    .. math:: 

        far = \begin{cases}
                1 - \frac{0.25 \sqrt{|2-H/c|}}{H/c} & \text{if } H/c < 2.0 \\
                c/H & \text{if } H/c \geq 2.0
                \end{cases}

    where :math:`c` is the blade chord.

    The loading parameters is caclualated from the following correlation:

    .. math::

        Z = 4 (\tan(\beta_\mathrm{in}) - \tan(\beta_\mathrm{out}))^2\frac{\cos^2(\beta_\mathrm{out})}{\cos(\beta_m)} 

    where:

        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{out}` is the inlet and exit relative flow angle
        - :math:`\beta_m = \tan^{-1}(0.5(\tan(\beta_\mathrm{in})-\tan(\beta_\mathrm{out})))` is the mean gas angle.
           
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

    Ma_rel_out = flow_parameters["Ma_rel_out"]
    Ma_rel_in = flow_parameters["Ma_rel_in"]
    beta_out = flow_parameters["beta_out"]
    beta_in = flow_parameters["beta_in"]

    b = geometry["axial_chord"]
    H = geometry["height"]
    c = geometry["chord"]
    theta_in = geometry["leading_edge_angle"]

    # Compute compressible flow correction factors
    Kp, K2, K1 = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)

    # Secondary loss coefficient
    K3 = (b / H) ** 2
    Ks = 1 - K3 * (1 - Kp)
    Ks = max(0.1, Ks)
    angle_m = math.arctand((math.tand(beta_in) + math.tand(beta_out)) / 2)
    Z = (
        4
        * (math.tand(beta_in) - math.tand(beta_out)) ** 2
        * math.cosd(beta_out) ** 2
        / math.cosd(angle_m)
    )
    far = (1 - 0.25 * np.sqrt(abs(2 - H / c))) / (H / c) * (H / c < 2) + 1 / (H / c) * (
        H / c >= 2
    )
    Y_s = 1.2 * Ks * 0.0334 * far * Z * math.cosd(beta_out) / math.cosd(theta_in)

    return Y_s


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
    r_to = min(0.4, t_te / o)

    # Interpolate data
    d_phi2_reaction = np.interp(r_to, r_to_data, phi_data_reaction)
    d_phi2_impulse = np.interp(r_to, r_to_data, phi_data_impulse)

    # Compute kinetic energy loss coefficient
    d_phi2 = d_phi2_reaction - abs(angle_in / angle_out) * (angle_in / angle_out) * (
        d_phi2_impulse - d_phi2_reaction
    )

    # Limit the extrapolation of the trailing edge loss
    d_phi2 = max(d_phi2, d_phi2_impulse / 2)
    Y_te = 1 / (1 - d_phi2) - 1

    return Y_te


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
    Calculate the incidence loss coefficient according to the correlation proposed by :cite:`moustapha_improved_1990`.
    
    The model first computes the incidence parameter, :math:`\chi`. Based on this parameter, the kinetic-energy incidence 
    loss coefficient is calculated to match experimental data.

    .. math::

        \Delta\phi^2_p = \begin{cases}
                        -5.1734e^{-6}\chi + 7.6902e^{-9}\chi^2 & \text{if } -800 \leq \chi \leq 0 \\
                        0.778e^{-5}\chi + 0.56e^{-7}\chi^2 + 0.4e^{-10}\chi^3 + 2.054e^{-19}\chi^6 & \text{if } 0 \leq \chi \leq 800
                        \end{cases}

    The kinetic-energy coefficient is converted to total pressure loss coefficient through the `convert_kinetic_energy_coefficient` function. 

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

    Raises
    ------
    Warning
        If incidence parameter (:math:`chi`) is out of range.

    """

    Ma_rel_out = flow_parameters["Ma_rel_out"]  # Exit relative mach number
    beta_in = flow_parameters["beta_in"]  # Inlet relativ flow angle
    gamma = flow_parameters["gamma_out"]  # Specific heat ratio

    cascade_type = geometry["cascade_type"]
    s = geometry["pitch"]  # Pitch
    le = geometry["leading_edge_diameter"]  # Leading edge diameter
    theta_in = geometry["leading_edge_angle"]  # Inlet metal angle
    theta_out = geometry["gauging_angle"]  # Exit metal angle

    # Compute incidence parameter
    chi = get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des, cascade_type)

    # Check if incidence parameter is within the range of the experimental data
    if abs(chi) > 800:
        raise Warning("Incidence parameter out of range: chi = {chi}")

    # Calculate kinetic-energy loss coefficient
    if chi >= 0:
        dPhi = 0.778e-5 * chi + 0.56e-7 * chi**2 + 0.4e-10 * chi**3 + 2.054e-19 * chi**6
    elif chi < 0:
        dPhi = -5.1734e-6 * chi + 7.6902e-9 * chi**2

    # Convert kinetic-energy loss coefficient to total pressure loss coeffcient
    Y_inc = convert_kinetic_energy_coefficient(dPhi, gamma, Ma_rel_out)

    return Y_inc


def get_secondary_loss_correction_factor(flow_parameters, geometry):
    r"""
    Calculate the correction factor for the secondary loss coefficient to account for incidence losses.

    The correction factor is calculated based on the secondary flow incidence parameter (:math:`\chi`), which is determined using
    geometrical and flow parameters.

    The correction factor (:math:`Y_\mathrm{corr}`) is given by:

    .. math::

        Y_\mathrm{corr} = \begin{cases}
        e^{0.9\chi} + 13\chi^2 + 400\chi^4 & \text{if } \chi \geq 0 \\
        e^{0.9\chi} & \text{if } \chi < 0
        \end{cases}

    and the incidence paramter is calculated as:

    .. math::

        \chi = \frac{\beta_\mathrm{in} - \theta_\mathrm{in}}{180 - (\theta_\mathrm{in} + \theta_\mathrm{out})}\left(\frac{\cos(\theta_\mathrm{in})}{\cos(\theta_\mathrm{out})}\right)^{-1.5}\left(\frac{\mathrm{d_{le}}}{c}\right)^{-0.3}

    where:

        - :math:`\chi` is the secondary flow incidence parameter.
        - :math:`c` is the chord length.
        - :math:`\mathrm{d_{le}}` is the leading edge diameter.
        - :math:`\theta_\mathrm{in}` is the inlet metal angle.
        - :math:`\theta_\mathrm{out}` is the exit metal angle.
        - :math:`\beta_\mathrm{in}` is the inlet relative flow angle.

    The function also checks if the secondary incidence parameter falls within the range of the experimental data used to 
    determine the correction factor.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.

    Returns
    -------
    float
        Secondary loss correction factor.

    """

    beta_in = flow_parameters["beta_in"]

    c = geometry["chord"]
    le = geometry["leading_edge_diameter"]
    theta_in = geometry["leading_edge_angle"]
    theta_out = geometry["gauging_angle"]

    chi = (
        (abs(beta_in) - abs(theta_in))
        / (180 - (theta_in + theta_out))
        * (math.cosd(theta_in) / math.cosd(theta_out)) ** -1.5
        * (le / c) ** -0.3
    )

    if not (-0.4 < chi < 0.3):
        raise Warning("Secondary incidence parameter out of range: chi = {chi}")

    if chi >= 0:
        Y_corr = np.exp(0.9 * chi) + 13 * chi**2 + 400 * chi**4
    else:
        Y_corr = np.exp(0.9 * chi)

    return Y_corr


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
    r_sc_min = (0.46 + phi / 77) * (phi < 30) + (0.614 + phi / 130) * (phi >= 30)
    X = r_sc - r_sc_min
    A = (0.025 + (27 - phi) / 530) * (phi < 27) + (0.025 + (27 - phi) / 3085) * (
        phi >= 27
    )
    B = 0.1583 - phi / 1640
    C = 0.08 * ((phi / 30) ** 2 - 1)
    n = 1 + phi / 30
    Yp_reaction = (A + B * X**2 + C * X**3) * (phi < 30) + (A + B * abs(X) ** n) * (
        phi >= 30
    )

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
    B = (0.3 + (30 - phi) / 50) * (phi < 30) + (0.3 + (30 - phi) / 275) * (phi >= 30)
    C = 0.88 - phi / 42.4 + (phi / 72.8) ** 2
    Yp_impulse = A + B * X**2 - C * X**3

    return Yp_impulse


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
    )
    K2 = (Ma_rel_in / Ma_rel_out) ** 2
    Kp = 1 - K2 * (1 - K1)
    Kp = max(0.1, Kp)
    return [Kp, K2, K1]


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

    if r_ht < 0.5:
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


def get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des, cascade_type):
    r"""
    Calculate the incidence parameter according to the correlation proposed by :cite:`moustapha_improved_1990`.

    The incidence parameter is used to calculate the increment in profile losses due to the effect of incidence according to function `get_incidence_loss`.

    The quantity is calculated as:

    .. math::

        \chi = \left(\frac{\mathrm{d_{le}}}{s}\right)^{-1.6}\left(\frac{\cos{\theta_\mathrm{in}}}{\cos{\theta_\mathrm{out}}}\right)^{-2}(\beta_\mathrm{in} - \beta_\mathrm{des})

    where:

        - :math:`\mathrm{d_{le}}` is the leading edge diameter.
        - :math:`s` is the pitch.
        - :math:`\theta_\mathrm{in}` and :math:`\theta_\mathrm{out}` is the blade metal angle at the inlet and outlet respectively.
        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{des}` is the inlet relative flow angle at given and design conditions respectively.

    Parameters
    ----------
    le : float
        Leading edge diameter.
    s : float
        Pitch.
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

    # TODO: explain smoothing/blending tricks

    sign_factor = 2 * (cascade_type == "rotor") - 1

    chi = (
        (le / s) ** (-1.6)
        * (math.cosd(theta_in) / math.cosd(theta_out)) ** (-2)
        * sign_factor*(beta_in - beta_des)
    )
    return chi


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
