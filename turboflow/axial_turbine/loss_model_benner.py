# import numpy as np
from turboflow import math

import jax
import jax.numpy as jnp

from . import loss_model_kacker_okapuu as lm_ko
from .loss_coefficient_conversion import convert_kinetic_energy_to_stagnation_pressure_loss

# TODO: Add smoothing to the min/max/abs/piecewise functions

# TODO: Create test scripts to plot how the different loss model functions behave when smoothing is applied / verify that the behavior is reasonable beyond their intended range
# TODO: Raise warninings to a log file to be aware when the loss model is outside its range of validity and because of which variables
# TODO: the log object can be passed as an additional variable to the functions with a default parameter Logger=None
# TODO: Whenever something is out of range we can do "if Logger: " to check if the logger was actually passed as an input and log an descriptive message

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
    Y_p = lm_ko.get_profile_loss(flow_parameters, geometry)

    # Trailing edge coefficient
    Y_te = lm_ko.get_trailing_edge_loss(flow_parameters, geometry)

    # Secondary loss coefficient
    # Y_s = 0.0
    Y_s = get_secondary_loss(flow_parameters, geometry, delta_height)

    # Tip clearance loss coefficient
    Y_cl = lm_ko.get_tip_clearance_loss(flow_parameters, geometry)

    # Incidence loss coefficient
    # Y_inc = 0.0
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
    CR = math.cosd(beta_in) / math.cosd(beta_out)
    if AR <= 2:  # TODO: sigmoid blending to convert to smooth piecewise function
        denom = (
            jnp.sqrt(math.cosd(stagger))
            * CR
            * AR**0.55
            * (math.cosd(beta_out) / (math.cosd(stagger))) ** 0.55
        )
        Y_sec = (0.038 + 0.41 * jnp.tanh(1.2 * delta_height)) / denom
    else:
        denom = (
            jnp.sqrt(math.cosd(stagger))
            * CR
            * AR
            * (math.cosd(beta_out) / (math.cosd(stagger))) ** 0.55
        )
        Y_sec = (0.052 + 0.56 * jnp.tanh(1.2 * delta_height)) / denom

    return Y_sec


def get_incidence_loss(flow_parameters, geometry, beta_des):
    r"""
    Calculate the incidence loss coefficient according to the correlation proposed by :cite:`benner_influence_1997`.

    This model calculates the incidence loss parameter through the function `get_incidence_parameter`. The kinetic energu coefficient
    can be obtained from `get_incidence_profile_loss_increment`, which is a function of the incidence parameter. The kinetic energy loss coefficient is converted
    to the total pressure loss coefficient through `convert_kinetic_energy_to_stagnation_pressure_loss`.

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
    theta_out = math.arccosd(geometry["A_throat"] / geometry["A_out"])

    chi = get_incidence_parameter(le, s, We, theta_in, theta_out, beta_in, beta_des)

    dPhip = get_incidence_profile_loss_increment(chi)

    Y_inc = convert_kinetic_energy_to_stagnation_pressure_loss(Ma_rel_out, gamma, dPhip, limit_output=True)

    return Y_inc


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

    Z_TE = 0.10 * Ft**0.79 / jnp.sqrt(CR) / (AR) ** 0.55 + 32.70 * delta_height**2

    Z_TE = min(Z_TE, 0.99)  # TODO: smoothing

    return Z_TE



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
    coeffs_1 = jnp.array(
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
    func_1 = lambda x: jnp.sum(
        jnp.asarray([a * x**i for (i, a) in enumerate(coeffs_1[::-1], start=1)]),
        axis=0,
    )

    # Define the polynomial function for negative values of chi
    coeffs_2 = jnp.array([1.358e-4, -8.72e-4])
    func_2 = lambda x: jnp.sum(
        jnp.asarray([b * x**i for (i, b) in enumerate(coeffs_2[::-1], start=1)]),
        axis=0,
    )

    # Calculate the profile losses increment based on polynomials
    loss_poly = jnp.where(chi >= 0, func_1(chi), func_2(chi))

    # Apply linear extrapolation beyond the threshold for chi
    _chi = chi_extrapolation
    loss = func_1(_chi)
    coeffs_slope = jnp.array([i * a for i, a in enumerate(coeffs_1[::-1], start=1)])
    slope = jnp.sum(jnp.array([a * _chi ** (i - 1) for i, a in enumerate(coeffs_slope, start=1)]))
    loss_extrap = loss + slope * (chi - _chi)
    loss_increment = jnp.where(chi > _chi, loss_extrap, loss_poly)

    # Limit the loss to the loss_limit
    if loss_limit is not None:
        loss_increment = math.smooth_minimum(
            loss_increment, loss_limit, method="logsumexp", alpha=25
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

    # chi = (
    #     ((le + 1e-9) / s) ** (-0.05)
    #     * (We + 1e-9) ** (-0.2)
    #     * ((math.cosd(theta_in) / math.cosd(theta_out)) + 1e-9) ** (-1.4)
    #     * (abs(beta_in) - abs(beta_des))
    # )
    chi = (
        (le / s) ** (-0.05)
        * (We) ** (-0.2)
        * ((math.cosd(theta_in) / math.cosd(theta_out))) ** (-1.4)
        * (math.smooth_abs(beta_in, "quadratic", epsilon=1e-1) - math.smooth_abs(beta_des,"quadratic", epsilon=1e-1)) # Absolute function give problem with JAX differentiation, so use smooth abs
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
