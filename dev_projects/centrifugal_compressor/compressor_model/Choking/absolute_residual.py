
import CoolProp as cp 
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative
from scipy import optimize
import turboflow as tf
import time


def get_res_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat):

        """ 
        Get mass flow rate residual at the throat for normalized velocity x
        """

        w_throat = w_throat[0]

        # Throat
        h0_rel_throat = rothalpy + 0.5*u_throat**2 # Assume same radius at throat and inlet
        h_throat = h0_rel_throat - 0.5*w_throat**2
        s_throat = s_in # Assume isentropic flow
        statsic_properties_throat = fluid.get_props(cp.HmassSmass_INPUTS, h_throat, s_throat)
        d_throat = statsic_properties_throat["d"]
        m_throat = d_throat*w_throat*A_throat

        res = 1-m_throat/mass_flow_rate

        return np.array([res, tf.math.smooth_abs(res, method="logarithmic", epsilon=1e-1)])
        # return res


def get_res_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat, eps, f0):
    jac = approx_derivative(
        get_res_throat,
        w_throat,
        abs_step = eps,
        f0 = f0,
        method="3-point",
        args = (
            fluid,
            mass_flow_rate,
            rothalpy,
            s_in,
            A_throat, 
            u_throat,
        )
    )

    return jac[:, 0]

if __name__ == "__main__":
      
    # Boundary conditions
    p01 = 101325
    T01 = 273.15 + 15
    fluid_name = "air"
    omega = 14000*2*np.pi/60
    fluid = tf.props.Fluid(fluid_name, exceptions=True)
    mass_flow_rate = 5.0
    h0_rel1 = 423569.8891

    # Geometry
    radius_hub1 = 0.09/2
    radius_tip1 = 0.28/2
    r1 = (radius_tip1+radius_hub1)/2
    A1 = np.pi * (radius_tip1**2 - radius_hub1**2)
    A_throat = A1*0.5

    # Calculate case input
    props = fluid.get_props(cp.PT_INPUTS, p01, T01)
    s_in = props["s"]
    u_throat = omega*r1
    rothalpy = h0_rel1 - 0.5*u_throat**2


    # Define velocity array
    N = 1000
    w_min = 50
    w_max = 500
    w_throats = np.linspace(w_min, w_max, N)
    residual1 = np.zeros(N)
    gradient1 = np.zeros(N)
    residual2 = np.zeros(N)
    gradient2 = np.zeros(N)
    second_deriv1 = np.zeros(N)
    second_deriv2 = np.zeros(N)
    rel_step_fd = 1e-4


    for i in range(N):
        w_throat = np.array([w_throats[i]])
        eps = rel_step_fd * w_throat
        residual = get_res_throat(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat)
        gradient = get_res_gradient(w_throat, fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat, eps, residual)
        second_deriv = approx_derivative(get_res_gradient, w_throat, abs_step=eps, method = "2-point", args = (fluid, mass_flow_rate, rothalpy, s_in, A_throat, u_throat, eps, residual))

        residual1[i] = residual[0]
        gradient1[i] = gradient[0]
        residual2[i] = residual[1]
        gradient2[i] = gradient[1]
        second_deriv1[i] = second_deriv[0]
        second_deriv2[i] = second_deriv[1]

    fig1, ax1 = plt.subplots()
    ax1.plot([w_min, w_max], [0.00, 0.00], 'k--')
    ax1.plot(w_throats, residual1)
    ax1.grid(False)
    ax1.set_ylabel("Residual")
    ax1.set_xlabel("Throat velocity [m/s]")
    ax1.set_title("Residual original")

    fig2, ax2 = plt.subplots()
    ax2.plot([w_min, w_max], [0.00, 0.00], 'k--')
    ax2.plot(w_throats, gradient1)
    ax2.grid(False)
    ax2.set_ylabel("Gradient")
    ax2.set_xlabel("Throat velocity [m/s]")
    ax2.set_title("Residual original")

    fig3, ax3 = plt.subplots()
    ax3.plot([w_min, w_max], [0.00, 0.00], 'k--')
    ax3.plot(w_throats, residual2)
    ax3.grid(False)
    ax3.set_ylabel("Residual")
    ax3.set_xlabel("Throat velocity [m/s]")
    ax3.set_title("Residual modified")

    fig4, ax4 = plt.subplots()
    ax4.plot([w_min, w_max], [0.00, 0.00], 'k--')
    ax4.plot(w_throats, gradient2)
    ax4.grid(False)
    ax4.set_ylabel("Gradient")
    ax4.set_xlabel("Throat velocity [m/s]")
    ax4.set_title("Residual modified")

    fig5, ax5 = plt.subplots()
    ax5.plot([w_min, w_max], [0.00, 0.00], 'k--')
    ax5.plot(w_throats, second_deriv1)
    ax5.grid(False)
    ax5.set_ylabel("Second derivative")
    ax5.set_xlabel("Throat velocity [m/s]")
    ax5.set_title("Residual original")

    fig6, ax6 = plt.subplots()
    ax6.plot([w_min, w_max], [0.00, 0.00], 'k--')
    ax6.plot(w_throats, second_deriv2)
    ax6.grid(False)
    ax6.set_ylabel("Second derivative")
    ax6.set_xlabel("Throat velocity [m/s]")
    ax6.set_title("Residual modified")

    plt.tight_layout()
    plt.show()

