In Oh's paper: The vaneless diffuser losses included as internal losses?
In Meroni's paper: The vaneless diffuser modelled by using a pressure coefficient

# 1D model
The flow in the vaneless diffuser can be described by the following set of differential equations:
$$v_m \frac{d\rho}{dm} + \rho \frac{dv_m}{dm} + \frac{\rho v_m}{br} \frac{d}{dm}(br) = 0$$
$$\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm} - \frac{\rho v_\theta^2}{r} \sin(\phi) + \frac{2 \tau_w}{b} \cos(\alpha) = 0$$
$$\rho v_m \frac{d v_\theta}{dm} + \frac{\rho v_\theta v_m}{r} \sin (\phi) + \frac{2\tau_w}{b}\sin(\alpha) = 0$$
$$\rho v_m - \rho v_m a^2 \frac{d\rho}{dm} - \frac{2(\tau_wv + \dot{q}_w)}{b (\frac{\partial e}{\partial p})_\rho} = 0$$
where:
- $v_m$ is the velocity in meridional direction
- $v_\theta$ is the velocity in tangential direction
- $v$ is the velocity
- $\alpha = \arctan(v_\theta/v_m)$ is the flow angle
- $p$ is pressure
- $\rho$ is density
- $a$ is the speed of sound
- $m$ is the meridional coordinate
- $b$ is the channel width
- $r$ is the radius
- $\phi$ is the mean wall cant angle (?)
- $\tau_w = C_f \rho v^2/2$ is the wall shear stress, where $C_f$ is the skin-friction coefficient
- $\dot{q}_w$ is the heat flux at the wall (set to 0 for adiabatic wall)
- $(\frac{\partial e}{\partial p})_\rho$ is the change in internal energy wrt pressure at constant density

**Equations**:
- Mass conservation
- Meridional momentum
- Tangential momentum
- Energy

**Assumptions**:
- Steady
- Axisymmetric
- Varies only in the meridional direction

The equations above pose a system of Ordinary Differential Equations that can be expressed as:
$$A \frac{dU}{dm} = S$$
where $U = [v_m, v_\theta, \rho, p]$ is the solution vector and the coefficient matrix is:
$$ A = \begin{bmatrix} \rho & 0 & v_m & 0 \\ \rho  v_m & 0 & 0 & 1 \\ 0 & \rho v_m & 0 & 0 \\ 0 & 0 & -\rho v_m  a^2 & \rho v_m \end{bmatrix} $$
and the source vector is given by:
$$S = \begin{bmatrix} -\frac{\rho v_m}{b r} \frac{d}{dm}(br) \\ \frac{\rho v_\theta^2}{r} \sin(\phi) - \frac{2 \tau_w}{b} \cos(\alpha) \\ -\frac{\rho v_\theta v_m}{r}  \sin(\phi) - \frac{2 \tau_w}{b} \sin(\alpha) \\ \frac{2 (\tau_w v + q_w)}{b  (\frac{\partial e}{\partial p})_\rho} \end{bmatrix}$$
The system of of ODEs can be solved using a linear equation solver to get $\frac{dU}{dm}$ .
Once $\frac{dU}{dm}$  is known, the solution vector $U$ at any position $x$ can be obtained by integrating $\frac{dU}{dm}$​ over the interval $[0, x]$. This is typically done using numerical methods for solving initial value problems, such as the Runge-Kutta method.