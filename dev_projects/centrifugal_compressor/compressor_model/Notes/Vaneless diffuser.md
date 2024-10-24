

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

# Algebraic model
The inlet of the vaneless diffuser is the same as the exit of the impeller
- Know: Inlet state, geometry $\vec{G}$
- Independents: $v_{out}$, $s_{out}$, $\alpha_{out}$
- Assumption: Assume skin friction factor $C_f$

$v_{m,out} = v_{out} \cos \alpha_{out}$
$v_{\theta,out} = v_{out} \cos \alpha_{out}$
$h_{0,out} = h_{0,in}$
$h_{out} = h_{0,out} - 0.5v_{out}^2$
$\rho_{out}, p_{out} = f(h_{out}, s_{out})$
$p_{0, out} = f(h_{0, out}, s_{out})$
$\dot{m}_{out} = \rho_{out}v_{m, out}A_{out}$
$Y^{LM} = 2C_f(r_{out}-r_{in})/b_{in}$
$Y^{def} = 2(h_{out} - h_{out,is})/v_{in}^2$

### Residuals
$\Delta \dot{m}_{out}  = (\dot{m}_{out} - \dot{m})/\dot{m}$
$\Delta Y = Y^{LM} - Y^{def}$
$\Delta AM = \frac{r_{out} v_{\theta, out}}{r_{in} v_{\theta, in}} -  \exp\left({-\frac{c_f (r_{out} - r_{in})}{b_{in} \cos{\alpha_{in}}}}\right)$

# Derivations
## Flow equations in simple ducts

The governing equations for steady-state flow in a one-dimensional duct with area change, heat transfer, and friction are given by:
$$

\begin{gather}

v \frac{\text{d} \rho}{\text{d} x}+\rho \frac{\text{d} v}{\text{d}x} = -\frac{\rho v}{A} \frac{\text{d} A}{\text{d} x} \\

\rho v \frac{\text{d} v}{\text{d} x}+\frac{\text{d} p}{d x} = -\frac{4}{D_h} \tau_\text{w} \\

\frac{\text{d} p}{\text{d} x}-a^2 \frac{\text{d} \rho}{\text{d} x} =  \left(\frac{4G}{v D_h} \right) \left(\tau_\text{w} \, v+\dot{q}_\text{w}\right)

\end{gather}

$$
where:

- $A$ is the cross sectional area of the duct.

- $D_h = 4 A/P$ is the hydraulic diameter.

- $G=\frac{1}{\rho}\left(\frac{\partial p}{\partial e} \right)_\rho$ is the Grüneisen parameter.

- $\tau_\text{w} = \frac{c_f}{2} \rho u^2$ is the viscous stress at the walls.

- $\dot{q}_\text{w} = U \, (T-T_\text{w})$ is the heat flux at the walls.

The flow equations can be written in compact form as:

$$

\begin{gather}

A\frac{\text{d}\textbf{U}}{\text{d}x} =\textbf{S}(\textbf{U}) \\

\mathbf{U}=\left[\begin{array}{c}

\rho \\

v \\

p

\end{array}\right], \quad

A = \left[\begin{array}{ccc} v & \rho & 0 \\ 0 & \rho & v \\ -a^2 & 0 & 1 \end{array}\right], \quad

\mathbf{S}=\left[\begin{array}{c}

 -\frac{\rho v}{A} \frac{\text{d} A}{\text{d} x}\\

-\frac{4}{D_h} \tau_\text{w}  \\

\left(\frac{4G}{v D_h} \right) \left(\tau_\text{w} \, v+\dot{q}_\text{w}\right)

\end{array}\right]

\end{gather}

$$

This system of ordinary differential equations can be solved using Runge-Kutta methods except for the sonic point $\text{Ma}=v/a=1$ where the matrix $A$ is singular.

### Entropy generation

The transport equation for entropy is given by:

$$\rho v \frac{\text{d} s}{\text{d} x} = \left(\frac{4}{D_h}\right) \left(\frac{\dot{q}_\text{w}}{T_\text{w}}\right) + \dot{\sigma}$$

where the entropy generation $\dot{\sigma}$ can be derived from the Gibbs equation ($T\text{d}s = \text{d}e - \frac{\text{d}\rho}{\rho^2}$)

$$ \dot{\sigma} =  \left(\frac{4}{D_h \, T}\right) \left( \tau_\text{w} \, v + \left(1 - \frac{T}{T_\text{w}} \right)\dot{q}_\text{w} \right) $$

For the special case when the heat flux is zero ($\dot{q}_\text{w}=0$) the entropy equation reduces to:

$$\rho v \frac{\text{d} s}{\text{d} x} =  \frac{4}{D_h T} \left( \tau_\text{w} \, v \right)$$

Approximating the integral assuming constant variables evaluated at the inlet:

$$\Delta s = s_2 - s_1 \approx  \frac{4 c_f}{D_h T} \left( \frac{v^2}{2} \right) \Delta x$$

or expressed in terms of the entropy loss coefficient:

$$\zeta_s = \frac{T \Delta s}{v^2/2} =  \left(\frac{4 \Delta x}{D_h}\right)c_f$$

This equation indicates that the entropy loss coefficient due to friction is proportional to the length of the duct and inversely proportional to the hydraulic diameter.

### Stagnation pressure equation  

The transport equation for stagnation pressure can be derived in two ways.

One is starting from its definition as:

$$ \text{d}p_0 = \text{d} p + \rho v \text{d}  v $$

This equation is valid for both incompressible and compressible flow because the differential of velocity is representative of an infinitesimal change in which density variation is neligible:

Comparing with expression with the momomentum equation we find that:

$$ \frac{\text{d} p_0}{\text{d} x}  = \frac{\text{d} p}{d x} + \rho v \frac{\text{d} v}{\text{d} x}  = -\frac{4}{D_h} \tau_\text{w} $$

An alternative way to reach this relation is to define the stagnation pressure as a function of enthalpy and entropy

$$p_0 = p (h_0, s) $$

Taking partial derivatives and replacing definitions:

$$\text{d} p_0 = \left(\frac{\partial p}{\partial h}\right)_s \text{d}h_0 + \left(\frac{\partial p}{\partial s}\right)_h \text{d}s = \rho \, \text{d}h_0 - \rho T \, \text{d}s $$

The transport equations for stagnation enthalpy and entropy are given as:

$$

\begin{gather}

\rho v \frac{\text{d} h_0}{\text{d} x } = \frac{4}{D_h} \dot{q}_\text{w} \\

\rho v T \frac{\text{d} s}{\text{d} x } = \frac{4}{D_h} (\dot{q}_\text{w} + \tau_\text{w}\, v)

\end{gather}

$$

Substituting into the stagnation pressure equation

$$

\frac{\text{d} p_0}{\text{d} x } = -\frac{4}{D_h} \tau_\text{w} \\

$$

which is exacly the same relation we found using the definition of stagnation pressure.
## Flow equations in annular ducts

The governing equations for steady-state flow in an annular duct with area change, heat transfer, and friction are given by:

$$

\begin{gather}

v_m  \frac{\text{d} \rho}{\text{d} m} + \rho  \frac{\text{d} v_m}{\text{d} m} = -\frac{\rho  v_m}{b \, r}  \frac{\text{d} ( b \, r)}{\text{d} m} \\

\rho v_m  \frac{\text{d} v_m}{\text{d} m} + \frac{\text{d} p}{\text{d} m} = \frac{\rho v_\theta^2 }{r} \, \sin{(\phi)} - \frac{4\tau_\text{w}}{D_h}  \cos{(\alpha)} \\

\rho v_m  \frac{\text{d} v_\theta}{\text{d} m} = -\frac{\rho v_\theta v_m }{r} \, \sin{(\phi)} - \frac{4\tau_\text{w}}{D_h}  \sin{(\alpha)} \\

\frac{\text{d} p}{\text{d} x}-a^2 \frac{\text{d} \rho}{\text{d} x} =  \left(\frac{4G}{v_m D_h} \right) \left(\tau_\text{w} \, v+\dot{q}_\text{w}\right)

\end{gather}

$$

where:

- $A=2\pi r b$ is the cross sectional area of the duct with $b$ being the channel height.

- The velocity components is defined as $v_m = v \cos{\alpha}$ and $v_\theta = v \sin{\alpha}$

- $D_h = 4 A/P=2/b$ is the hydraulic diameter.

- $G=\frac{1}{\rho}\left(\frac{\partial p}{\partial e} \right)_\rho$ is the Grüneisen parameter.

- $\tau_\text{w} = \frac{c_f}{2} \rho u^2$ is the viscous stress at the walls.

- $\dot{q}_\text{w} = U \,(T-T_\text{w})$ is the heat flux at the walls.

For the special case of a radial duct (e.g., the vaneless diffuser of a centrifugal compressor) the geometry simplified to $\text{d}m = \text{d}r$ and $\phi = \pi/2$ such that:

$$

\begin{gather}

v_m  \frac{\text{d} \rho}{\text{d} r} + \rho  \frac{\text{d} v_m}{\text{d} r} = -\frac{\rho  v_m}{b \, r}  \frac{\text{d}}{\text{d} r}( b \, r) \\

\rho v_m  \frac{\text{d} v_m}{\text{d} r} + \frac{\text{d} p}{\text{d} r} = \frac{\rho v_\theta^2 }{r}  - \frac{4\tau_\text{w}}{D_h}  \cos{(\alpha)} \\

\rho v_m  \frac{\text{d} v_\theta}{\text{d} m} = -\frac{\rho v_\theta v_m }{r} - \frac{4\tau_\text{w}}{D_h}  \sin{(\alpha)} \\

\frac{\text{d} p}{\text{d} x}-a^2 \frac{\text{d} \rho}{\text{d} x} =  \left(\frac{4G}{v_m D_h} \right) \left(\tau_\text{w} \, v+\dot{q}_\text{w}\right)

\end{gather}

$$

The flow equations can be written in compact form as:

$$

\begin{gather}

A\frac{\text{d}\textbf{U}}{\text{d}r} =\textbf{S}(\textbf{U}) \\

  

U = \begin{bmatrix} v_m \\ v_\theta \\ \rho \\ p \end{bmatrix}

\quad

  

A = \begin{bmatrix}

    \rho & 0 & v_m & 0 \\

    \rho v_m & 0 & 0 & 1 \\

    0 & \rho v_m & 0 & 0 \\

    0 & 0 & - a^2 & 1

\end{bmatrix}

\quad

  

S = \begin{bmatrix}

    -\frac{\rho  v_m}{b \, r} \frac{\text{d}( b \, r)}{\text{d} m} \\

    \frac{\rho v_\theta^2 }{r} \, - \frac{4\tau_\text{w}}{D_h}  \cos{(\alpha)}           \\

    -\frac{\rho v_\theta v_m }{r} - \frac{4\tau_\text{w}}{D_h}  \sin{(\alpha)}    \\

    \left(\frac{4G}{v_m D_h} \right) \left(\tau_\text{w} \, v+\dot{q}_\text{w}\right)

\end{bmatrix}

\end{gather}

$$

This system of ordinary differential equations can be solved using Runge-Kutta methods except for the sonic point $\text{Ma}_m=v_m/a=1$ where the matrix $A$ is singular.
### Entropy generation

The transport equation for entropy is given by:

$$\rho v \frac{\text{d} s}{\text{d} r} = \left(\frac{4}{D_h}\right) \left(\frac{\dot{q}_\text{w}}{T_\text{w}}\right) + \dot{\sigma}$$

where the entropy generation $\dot{\sigma}$ can be derived from the Gibbs equation ($T\text{d}s = \text{d}e - \frac{\text{d}\rho}{\rho^2}$)

$$ \dot{\sigma} =  \left(\frac{4}{D_h \, T}\right) \left( \tau_\text{w} \, v + \left(1 - \frac{T}{T_\text{w}} \right)\dot{q}_\text{w} \right) $$

For the special case when the heat flux is zero ($\dot{q}_\text{w}=0$) the entropy equation reduces to:

$$\rho v \frac{\text{d} s}{\text{d} r} =  \frac{4}{D_h T} \left( \tau_\text{w} \, v \right)$$

Approximating the integral assuming constant variables evaluated at the inlet:

$$\Delta s = s_2 - s_1 \approx  \frac{4 c_f}{D_h T} \left( \frac{v^2}{2} \right) \Delta r$$

or expressed in terms of the entropy loss coefficient:

$$\zeta_s = \frac{T \Delta s}{v^2/2} =  \left(\frac{2 \Delta r}{b}\right)c_f$$

This equation indicates that the entropy loss coefficient due to friction is proportional to the length of the annular duct and inversely proportional to the duct height.

The entropy increase due to losses can be expressed as:

  

$$

h - h_s \approx T \Delta s =  \left(\frac{\Delta r}{b}\right)c_f \, v^2

$$

This expression, derived using only physical principles, matches the equation that Meroni uses in his paper for centrifugal compressors (note Meroni adopts the arithmetic average velocity):

$$

h - h_s = c_f \left( \frac{L}{D_h} \right) \frac{1}{2} \left( \frac{v_1 + v_2}{2} \right)^2 = c_f \left( \frac{\Delta r}{b} \right) \left( \frac{v_1 + v_2}{2} \right)^2

$$
### Tangential momentum balance

The tangential component of the momentum equation can be expressed as:

$$

\frac{\text{d} v_\theta}{v_\theta} = -\frac{ \text{d} r}{r} - \frac{4\tau_\text{w}  \sin{(\alpha)}}{D_h \, \rho \,v_m v_\theta} = -\frac{ \text{d} r}{r} - \frac{c_f}{b \cos{\alpha}}

$$

This differential equation is not tightly coupled with the other flow equations and can be approximately integrated assuming that the flow angle is constant

$$\ln{ \left(\frac{v_{\theta2}}{v_{\theta1}}\right)} = -\ln{ \left(\frac{r_{2}}{r_{1}}\right)} - \frac{c_f \Delta r}{b \cos{\alpha}}$$

Grouping up and solving in terms of the angular momentum $r v_\theta$ at the exit:

$$ r_2 v_{\theta2} = r_1 v_{\theta1} \cdot  \exp\left({-\frac{c_f \Delta r}{b \cos{\alpha}}}\right)$$

Linearizing the exponential term we find that:

$$ r_2 v_{\theta2} = r_1 v_{\theta1} \cdot  \left({1-\frac{c_f \Delta r}{b \cos{\alpha}}}\right)$$
Observations:

- The tangential component of velocity decreases when the radius increases and due to friction

- In the absence of friction ($c_f=0$) the angular momentum is conserved leading to the well known relation $r_2 v_{\theta2} = r_1 v_{\theta1}$.

- The assumption that the flow angle is constant during the integration is appropriate when the friction in the annular duct is small. The reason for this is that the flow angle has a constant value for frictionless flow, so the change in flow angle during integration is usually small.

- This equation is similar to the modeling equation that Meroni uses:

$$\frac{C_3 \sin \alpha_3}{C_4 \sin \alpha_4} = \frac{r_4}{r_3} + \frac{2 \pi C_f \rho_4 C_4 \sin \alpha_4 (r_3^2 - r_3 r_4)}{\dot{m}}$$

- Additionally, I think that there are typos or errors in the equation from Meroni. I think that the numerator and denominator of the first 2 terms should be swapped. In addition, I think that the term $\sin{\alpha_4}$ in the numerator could also be a mistake. If these two aspects are corrected, the equation used by Meroni is identical to the equation derived based on physical principles.

$$
\begin{gather}

R_1 = \rho_2 v_{m2} A_2 - \rho_1 v_{m1} A_1 = 0 \\

R_2 = \frac{r_2 v_{\theta2}}{r_1 v_{\theta1}} -  \exp\left({-\frac{c_f \Delta r}{b \cos{\alpha}}}\right) = 0 \\

R_3 = \zeta^\text{definition} - \zeta^\text{loss model} = \frac{h_2 - h(p_2, s_1)}{v_1^2/2} - \left(\frac{\Delta r}{b}\right)c_f(*2?) = 0

\end{gather}
$$