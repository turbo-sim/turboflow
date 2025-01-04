# Model Equations
- Known: Inlet state, geometry $\vec{G}$
- Independent variables: $v_{out}$, $s_{out}$
- Assumptions: Purely meridional velocity component at the exit

$h_{0,out} = h_{0,in}$
$h_{out} = h_{0,out} - 0.5v_{out}^2$
$\rho_{out}, p_{out} = f(h_{out}, s_{out})$
$p_{0, out} = f(h_{0, out}, s_{out})$
$\dot{m}_{out} = \rho_{out}v_{out}A_{out}$
$v_{scroll} = v_{out}A_{out}/A_{scroll}$
$Y^{LM} = f(all)$
$Y^{def} = h_{0,out} - h_{0,out,is}$

### Residuals
$\Delta \dot{m}_{out}  = (\dot{m}_{out} - \dot{m})/\dot{m}$
$\Delta Y = Y^{LM} - Y^{def}$

# Loss equations
According to Ceyrowsky et al, the loss coefficient in the volute is the sum of 3 components:

$$ \zeta_\text{volute} = \zeta_\text{radial} + \zeta_\text{tangential} + \zeta_\text{cone}$$

where the original expressions provided by Ceyrowsky can be reformulated as:
$$

\begin{gather}

\zeta_\text{radial} = \frac{h-h_s}{v_4^2/2} = \frac{\Delta p_0}{\rho v_4^2/2} = v_{r,4}^2/v_4^2 = \cos{(\alpha_4)}^2 \\

\zeta_\text{tangential} = \frac{h-h_s}{v_4^2/2} = \frac{\Delta p_0}{\rho v_4^2/2} = \left( v_{\theta,4}/v_4\right)^2 \left( 1 - \frac{A_4\sin{\alpha_4}}{A_5}\right)^2 = \sin({\alpha_4})^2 \left( 1 - \frac{A_4\sin{\alpha_4}}{A_5}\right)^2 \\

  

\zeta_\text{cone} = \frac{h-h_s}{v_4^2/2} = \frac{\Delta p_0}{\rho v_4^2/2} = \left(v_{5}/v_4 \right)^2 \left( 1 - \frac{A_5}{A_6}\right)^2 = \cos({\alpha_4})^2 \left( \frac{A_4}{A_5} \right)^2\left( 1 - \frac{A_5}{A_6}\right)^2 \\

  

\end{gather}
$$
The **radial** component imply that 100% of the radial velocity kinetic energy is dissipated through the volute. The **tangential** and **cone** component account for a sudden expansion loss of the tangential velocity's kinetic energy in the case of deceleration in the scroll. 
# Derivations
### Derivation of sudden expansion loss

Mass balance:

$$ A_1 v_1 = A_2 v_2$$

Momentum balance considering the contribution of the force $p_1(A_2 - A_1)$ exerted by the expansion's wall:

$$(p_1 + \rho v_1^2)A_1 + p_1(A_2 - A_1) = (p_2 + \rho v_2^2)A_2 $$

Which can be rewritten as:

$$p_1 - p_2 = \rho v_1^2 \left( \frac{A_1}{A_2}\right)  \left(\frac{A_1}{A_2} - 1 \right) $$

Stagnation pressure balance:

$$ \Delta p_0/\rho = (p_1 - p_2)/\rho + \frac{1}{2}(v_1^2 - v_2^2)$$

Combining both relations:

$$\zeta = \frac{\Delta p_0}{\rho v_1^2/2} = \left(\frac{A_1}{A_2} - 1 \right)^2$$