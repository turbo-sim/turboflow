
According to Ceyrowsky et al, the loss coefficient in the volute is the sum of 3 components:
$$ \zeta_\text{volute} = \zeta_\text{radial} + \zeta_\text{tangential} + \zeta_\text{cone}$$
where the original expressions provided by Ceyrowsky can be reformulated as:
$$
\begin{gather}
\zeta_\text{radial} = \frac{h-h_s}{v_4^2/2} = \frac{\Delta p_0}{\rho v_4^2/2} = v_{r,4}^2/v_4^2 = \cos{(\alpha_4)}^2 \\
\zeta_\text{tangential} = \frac{h-h_s}{v_4^2/2} = \frac{\Delta p_0}{\rho v_4^2/2} = \left(\frac{v_{5,\text{ideal}}-v_{\theta,4}}{v_4}\right)^2 = \left( \cos{\alpha_4}  \left(\frac{r_4}{r_{5m}}\right) - \sin{\alpha_4} \left(\frac{A_4}{A_5}  \right)\right)^2 \\

\zeta_\text{cone} = \frac{h-h_s}{v_4^2/2} = \frac{\Delta p_0}{\rho v_4^2/2} = \left(v_{5}/v_4 \right)^2 \left( 1 - \frac{A_5}{A_6}\right)^2 = \cos({\alpha_4})^2 \left( \frac{A_4}{A_5} \right)^2\left( 1 - \frac{A_5}{A_6}\right)^2 \\

\end{gather}
$$

Where the following assumptions are used:
- Incompressible flow: $\rho_4 = \rho_5$
- Mass balance between 4 and 5:
$$ 
\begin{gather}
v_{5,\text{ideal}}  = \left( \frac{r_4}{r_{5m}} \right) \, v_{\theta,4} = \left( \frac{r_4}{r_{5m}} \right) \, \cos{\alpha_4} \, v_{4}
\end{gather}
$$
- Ideal angular momentum balance between 4 and 5
$$
\begin{gather}
v_5 A_5  = v_{m4} A_4 = v_4 \cos{\alpha_4} A_4 \\
v_5 = \left( \frac{A_4}{A_{5}}  \right)\cos{\alpha_4} v_4

\end{gather}
$$
**Ideal flow in the volute space**


Is this not over-defined? The exit velocity  can be calculated in two different ways for a given geometry. Could it be that the second equation does not make so much sense?

## Derivation of sudden expansion loss

Mass balance:
$$(p_1 + \rho v_1^2)A_1 + p_1(A_2 - A_1) = (p_2 + \rho v_2^2)A_2 $$
Which can be rewritten as:
$$p_1 - p_2 = \rho v_1^2 \left( \frac{A_1}{A_2}\right)  \left(\frac{A_1}{A_2} - 1 \right) $$
Stagnation pressure balance:
$$ \Delta p_0/\rho = (p_1 - p_2)/\rho + \frac{1}{2}(v_1^2 - v_2^2)$$
Combining both relations:
$$\zeta = \frac{\Delta p_0}{\rho v_1^2/2} = \left(\frac{A_1}{A_2} - 1 \right)^2$$

