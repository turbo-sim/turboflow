
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

## Derivation of sudden expansion loss

Mass balance:
$$ A_1 v_1 = A_2 v_2$$
Momentum balance assuming $p=p_1$ right after the expansion and separation:
$$(p_1 + \rho v_1^2)A_1 + p_1(A_2 - A_1) = (p_2 + \rho v_2^2)A_2 $$
Which can be rewritten as:
$$p_1 - p_2 = \rho v_1^2 \left( \frac{A_1}{A_2}\right)  \left(\frac{A_1}{A_2} - 1 \right) $$
Stagnation pressure balance:
$$ \Delta p_0/\rho = (p_1 - p_2)/\rho + \frac{1}{2}(v_1^2 - v_2^2)$$
Combining both relations:


$$\zeta = \frac{\Delta p_0}{\rho v_1^2/2} = \left(\frac{A_1}{A_2} - 1 \right)^2$$