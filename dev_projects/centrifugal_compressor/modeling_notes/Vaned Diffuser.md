### Equations
The inlet of the vaned diffuser is the same as the exit of the vaneless diffuser
- Know: Inlet state, $\vec{G}$
- Independents: $v_{out}$, $s_{out}$

$\alpha_{out} = f(\mathrm{incidence}, \vec{G})$
$v_{m,out} = v_{out} \cos \alpha_{out}$
$v_{\theta,out} = v_{out} \cos \alpha_{out}$
$h_{0,out} = h_{0,in}$
$h_{out} = h_{0,out} - 0.5v_{out}^2$
$\rho_{out}, p_{out} = f(h_{out}, s_{out})$
$p_{0, out} = f(h_{0, out}, s_{out})$
$\dot{m}_{out} = \rho_{out}v_{m, out}A_{out}$
$Y^{LM} = f(all)$
$Y^{def} = h_{0,out} - h_{0,out,is}$

**Deviation**
Aungier used a combination of minimum loss deviation and variation of deviation with incidence:
$\alpha_{out} = \theta_{out} - \delta^* - \frac{\partial \delta}{\partial i}(\theta_{in} - \alpha_{in})$
Minimum-loss deviation (Howell, 1947):  $$\delta^* = \frac{\Delta \theta[0.92(a/c)^2 + 0.02(90^o-\theta_{out})]}{\sqrt(\sigma)-0.02\Delta \theta}$$
where:
- Location of maximum camber: $(a/c) = [2-(\theta_{avg}-\theta_{in})/(\theta_{out}-\theta_{in})]/3$
- Solidity: $\sigma = z(r_{out} - r_{in})/(2\pi r_{in} \sin \theta_{avg})$
- Camber angle: $\Delta \theta = \theta_{out} - \theta_{in}$
Variation of deviation with incidence (Johnsen and Bullock, 1965):
$$\frac{\partial \delta}{\partial i} = \exp[((1.5-\theta_{out}/60)^2-3.3)\sigma]$$
NB! This deviation model use angles relative for tangential direction, and use a sign convention such that the camber angle is positive

### Residuals
$\Delta \dot{m}_{out}  = (\dot{m}_{out} - \dot{m})/\dot{m}$
$\Delta Y = Y^{LM} - Y^{def}$

### Notes
- The density can be calculated directly from the mass flow rate to avoid giving $s_{out}$ as independent variable. This gives a simpler problem but is not as flexible if other function calls should be used. 
- The exit flow angle could be given as an independent variable, in case other deviation models is adopted that is not suitable with the explicit formulation (e.g. if it depends on the mach number).
- What sign convention should be used, and how is the sign set (positive in the direction of the impeller blade velocity?)