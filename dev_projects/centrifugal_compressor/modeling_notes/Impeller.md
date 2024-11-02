### Impeller inlet
- Known: $p_{0,in}$,$T_{0,in}$,$\dot{m}$,$\alpha_{in}$ $\omega$, $\vec{G}$
- Independents: $v_{in}$
- Assumption: No streamline slope

**Velocity triangles**
$u_{in} = r_{in}\omega$
$v_{m, in} = v_{in} \cos \alpha_{in}$
$v_{\theta, in} = v_{in} \sin \alpha_{in}$
$w_{m, in} = v_{m, in}$
$w_{\theta, in} = v_{\theta, in} - u_{in}$
$w_{in} = \sqrt{w_{\theta, in}^2 + w_{m, in}^2}$
$\beta_{in} = \arctan w_{\theta, in}/w_{m, in}$

**Thermophysical properties**
$h_{0,in}, s_{in} = f(p_{0,in}, T_{0,in})$
$h_{in} = h_{0,in} - 0.5v_{in}^2$
$h_{0,rel,in} = h_{in} + 0.5w_{in}^2$
$\rho_{in}, p_{in} = f(h_{in}, s_{in})$
$p_{0,rel,in}, T_{0,rel,in} = f(h_{0,rel,in}, s_{in})$

**Rothalpy and mass flow rate**
$\dot{m}_{in} = \rho_{in}v_{m,in}A_{in}$
$I = h_{0,rel,in} - 0.5u_{in}^2$
### Impeller exit
- Known: $I$, $\omega$, $\vec{G}$
- Independents: $w_{out}$, $s_{out}$, $\beta_{out}$
- Assumption: No streamline slope

**Velocity triangles**
$u_{out} = r_{out}\omega$
$w_{m, out} = w_{out} \cos \beta_{out}$
$w_{\theta, out} = w_{out} \sin \beta_{out}$
$v_{m, out} = w_{m, out}$
$v_{\theta, out} = w_{\theta, out} + u_{out}$ 
$v_{out} = \sqrt{v_{\theta, out}^2 + v_{m, out}^2}$
$\alpha_{out} = \arctan v_{\theta, out}/v_{m, out}$

**Thermophysical properties**
$h_{0,rel,out} = I + 0.5u_{out}^2$
$h_{out} = h_{0,rel,out} - 0.5w_{out}^2$
$h_{0,out} = h_{out} + 0.5v_{out}^2$
$p_{0,rel,out}, T_{0,rel,out} = f(h_{0,rel,out}, s_{out})$
$p_{out}, \rho_{out} = f(h_{out}, s_{out})$
$p_{0,out} = f(h_{0,out}, s_{out})$

**Loss coefficients**
$Y^{def} = h_{0,out} - h_{0,out,is}$

**Loss model**
$Y^{LM} = f(all)$

**Slip**
$v_{\theta,s} = u_{out} + v_{m,out} \tan \theta_{out} - v_{\theta,out}$
Wiesner: 
$$v_{\theta,s}^{SM} = \frac{u_{out}\sqrt{\cos \theta_{out}}}{z^{0.7}}$$
**Mass flow rate**
$\dot{m}_{out} = \rho_{out}w_{m,out}A_{out}$
### Impeller residuals
$\Delta \dot{m}_{in}  = (\dot{m}_{in} - \dot{m})/\dot{m}$
$\Delta \dot{m}_{out}  = (\dot{m}_{out} - \dot{m})/\dot{m}$
$\Delta Y = Y^{LM} - Y^{def}$
$\Delta v_{\theta,s} = v_{\theta,s} - v_{\theta,s}^{SM}$
### Notes
- What changes are necessary to account for streamline slopes?
- By giving the meridonial velocity at impeller exit, the velocity triangle could be given explicit without guessing the flow angle. The implicit calculation may be more flexible in terms of different slip correlations.
- We could omit giving the entropy, by rather calculating the density from the mass flow rate. 
### Nomenclature
Metal angle: $\theta$
Number of impeller blades: $z$
Slip velocity: $v_{\theta,s}$
Angular speed: $\omega$

**Superscript**
Slip model: $SM$
Loss model: $LM$
Definition: $def$