

## Derivation of relations between loss coefficient definitions

**Assumptions**

- Fluid is modelled as a perfect gas
- Zero-dimensional flow in convergent nozzle
- Losses are modelled by a kinetic energy loss coefficient or a stagnation pressure loss coefficient

**Observations**

- Kinetic energy loss coefficient varies between zero and one
- The enthalpy and stagnation pressure loss coefficients vary between zero and infinity

**Relation between kinetic and stagnation pressure loss coefficients**

The kinetic energy loss coefficient is defined as:
$$
\begin{aligned}
& \eta =1 - \Delta \phi^{2} = \left(\frac{v}{v_{s}}\right)^{2}=\left(\frac{h_{0}-h_{2}}{h_{0}-h_{2s}}\right)=\left(\frac{T_{0}-T_{2}}{T_{0}-T_{2s}}\right) \\
& \eta =1 - \Delta \phi^{2} = \frac{1-\left(\frac{T_{2}}{T_0}\right)}{1-\left(\frac{T_{2s}}{T_0}\right)} = \frac{1-\left(\frac{p_{2}}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}}}{1-\left(\frac{p_{2}}{p_{01}}\right)^{\frac{\gamma-1}{\gamma}}} = \frac{1-\left(\frac{p_{2}}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}}}{1-\left[\left(\frac{p_{2}}{p_{02}}\right)\left(\frac{p_{02}}{p_{01}}\right)\right]^{\frac{\gamma-1}{\gamma}}}
\end{aligned}
$$
Solving for the kinetic energy loss coefficient we get:
$$
\Delta \phi^{2} = 1-\frac{1-\left(\frac{p_{2}}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}}}{1-\left[\left(\frac{p_{2}}{p_{02}}\right)\left(\frac{p_{02}}{p_{01}}\right)\right]^{\frac{\gamma-1}{\gamma}}}
$$
where the static-to-total pressure ratio is given by:
$$\left(\frac{p_2}{p_{02}}\right)=\left(\frac{T_2}{T_{02}}\right)^{\frac{\gamma}{\gamma-1}}=\left[1+\left(\frac{\gamma-1}{2}\right) \text{Ma}^2\right]^{-\frac{\gamma}{1-1}}$$

The ratio of inlet-to-outlet stagnation pressures can be computed as a function of the stagnation pressure loss coefficient  and exit Mach number:
$$\begin{aligned} & Y=\frac{p_{01}-p_{02}}{p_{02}-p_2} \\
& Y=\frac{\left(\frac{p_{01}}{p_{02}}\right)-1}{1-\left(\frac{p_2}{p_{02}}\right)} \\
&\left(\frac{p_{01}}{p_{02}}\right)=1+Y \cdot\left[1-\left(\frac{p_2}{p_{02}}\right)\right] \\ 
\end{aligned}$$

After some algebra we obtain the following expression for the kinetic energy loss coefficient as a function of the stagnation pressure loss coefficient
$$\Delta \phi^2=\frac{\left(\frac{p_2}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}} \left(\left[1+Y \left(1- \frac{p_2}{p_{02}}\right)\right]^{\frac{\gamma-1}{\gamma}}-1\right)}{\left[1+Y \left(1-\frac{p_2}{p_{02}}\right)\right]^{\frac{\gamma-1}{\gamma}}-\left(\frac{p_2}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}}}$$

Conversely, the stagnation pressure loss coefficient can be solved in terms of the kinetic energy loss coefficient:
$$ Y=\frac{\left[\left(\frac{1}{1-\Delta \phi^2}\right) \left(1-\Delta \phi^2 \left(\frac{p_2}{p_{02}}\right)^{-\frac{\gamma-1}{\gamma}}\right)\right]^{-\left(\frac{\gamma}{\gamma-1}\right)}-1}{1- \left(\frac{p_2}{p_{02}}\right)} $$

The conversion between kinetic energy and stagnation pressure loss coefficients is illustrated below:

<div style="text-align:center;">
    <img src="figures/kinetic_energy_to_stagnation_pressure_loss_coefficient.svg" style="width:45%;">   
    <img src="figures/stagnation_pressure_to_kinetic_energy_loss_coefficient.svg" style="width:45%;">
</div>



**Relation between enthalpy and stagnation pressure loss coefficients**

When working with enthalpy loss coefficients, then the conversion can be done in two steps.
- If the stagnation pressure loss coefficient is given, first calculate the kinetic energy loss coefficient and then retrieve the enthalpy loss coefficient as $\zeta = \frac{\Delta \phi^2}{1 - \Delta \phi^2}$
- If the the enthalpy loss coefficient is given, first calculate the kinetic energy loss coefficient as $\Delta \phi^2 = \frac{\zeta}{1 + \zeta}$ and then retrieve the stagnation pressure loss coefficient.

<div style="text-align:center;">
    <img src="figures/enthalpy_to_stagnation_pressure_loss_coefficient.svg" style="width:45%;">   
    <img src="figures/stagnation_pressure_to_enthalpy_loss_coefficient.svg" style="width:45%;">
</div>