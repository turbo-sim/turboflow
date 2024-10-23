Meroni: 
- Assume purely no tangential component at the volute exit
- Assume incompressible flow
- Enthalpy loss calculated with an empirical correlation
$\rho_{out} = \rho_{in}$
$v_{out} = \dot{m}/\rho_{out}A_{out}$
$h_{out} = h_{0, in} - 0.5v_{out}^2$
$s_{out} = f(h_{out}, \rho_{out})$
$p_{0,out} = f(h_{0, in}, s_{out})$
$h_{0, out} - h(p_{0,out}, s_{in}) = k_vv_{in}^2$ 

- This systems seems overdefined? 
- $h(p_{0,out}, s_{in})$ can now be calculated both from the empirical correlation and the exit total pressure

Aungier:
- Make one of the assumption as above (either compressible or velocity)
- Need one independent if only one of the assumptions are made, and the loss correlation is more complex than the one above
- Just a more complex loss correlation