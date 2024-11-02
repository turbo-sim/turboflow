### Impeller

Given:
- Radius hub inlet, $r_{hub, in}$
- Radius tip inlet, $r_{tip, in}$
- Radius exit, $r_{out}$
- Width exit, $b_{out}$
- Inlet blade angle: $\theta_{in}$
- Exit blade angle: $\theta_{out}$
- Number of full blades: $z_{fb}$
- Number of splitter blades: $z_{sb}$
- Blade mean streamline meridional length: $L$
- Splitter blade mean streamline meridional length: $L_{sb}$

Calculated:
- Inlet mean radius: $r_{mean, in} = (r_{hub, in} + r_{tip, in})/2$
- Inlet area: $A_{in} = \pi(r_{tip, in}^2 - r_{hub, in}^2)$
- Exit area: $A_{out} = 2\pi r_{out}b_{out}$
- Effective number of blades: $z$
	- Aungier: $z = z_{fb} + z_{sb}L_{sb}/L$
	- Romei: $z = z_{fb} + z_{sb}0.75$
### Vaneless diffuser

Given:
- Radius exit, $r_{vld, out}$
- Mean wall cant angle: $\phi$
- Wall divergence semi-angle: $\psi$

Calculated:
- Exit width: $b_{vld, out} = b_{vld, in} + 2\tan(\psi)\frac{r_{vld, out} - r_{vld, in}}{\sin(\phi)}$
- Exit area: $A_{vld, out} = 2\pi r_{vld, out} b_{vld,out}$ 
### Vaned diffuser
From previous component:
- Radius inlet: $r_{in}$

Given:
- Radius exit, $r_{vd, out}$
- Width exit: $b_{vd_out}$
- Inlet blade angle: $\theta_{vd,in}$
- Exit blade angle: $\theta_{vd,out}$

Calculated:
- Exit area: $A_{vd, out} = 2\pi r_{vd, out} b_{vd_out}$
- Camber angle: $\Delta \theta_{vd} = \theta_{vd, out} - \theta_{vd, in}$
- Solidity: $\sigma_{vd} = z(r_{vd, out} - r_{vd, in})/(2\pi r_{vd, in} \sin \theta_{avg})$
- Location of maximum camber: $(a/c)_{vd} = [2-(\theta_{avg}-\theta_{vd, in})/(\theta_{vd, out}-\theta_{vd, in})]/3$
### Volute
Given: 
- Radius exit, $r_{v, out}$
Calculated:
- Exit area: $A_{vd, out} = \pi r_{v, out}^2$