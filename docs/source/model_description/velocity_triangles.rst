.. _velocity_triangles:

Velocity triangles
===================

The figure below illustrate the velocity triangles, which is evaluate at each station of the cascade. The angles are measured from the axial direction, and they are positive in counter-clockwise direction.

.. image:: ../images/velocity_triangles.png
   :alt: Velocity triangles.
   :scale: 60%

The equations below show the equations to determine the velocity triangle at the inlet and exit of a cascade.

.. math::

    \begin{align}
        &v_{m,\mathrm{in}} = v_\mathrm{in} \cdot \cos(\alpha_\mathrm{in})  && w_{m, \mathrm{out}} = w_\mathrm{out} \cdot \cos(\beta_\mathrm{out})  \\
        &v_{\theta, \mathrm{in}} = v_\mathrm{in} \cdot \sin(\alpha_\mathrm{in}) && v_{\theta, \mathrm{out}} = w_\mathrm{out} \cdot \sin(\beta_\mathrm{out})\\ 
        &w_{\theta, \mathrm{in}} = v_{\theta, \mathrm{in}} - u_\mathrm{in} &&  v_{\theta, \mathrm{out}} = w_{\theta, \mathrm{out}} + u_\mathrm{out}\\
        &w_{m, \mathrm{in}} = v_{m, \mathrm{in}} && v_{m, \mathrm{out}} = w_{m, \mathrm{out}}\\
        &w_\mathrm{in} = (w_{m, \mathrm{in}}^2 + w_{\theta, \mathrm{in}}^2)^{1/2} && v_\mathrm{out} = (v_{m, \mathrm{out}}^2 + v_{\theta, \mathrm{out}}^2)^{1/2} \\
        &\beta_\mathrm{in} = \arctan(w_{\theta, \mathrm{in}}/w_{m, \mathrm{in}}) && \alpha_\mathrm{out} = \arctan(v_{\theta, \mathrm{out}}/v_{m, \mathrm{out}}).
    \end{align}

where:

    - :math:`v` and :math:`w` refer to the absolute and relative velocity.
    - :math:`\alpha` and :math:`\beta` refer to the aboslute and relatove flow angle.
    - :math:`u` refer to the blade speed.
    - Subscripts :math:`\mathrm{in}` and :math:`\mathrm{out}` refer to the inlet and exit plane of the cascade.
    - Subscripts :math:`m` and :math:`\theta` refer to the meridonial and tangential vector component. 

For the inlet plane, the absolute velocity, absolute flow angle and balde speed is required to calculate the full velocity triangle.
For the exit plane, the relative velocity, relative flow angle and blade speed is required to calculate all velocitry triangles properties. 