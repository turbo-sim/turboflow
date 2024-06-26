.. _geometry:

Geometry
===================

Meanline models are based on a simplified geometric representation of the turbine, focusing on the parameters defining the flow areas at the inlet, throat, and exit of the cascades, along with several variables necessary for the evaluation of the loss correlations. 

Axial Turbine
--------------
This section gives a description of the geometry used in simulations of axial turbines. 
To evaluate the turbine model, the following geometrical variables are required for each cascade:

    - Inlet radius at hub, :math:`r_\mathrm{in, hub}`
    - Inlet radius at tip, :math:`r_\mathrm{out, tip}`
    - Exit radius at hub, :math:`r_\mathrm{in, hub}`
    - Exit radius at tip, :math:`r_\mathrm{out, tip}`
    - Pitch, :math:`s`
    - Chord, :math:`c`
    - Stagger angle, :math:`\xi`
    - Opening, :math:`o`
    - Leading edge metal angle, :math:`\theta_\mathrm{le}`
    - Leading edge wedge angle, :math:`We_\mathrm{le}`
    - Leading edge diameter, :math:`d_\mathrm{le}`
    - Trailing edge thickness, :math:`t_\mathrm{te}`
    - Maximum blade thickness, :math:`t_\mathrm{max}`
    - Tip clearance, :math:`t_\mathrm{cl}`
    - Throat location factor, :math:`loc_\mathrm{throat}`

From this the mean radius at the inlet and exit can be calculated:

.. math::

    r_\mathrm{mean} = \frac{r_\mathrm{hub} + r_\mathrm{tip}}{2} 

And the blade height:

.. math::

    H = r_\mathrm{tip} - r_\mathrm{hub}

The mean blade height is the mean of the inlet and exit blade height:

.. math::

    H_\mathrm{mean} = \frac{H_\mathrm{in} + H_\mathrm{out}}{2}

The areas are calculated according to the following function:

.. math::

    A = \pi (r_\mathrm{tip}^2 - r_\mathrm{hub}^2)

The axial chord is calculated from the stagger angle and chord:

.. math::

    c_\mathrm{ax} = c\cos(\xi)

The flaring angle is calculated from the blade heights and axial chord:

.. math::

    \delta_\mathrm{fl} = \arctan(\frac{H_\mathrm{out} - H_\mathrm{in}}{2c_\mathrm{ax}})

The throat radius at hub to tip depend on the flaring and the throat location factor:

.. math::

    r_\mathrm{throat} = (1-loc_\mathrm{throat})r_\mathrm{in} + loc_\mathrm{throat}r_\mathrm{out}

The blade height at the throat is calculated similarly as for the ither planes and the throat area as

.. math::

    A_\mathrm{throat} = 2\pi r_\mathrm{throat, mean} H_\mathrm{throat} \frac{o}{s}

The gauging angle is calculated from the throat and exit area:

.. math:: 

    \beta_g = \arccos(\frac{A_\mathrm{throat}}{A_\mathrm{out}})

The geometry is illustrated below, both in axial-radial direction (left) and for the cascade profile geometry (right). 

.. image:: ../images/geometry_radial_axial.png
   :alt: Turbine geometry in axial-radial plane.
   :scale: 60%

.. image:: ../images/geometry_cascade.png
   :alt: Geometry of cascade profiles.
   :scale: 60%

..     The stagger angle and maximum thickness is for now required as an input variable for the performance prediction mode, while calculated for design optimization.
..     This causes a slight inconsistency between the two modes, which could/should be fixed. 
