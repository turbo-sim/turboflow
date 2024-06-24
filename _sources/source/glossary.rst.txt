
.. _glossary:


Glossary
============


.. _glossary_turbomachinery:

Axial turbines
-----------------


.. glossary::
    :sorted:

    Blade
        Aerodynamic profile used in turbomachines to change the velocity and direction of the working fluid. Depending its function, a blade can either be stationary (stator blades) or rotate (rotor blades). 

    Blade, Stator
        Stationary blade that deflects and accelerates the fluid, preparing it for the subsequent rotor blade.
        Stator blades are also know as vanes or nozzles.

    Blade, Rotor
        Rotating blade that deflects the fluid and extract work due to the fluid's change in angular momentum.
        Rotor blades are also known as buckets.

    Blade, Pressure side
        The concave surface of the blade characterized by higher pressure and lower velocity, typically experiencing stable flow conditions.

    Blade, Suction side
        The convex surface of the blade with reduced pressure and increased velocity, potentially prone to flow instabilities and separation.

    Blade, Camber line
        Curve half-way between the suction and pressure surfaces of the blade.

    Blade, Camber length
        The distance measured along the camber line from the leading edge to the trailing edge.

    Blade, Leading edge
        The foremost point where the suction and pressure surfaces of the blade meet.

    Blade, Trailing edge
        The rearmost point where the suction and pressure surfaces converge.

    Blade, Chord line
        A straight line connecting the blade's leading and trailing edges.

    Blade, Chord length
        The straight-line distance between the blade's leading and trailing edges.

    Blade, Axial chord length
        The length of the chord's projection aligned with the turbine's axial direction.

    Blade, Stagger angle
        Angle formed between the chord line and the axial direction. It relates the actual and axial chord lengths, such that the cosine of this angle equals the ratio of the axial chord to the actual chord length.
        Angle between the chord line and the turbine axial direction. The cosine of the stagger angle is equal to the ratio of axial chord to true chord.

    Blade, Setting angle
        Same as :term:`stagger angle <Blade, Stagger angle>`.

    Blade, camber
        Distance between the camber line and the chord line measured perpendicular to the chord line.

    Blade, thickness
        Distance between the pressure and suction surfaces, measured perpendicular to the camber line.

    Blade, spacing
        The circumferential distance separating corresponding points on consecutive blades within a cascade.

    Blade, Pitch
        Same as :term:`blade spacing <Blade, Spacing>`

    Blade, Pitch-to-chord ratio
        Dimensionless ratio comparing the blade spacing to the chord length of a blade. This parameter has a string influence on the profile losses within a cascade.

    Blade, Solidity
        The reciprocal of the :term:`pitch-to-chord ratio <Blade, Pitch-to-chord ratio>`

    Blade, Axial pitch-to-chord ratio
        Dimensionless ratio comparing the axial chord length of a blade to the spacing between adjacent blades in a cascade.
        
    Blade, Height
        Difference between the blade radius at the tip and the blade radius at the root.

    Blade, Aspect ratio
        Ratio of the blade height to the chord length of the blades.

    Blade, Axial aspect ratio 
        Ratio of the blade height to the axial chord length of the blades.

    Blade, Root
        Section of the blade attached to the casing for stator blades (shroud) and to the disks for rotor blades (hub).

    Blade Tip
        End section of the blade. Rotor-blade tip sections are at the shroud and stator-blade tip sections are at the hub.

    Blade, Hub-to-tip ratio
        Ratio of the blade radius at the hub to the blade radius at the tip.


    Cascade
        Row of circumferentially spaced blades designed to guide and modify the direction and/or velocity of the fluid.

    Cascade, Linear
        Configuration where turbine blades are aligned in a straight row, typically employed for aerodynamic testing under two-dimensional flow conditions.

    Cascade, Annular
        Configuration where turbine blades are assempled in a ring-like arrangement. Annular cascades are found in actual turbomachines or in annular test rigs that emulate the geometry of turbomachines.

    Cascade, Stator
        Row of stationary blades that guides and accelerates the working fluid to prepare it for the rotor cascade.

    Cascade, Rotor
        Row of rotating blades that extracts energy from the fluid and transforms it into mechanical work.

    Stage
        A unit in a turbine that consists of a stator cascade followed by a rotor cascade.

    Stage, Impulse
        A stage in which the majority of the pressure (or enthalpy) drop occurs in the stator, resulting in a degree of reaction close to zero. In such stages, the static pressure in the rotor cascade is approximately constant and work is produced due to the change of direction of the fluid.

    Stage, Reaction
        A stage in which both the rotor and the stator contribute to the pressure (or enthalpy) drop of the working fluid. The degree of reaction is between zero and one depending on the distribution of the energy conversion. Many reaction stages are designed with a degree of reaction about 50% at nominal conditions.

    Stage, Degree of reaction
        Dimensionaless quantity defined as the ratio of the static pressure (or enthalpy) change across the rotor to the static pressure (or enthalpy) change across the entire stage. It can be interpreted as the fraction of fluid expansion that takes place within the rotor.

    Stage, Spacing
        Axial distance between the outlet of the stator stage and the inlet of the rotor stage.

    Angle, flow
        Angle between the absolute or relative velocity vector at a flow station and the axial direction.

    Angle, blade inlet
        Angle between the tangent of the camber line  at a flow station and the axial direction.

    Angle, incidence
        Difference between the relative inlet flow angle and the metal angle at the leading edge.

    Angle, deviation
        Difference between the relative outlet flow angle and the metal angle at the trailing edge.

    Angle, flaring
        Angle defined by the increase of blade height in the axial

    Angle, metal 
        Angle formed by the blade surface of a blade relative to a reference direction.

    Turbine
        Machine that extracts energy from a fluid flow and converts it into useful work.

    Turbine, axial
        Turbine in which the flow is parallel to the shaft.

    Turbine, radial inflow
        Turbine in which the flow is in the radial inward direction.

    Turbine, radial outflow
        Turbine in which the flow is in the radial outward direction.

    Turbine, mixed-flow
        Turbine in which the flow is deflected from the radial to the axial direction.

    Casing
        Stationary part of the turbine that contains the rest of the components.

    Casing, Hub
        Surface defining the inner diameter of the flow, see shroud.

    Casing, Shroud
        Surface defining the outer diameter of the flow, see hub.
    
    Annulus
        Annular duct defined by the shroud and the hub surfaces.



.. Sweep angle, lean angle, dihedral. Relation to the axis of blade stacking in axial turbines


.. _glossary_optimization:


Optimization
-----------------

.. glossary::
    :sorted:

    Gradient-based optimization
        An optimization method that uses the gradient of the objective function to guide the search for the minimum or maximum. It typically involves iterative steps in the direction of the negative gradient for minimization problems.

    Gradient-free optimization
        An optimization method that does not require the gradient of the objective function. These methods are useful for problems where the gradient is not available or difficult to compute. Examples include genetic algorithms, simulated annealing, and particle swarm optimization.

    Objective function
        The function that is being optimized, which can be either minimized or maximized. It represents the goal of the optimization process.

    Equality constraint
        A type of constraint in an optimization problem that requires a specific condition to be exactly met. It is typically represented as \( g(x) = 0 \).

    Inequality constraint
        A type of constraint that restricts the values of the variables to a certain range, typically represented as \( h(x) \leq 0 \) or \( h(x) \geq 0 \).

    Lagrangian function
        A function used in constrained optimization problems that combines the objective function and the constraints using Lagrange multipliers. It is used to convert a constrained problem into an unconstrained one.

    Independent variables
        The variables in an optimization problem that can be adjusted or controlled in order to find the optimal solution. They are also known as decision variables.

    Design variables
        Another term for independent variables, specifically in the context of design optimization. These are the parameters that can be modified to achieve the optimal design.

    Degrees of freedom
        The number of independent parameters that can vary in a system or optimization problem. It represents the number of independent directions in which the system can move.

    Optimization bounds
        The constraints that define the minimum and maximum values that the design variables can take. These bounds limit the search space for the optimization process.

    Gradient vector
        A vector that contains the partial derivatives of the objective function with respect to all the design variables. It points in the direction of the steepest ascent or descent.

    Jacobian matrix
        A matrix of all first-order partial derivatives of a vector-valued function. In optimization, it represents the gradients of the constraints with respect to the design variables.

    Hessian matrix
        A square matrix of second-order partial derivatives of a scalar-valued function. It describes the local curvature of the objective function and is used in second-order optimization methods.

    Line search
        An iterative optimization technique that involves moving along a search direction to find an acceptable step size that reduces the objective function.

    Trust region
        An optimization technique where the search for the optimal solution is restricted to a region around the current point, and this region is adjusted based on the success of the optimization steps.

    Feasibility
        The condition of satisfying all constraints in an optimization problem. A feasible solution meets all equality and inequality constraints.

    Finite differences
        A numerical method used to approximate the derivatives of a function by using the differences in function values at specific points. It is commonly used in gradient-free optimization methods.

    Step size
        The magnitude of the change applied to the design variables in each iteration of an optimization algorithm. It determines how far the variables move in the search space during each iteration.



