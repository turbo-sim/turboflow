function half_thickness = compute_thickness_distribution_NACA_modified(x_norm, chord, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading)

    % Compute the half-thickness distribution of an airfoil using a
    % variation of the NACA series 4 parametrization method described in:
    % "The characteristics of 78 related airfoil sections from tests in the
    % variable-density wind tunnel"

    % Parameters
    % - x_norm: Normalized chordwise coordinate
    % - thickness_max: Maximum thickness
    % - loc_max: Location of the point of maximum thickness
    % - thickness_trailing: trailing edge thickness
    % - wedge_trailing: wedge full angle at the trailing edge
    % - radius_leading: leading edge radius of curvature
    
    % Radius of curvature
    i = 1;
    LHS(i, :) = [1 0 0 0 0];
    RHS(i, 1) = sqrt(2*(radius_leading/chord));

    % Maximum thickness
    i = i + 1;
    LHS(i, :) = [sqrt(loc_max) loc_max loc_max^2 loc_max^3 loc_max^4];
    RHS(i, 1) = 0.5*(thickness_max/chord);
    
    % Zero-slope at maximum thickness point
    i = i + 1;
    LHS(i, :) = [0.5/sqrt(loc_max) 1.0 2*loc_max 3*loc_max^2 4*loc_max^3];
    RHS(i, 1) = 0.00;
    
    % Trailing edge thickness
    i = i + 1;
    LHS(i, :) = [1.0 1.0 1.0 1.0 1.0];
    RHS(i, 1) = 0.5*(thickness_trailing/chord);

    % Trailing edge slope (wedge angle)
    i = i + 1;
    slope_trailing = -tan(wedge_trailing/2);
    LHS(i, :) = [0.5 1.0 2.0 3.0 4.0];
    RHS(i, 1) = slope_trailing;
    
    % Solve linear system of equations
    coeff = LHS\RHS;

    % Compute the thickness distribution
    A = coeff(1);
    B = coeff(2);
    C = coeff(3);
    D = coeff(4);
    E = coeff(5);
    half_thickness = chord*(A*x_norm.^0.5 + B*x_norm + C*x_norm.^2 + D*x_norm.^3 + E.*x_norm.^4);

end
