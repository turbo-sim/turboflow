function [x, y, stagger, chord] = compute_blade_coordinates_cartesian(camberline_type, x_1, y_1, beta_1, beta_2, chord_ax, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading, N)

    % Compute the camberline coordinates
    u = linspace(0, 1, N);
    [x_camber, y_camber, dydx, stagger, chord] = compute_camberline_cartesian(camberline_type, x_1, y_1, beta_1, beta_2, chord_ax, u);

    % Re-stagger the camberline
    x_norm = (x_camber-x_1)/chord;
    y_norm = (y_camber-y_1)/chord;
    [x_norm, ~] = rotate_counterclockwise_2D(x_norm, y_norm, -stagger);
    
    % Compute thickness distribution
    half_thickness = compute_thickness_distribution_NACA_modified(x_norm, chord, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading);
    
    % Impose the thickness distribution
    theta = atan(dydx);
    x_lower = x_camber + half_thickness.*sin(theta);
    y_lower = y_camber - half_thickness.*cos(theta);
    x_upper = x_camber - half_thickness.*sin(theta);
    y_upper = y_camber + half_thickness.*cos(theta);

    % Compute camberline endpoint
    x_2 = x_1 + chord_ax;
    y_2 = y_1 + chord_ax*tan(stagger);

    % Computate trailing edge radius
    radius_trailing = 0.5*thickness_trailing/cos(wedge_trailing/2);
   
    % Compute center of curvature location
    x_c = x_2 - radius_trailing*sin(wedge_trailing/2)*cos(beta_2);
    y_c = y_2 - radius_trailing*sin(wedge_trailing/2)*sin(beta_2);

    % Compute circular arc limit angles
    phi_1 = +(pi/2 - wedge_trailing/2) + beta_2;
    phi_2 = -(pi/2 - wedge_trailing/2) + beta_2;
    angle = linspace(phi_1, phi_2, N/2);

    % Compute trailing edge coordinates
    x_trailing = x_c + radius_trailing*cos(angle);
    y_trailing = y_c + radius_trailing*sin(angle);

    % Combine the coordinates of the lower, trailing and upper segments
    x = [x_lower, flip(x_trailing), flip(x_upper)];
    y = [y_lower, flip(y_trailing), flip(y_upper)];

end
