function [x, y, stagger, chord] = compute_blade_coordinates_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading, N_points)

    % Compute the camberline coordinates
    u = linspace(0, 1, ceil(1/3*N_points));
    [x_camber, y_camber, ~, theta, ~, phi, stagger, chord] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0, u);
           
    % Re-stagger the camberline
    x_norm = (x_camber-r_1*cos(theta_0))/chord;
    y_norm = (y_camber-r_1*sin(theta_0))/chord;
    [x_norm, ~] = rotate_counterclockwise_2D(x_norm, y_norm, -(stagger+theta_0));
    x_norm = abs(x_norm);

    % Compute thickness distribution
    half_thickness = compute_thickness_distribution_NACA_modified(x_norm, chord, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading);

    % Impose the thickness distribution
    x_lower = x_camber + half_thickness.*sin(phi);
    y_lower = y_camber - half_thickness.*cos(phi);
    x_upper = x_camber - half_thickness.*sin(phi);
    y_upper = y_camber + half_thickness.*cos(phi);

    % Compute camberline endpoint
    x_2 = r_1*cos(theta_0) + chord*cos(stagger + theta_0);
    y_2 = r_1*sin(theta_0) + chord*sin(stagger + theta_0);

    % Computate trailing edge radius
    radius_trailing = 0.5*thickness_trailing/cos(wedge_trailing/2);
   
    % Compute center of curvature location
    % sign(r2-r1) generarizes for r2>r1 (compressor) and r2<r1 (turbine)
    phi_2 = beta_2 + theta(end);
    x_c = x_2 - sign(r_2-r_1)*radius_trailing*sin(wedge_trailing/2)*cos(phi_2);
    y_c = y_2 - sign(r_2-r_1)*radius_trailing*sin(wedge_trailing/2)*sin(phi_2);

    % Compute circular arc limit angles
    angle_1 = +(pi/2 - wedge_trailing/2) + phi_2;
    angle_2 = -(pi/2 - wedge_trailing/2) + phi_2;
    angle = linspace(angle_1, angle_2, floor(1/3*N_points));

    % Compute trailing edge coordinates
    % sign(r2-r1) generarizes for r2>r1 (compressor) and r2<r1 (turbine)
    x_trailing = x_c + sign(r_2-r_1)*radius_trailing*cos(angle);
    y_trailing = y_c + sign(r_2-r_1)*radius_trailing*sin(angle);

    % Combine the coordinates of the lower, trailing and upper segments
    if r_1 > r_2
        x_trailing = flip(x_trailing);
        y_trailing = flip(y_trailing);
    end
    x = [x_lower, flip(x_trailing), flip(x_upper)];
    y = [y_lower, flip(y_trailing), flip(y_upper)];
    
%     % Testing
%     figure(); hold on
%     axis image
%     plot(x_2, y_2, 'ko')
%     plot(x_lower, y_lower, 'r')
%     plot(x_upper, y_upper, 'b')
%     plot(x_camber, y_camber, 'k:')
%     plot(x_c, y_c, 'g*')
%     plot(x_trailing, y_trailing, 'g')

end
