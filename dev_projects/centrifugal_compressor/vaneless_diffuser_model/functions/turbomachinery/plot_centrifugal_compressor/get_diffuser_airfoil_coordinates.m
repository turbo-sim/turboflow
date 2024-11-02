function coords = get_diffuser_airfoil_coordinates(geom, N_points)

    % Define camberline parameters
    r_3 = geom.r_3;
    r_5 = geom.r_5;
    beta_1 = geom.alpha_3b*pi/180;
    beta_2 = geom.alpha_5b*pi/180;
    theta_0 = 0*pi/180;
    d_theta = 2*pi/geom.Z_vd;
    camberline_type = 'circular_arc_conformal';
    
    % Define thickness distribution parameters
    loc_max = 0.300;
    thickness_max = geom.t_b_3;
    thickness_trailing = 0.5*geom.t_b_3;
    radius_leading = 0.1*geom.t_b_3;
    wedge_trailing = 0.0*pi/180;

    % Compute blade coordinates
    [x_b, y_b] = compute_blade_coordinates_radial(camberline_type, r_3, r_5, beta_1, beta_2, theta_0, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading, N_points);
    x_b = x_b'; y_b = y_b';
    r_b = sqrt(x_b.^2 + y_b.^2);
    r_5_frac = (r_b - r_3)/(r_5 - r_3);
    r_3_frac = (r_5 - r_b)/(r_5 - r_3);
    xyz_h = [x_b, y_b, geom.L_z - 0*r_b]';
    xyz_s = [x_b, y_b, geom.L_z - geom.b_3*r_3_frac - geom.b_5*r_5_frac]'; % Linear variation (convex combination) of channel width along the radius
    
    % Process coordinates
    xyz = [xyz_h, xyz_s];
  
    % Calculate the throat at the leading and trailing edges
    calculation_method = 'intersection';
    sol_in = compute_leading_edge_throat(camberline_type, r_3, r_5, beta_1, beta_2, d_theta, theta_0, calculation_method);
    sol_out = compute_trailing_edge_throat(camberline_type, r_3, r_5, beta_1, beta_2, d_theta, theta_0, calculation_method);

    % Organize coordinates
    u1 = linspace(sol_in.u_throat, 1.00, ceil(N_points/2))';
    u2 = linspace(sol_out.u_throat, 0.00, ceil(N_points/2))';
    [x1, y1] = compute_camberline_radial(camberline_type, r_3, r_5, beta_1, beta_2, theta_0, u1);
    [x2, y2] = compute_camberline_radial(camberline_type, r_3, r_5, beta_1, beta_2, theta_0+d_theta, u2);

    xyz_channel = [x1, y1, geom.L_z-geom.b_3 + 0*u1 
                   x2, y2, geom.L_z-geom.b_3 + 0*u2]';

    xyz_inlet = [sol_in.x_leading, sol_in.y_leading, geom.L_z 
                 sol_in.x_throat,  sol_in.y_throat, geom.L_z
                 sol_in.x_throat,  sol_in.y_throat, geom.L_z-geom.b_3
                 sol_in.x_leading, sol_in.y_leading, geom.L_z-geom.b_3]';

    xyz_outlet = [sol_out.x_trailing, sol_out.y_trailing, geom.L_z 
                  sol_out.x_throat,  sol_out.y_throat, geom.L_z
                  sol_out.x_throat,  sol_out.y_throat, geom.L_z-geom.b_3
                  sol_out.x_trailing, sol_out.y_trailing, geom.L_z-geom.b_3]';

    % Store coordinates in structure
    coords.xyz = xyz;
    coords.xyz_h = xyz_h;
    coords.xyz_s = xyz_s;
    coords.xyz_channel = xyz_channel;
    coords.xyz_inlet = xyz_inlet;
    coords.xyz_outlet = xyz_outlet;
    
end