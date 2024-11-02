
function out = compute_trailing_edge_throat(camberline_type, r_1, r_2, beta_1, beta_2, d_theta, theta_0, method)

    % Compute the throat at the trailing edge of a radial cascade
    % The throat can be computed by two methods:
    %
    %   1. Projection method 
    %      The throat is computed as the projection of the trailing edge
    %      into the camberline of the adjacent blade. The line connecting
    %      the trailing edge with the projected point is normal to the
    %      camberline. As a result, the direction of the throat line is not
    %      exactly aligned with beta_in+d_theta/2
    %
    %   2. Intersection method
    %      The throat is computed as the as te intersection of the blade
    %      camberline with the throat line. The throat line is defined as a
    %      stright line that passes through the trailing edge of the blade
    %      and has a slope given by beta_in+d_theta/2
    %
    % Both methods yield similar results in most cases, but the
    % intersection method should be preferred because the slope of the
    % throat line is more realistic. 
    %
    % This function does not take into account the thickness of the blades
    % to compute the throat (big limitation when there are many blades and
    % the resulting metal blockage is high
    %

    arguments
        camberline_type (1, 1) string
        r_1 (1, 1) double 
        r_2 (1, 1) double 
        beta_1 (1, 1) double 
        beta_2 (1, 1) double 
        d_theta (1, 1) double 
        theta_0 (1, 1) double 
        method (1, 1) string = 'intersection';
    end
    
    if strcmp(method, 'projection')

        % Compute the coordinates of the trailing edge
        [x_trailing, y_trailing] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0, 1.00);
        
        % Define optimization options
        options = optimoptions('fmincon', ...
                               'Display','none', ...
                               'Algorithm', 'active-set');
    
        % Solve the point projection problem
        u0 = 0.9;
        u_min = fmincon(@evaluate_distance, u0, [], [], [], [], 0, 1, [], options);
        
        % Get the throat coordinates
        u_throat = u_min;
        w_throat = evaluate_distance(u_min);
        [x_throat, y_throat] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0+d_theta, u_throat);
        x_throat_check = [];
        y_throat_check = [];
            
    elseif strcmp(method, 'intersection')

        % Compute the coordinates of the trailing edge
        [x_trailing, y_trailing, ~, ~, ~, phi_trailing] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0, 1.00);

        % Define the throat slope
        % Plus sign because initial condition is theta_0
        slope_angle = phi_trailing + d_theta/2; 

        % Define optimization options
        options = optimoptions('fsolve', ...
                               'Display','none');
    
        % Solve the point projection problem
        x_0 = [0.9, r_2-r_1];
        x_min = fsolve(@evaluate_intersection, x_0, options);
        
        % Get the throat coordinates
        u_throat = x_min(1);
        w_throat = x_min(2);
        x_throat = x_trailing - w_throat*sin(slope_angle);
        y_throat = y_trailing + w_throat*cos(slope_angle);
        [x_throat_check, y_throat_check] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0+d_theta, u_throat);
    
    else

        error('Invalid solution method')

    end

    % Export solution
    out.u_throat = u_throat;
    out.w_throat = w_throat;
    out.x_trailing = x_trailing;
    out.y_trailing = y_trailing;
    out.x_throat = x_throat;
    out.y_throat = y_throat;
    out.x_throat_check = x_throat_check;
    out.y_throat_check = y_throat_check;

    % Check computations
    check = w_throat - sqrt((x_throat - x_trailing)^2 + (y_throat - y_trailing)^2);

    % Define objective functios
    function norm = evaluate_distance(u)
        
        % Compute camberline coordinates
        [x_c, y_c] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0+d_theta, u);

        % Compute distance between current coordinates and trailing edge
        norm = sqrt((x_c - x_trailing).^2 + (y_c - y_trailing).^2);

    end

    function res = evaluate_intersection(x)

        % Rename variables
        u = x(1);
        v = x(2);
        
        % Compute camberline coordinates
        [x_camber, y_camber] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0+d_theta, u);

        % Compute throat line coordinates
        x_throat = x_trailing - v*sin(slope_angle);
        y_throat = y_trailing + v*cos(slope_angle);

        % Compute distance between current coordinates and trailing edge
        res(1) = x_throat - x_camber;
        res(2) = y_throat - y_camber;

    end

end

