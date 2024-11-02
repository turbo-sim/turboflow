function [x_backplate, y_backplate] = get_impeller_backplate_coordinates(geom, N_points)
    
    arguments
        geom (1, 1) struct
        N_points (1, 1) double = 50;
    end

    check_valid_options(geom.backplate_type, {'linear', 'circular_arc'});
    if strcmp(geom.backplate_type, 'circular_arc')

        % Circular arc parametrization
        x_1 = geom.t_backplate_1;
        x_2 = geom.t_backplate_2;
        y_1 = geom.r_backplate_1;
        y_2 = geom.r_backplate_2;
    
        % Compute the x-coordinate of the center
        alpha = 0.5*((x_1+x_2) + (y_2-y_1)^2/(x_2-x_1));
        beta = ((x_2 - x_1)/(y_2 - y_1))^2;
        a = 1 + beta;
        b = -2*(x_1+alpha*beta);
        c = beta*alpha^2 + x_1^2 - geom.R_backplate^2;
        x_c = max(roots([a, b, c]));
    
        % Compute the y-coordinate of the center
        alpha = 0.5*((y_1+y_2) + (x_2-x_1)^2/(y_2-y_1));
        beta = ((y_2 - y_1)/(x_2 - x_1))^2;
        a = 1 + beta;
        b = -2*(y_1+alpha*beta);
        c = beta*alpha^2 + y_1^2 - geom.R_backplate^2;
        y_c = max(roots([a, b, c]));
        
        % Compute the limiting angles of the circular arc
        theta_1 = mod(atan2(y_1-y_c, x_1-x_c), 2*pi);
        theta_2 = mod(atan2(y_2-y_c, x_2-x_c), 2*pi);
        theta = linspace(theta_1, theta_2, N_points)';
        x_circle = x_c+geom.R_backplate*cos(theta);
        y_circle = y_c+geom.R_backplate*sin(theta);
    
        % Define the backplate x-y (r-z) coordinates
        x_backplate = [geom.L_z
                       geom.L_z + geom.t_backplate_1
                       geom.L_z + x_circle
                       geom.L_z + geom.t_backplate_2];
    
        y_backplate = [geom.r_2
                       geom.r_2
                       y_circle
                       geom.r_shaft];

    elseif strcmp(geom.backplate_type, 'linear')

        % Define the backplate x-y (r-z) coordinates
        x_backplate = [geom.L_z
                       geom.L_z + geom.t_backplate_1
                       geom.L_z + geom.t_backplate_1
                       geom.L_z + geom.t_backplate_2
                       geom.L_z + geom.t_backplate_2];
    
        y_backplate = [geom.r_2
                       geom.r_2
                       geom.r_backplate_1
                       geom.r_backplate_2
                       geom.r_shaft];

    else
        error('Invalid option')

    end


end