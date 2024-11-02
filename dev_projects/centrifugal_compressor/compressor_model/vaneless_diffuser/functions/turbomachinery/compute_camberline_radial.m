function [x, y, r, theta, beta, phi, stagger, chord] = compute_camberline_radial(camberline_type, r_1, r_2, beta_1, beta_2, theta_0, u)

    % Compute camberline coordinates depending on input type
    if strcmp(camberline_type, "straight")
        [x, y, r, theta, beta, phi, stagger] = compute_camberline_straight_polar(r_1, r_2, beta_1, theta_0, u);
        
    elseif strcmp(camberline_type, "circular_arc")
        [x, y, r, theta, beta, phi, stagger] = compute_camberline_circular_arc_polar(r_1, r_2, beta_1, beta_2, theta_0, u);

    elseif strcmp(camberline_type, "linear_angle_change")
        [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_angle_change_polar(r_1, r_2, beta_1, beta_2, theta_0, u);

    elseif strcmp(camberline_type, "linear_slope_change")
        [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_slope_change_polar(r_1, r_2, beta_1, beta_2, theta_0, u);

    elseif strcmp(camberline_type, "circular_arc_conformal")
        [x, y, r, theta, beta, phi, stagger] = create_camberline_circular_arc_conformal(r_1, r_2, beta_1, beta_2, theta_0, u);

    elseif strcmp(camberline_type, "linear_angle_change_conformal")
        [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_angle_change_conformal(r_1, r_2, beta_1, beta_2, theta_0, u);
    
    elseif strcmp(camberline_type, "linear_slope_change_conformal")
        [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_slope_change_conformal(r_1, r_2, beta_1, beta_2, theta_0, u);
    
    else
        errID = 'myComponent:inputError';
        msgtext = ['The camberline input type is not valid\n' ...
                    'Valid types:\n' ...
                    '\t circular_arc \n' ...
                    '\t circular_arc_conformal \n' ...
                    '\t linear_angle_change \n' ...
                    '\t linear_angle_change_conformal \n' ...
                    '\t linear_slope_change \n' ...
                    '\t linear_slope_change_conformal \n'];
        fprintf("Input type: %s\n", string(camberline_type))
        ME = MException(errID,msgtext);
        throw(ME)

    end

    % Compute the blade chord (cosine theorem) 
%     chord = (sqrt((r_2/r_1)^2 - sin(stagger)^2) - cos(stagger))*r_1;
    chord = sqrt(r_1^2 + r_2^2 - 2*r_1*r_2*cos(theta(end) - theta(1)));

end


%% Radial cascade camberlines
function [x, y, r, theta, beta, phi, stagger] = compute_camberline_straight_polar(r_1, r_2, phi, theta_0, u)
    
    % Analytic calculation
    L = (sqrt((r_2/r_1)^2 - sin(phi)^2) - cos(phi));
    x = r_1*cos(theta_0) + u*L*cos(phi + theta_0);
    y = r_1*sin(theta_0) + u*L*sin(phi + theta_0);
    r = sqrt(x.^2 + y.^2);
    theta = atan2(y, x);
    beta = atan(sin(phi)./(sqrt((r/r_1).^2-sin(phi).^2)));
    d_theta = theta(end) - theta(1);
    stagger = atan2(r_2*sin(d_theta), (r_2*cos(d_theta) - r_1));
    phi = phi + theta_0;

end
   

function [x, y, r, theta, beta, phi, stagger] = compute_camberline_circular_arc_polar(r_1, r_2, beta_1, beta_2, theta_1, u)
    
    % Compute the stagger angle iteratively
    stagger = fzero(@(stagger) exit_metal_angle_error(stagger), beta_1+theta_1);

    % Compute chord
    c = r_1*(sqrt((r_2/r_1)^2-sin(stagger)^2)-cos(stagger));
    
    % Compute leading edge coordinates
    x1 = r_1*cos(theta_1);
    y1 = r_1*sin(theta_1);
    
    % Compute trailing edge coordinates
    x2 = x1 + c*cos(stagger+theta_1);
    y2 = y1 + c*sin(stagger+theta_1);
    
    % Calculate the parametric angle at the leading and trailing edges
    angle_1 = pi/2 - beta_1 - theta_1;
    angle_2 = 2*(pi/2 - stagger) - 2*theta_1 - angle_1;
    
    % Calculate blade coordinates
    angle = angle_1 + u*(angle_2 - angle_1);
    x = x1 + (x2-x1)*(cos(angle) - cos(angle_1))/(cos(angle_2) - cos(angle_1));
    y = y1 - (x2-x1)*(sin(angle) - sin(angle_1))/(cos(angle_2) - cos(angle_1));

    % Compute slope angles
    r = sqrt(x.^2 + y.^2);
    theta = atan2(y, x);
    beta = pi/2 - theta - angle;
    phi = pi/2 - angle;  % Where dy/dx = tan(phi)
    
    % Trick added to get good behavior for turbines
    % This is a quick fix, I do not understand well the origin of the
    % problem. It seems this parametrization of the camberline is not
    % effective because it is convoluted and requires equation solving
    if r_1 > r_2
        stagger = pi + stagger;
    end

    % Compute trailing edge metal angle residual
    function res = exit_metal_angle_error(stagger)
        c = r_1*(sqrt((r_2/r_1)^2-sin(stagger)^2)-cos(stagger));
        x1 = r_1*cos(theta_1);
        y1 = r_1*sin(theta_1);
        x2 = x1 + c*cos(stagger+theta_1);
        y2 = y1 + c*sin(stagger+theta_1);
        angle_1 = pi/2 - beta_1 - theta_1;
        angle_2 = 2*(pi/2 - stagger) - 2*theta_1 - angle_1;
        % theta_2 = mod(atan2(y2, x2), 2*pi); not as robust
        theta_2 = theta_1 + acos((r_1^2 + r_2^2 - c^2)/(2*r_1*r_2));
        trial_beta_2 = pi/2 - angle_2 - theta_2;
        res = beta_2 - trial_beta_2;
    end

end


function [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_angle_change_polar(r_1, r_2, beta_1, beta_2, theta_0, u)
    
    % Compure wrap angle distribution
    r = r_1 + u*(r_2 - r_1);
    if numel(u) > 1
        [r, theta] = ode45(@(r, theta) wrap_angle_ode(r, theta, r_1, r_2, beta_1, beta_2), r, theta_0);
        r = reshape(r, [1, numel(r)]); theta = reshape(theta, [1, numel(theta)]);
    else
        [r, theta] = ode45(@(r, theta) wrap_angle_ode(r, theta, r_1, r_2, beta_1, beta_2), [r_1-1e-9, r], theta_0);
        r = r(end); theta = theta(end);
    end

    x = r.*cos(theta);
    y = r.*sin(theta);
    d_theta = theta(end)-theta(1);
    stagger = atan2(r_2*sin(d_theta), (r_2*cos(d_theta) - r_1));

    % Compute slope angles
    beta = (r_2-r)/(r_2-r_1)*beta_1 + (r-r_1)/(r_2-r_1)*beta_2;
    phi = beta+theta;

    % Linear angle distribution
    function dtheta_dr = wrap_angle_ode(r, ~, r_1, r_2, beta_1, beta_2)
        beta_ = (r_2-r)/(r_2-r_1)*beta_1 + (r-r_1)/(r_2-r_1)*beta_2;
        dtheta_dr = tan(beta_)./r;
    end

end


function [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_slope_change_polar(r_1, r_2, beta_1, beta_2, theta_0, u)
    
    % Analytic calculation
    r = r_1 + u*(r_2 - r_1);
    theta = theta_0 + (r_2*tan(beta_1) - r_1*tan(beta_2))*log(r/r_1)/(r_2-r_1) - (tan(beta_1) - tan(beta_2))*(r-r_1)/(r_2-r_1);
    x = r.*cos(theta);
    y = r.*sin(theta);
    d_theta = theta(end) - theta(1);
    stagger = atan2(r_2*sin(d_theta), (r_2*cos(d_theta) - r_1));

    % Compute slope angles
    tan_beta = (r_2-r)/(r_2-r_1)*tan(beta_1) + (r-r_1)/(r_2-r_1)*tan(beta_2);
    beta = atan(tan_beta);
    phi = beta+theta;

%     % Numerical calculation
%     r = r_1 + u*(r_2 - r_1);
%     [r, theta] = ode45(@(r, theta) wrap_angle_ode(r, theta, r_1, r_2, beta_1, beta_2), r, theta_0);
%     x = r.*cos(theta);
%     y = r.*sin(theta);
%     d_theta = theta(end)-theta(1);
%     stagger = r_2*sin(d_theta)/(r_2*cos(d_theta) - r_1);
% 
%     function dtheta_dr = wrap_angle_ode(r, ~, r_1, r_2, beta_1, beta_2)
%         tan_beta = (r_2-r)/(r_2-r_1)*tan(beta_1) + (r-r_1)/(r_2-r_1)*tan(beta_2);
%         dtheta_dr = tan_beta./r;
%     end

end
   

%% Conformal transformations from cartesian
function [x, y, r, theta, beta, phi, stagger] = create_camberline_conformal(camberline_type, r_1, r_2, beta_1, beta_2, theta_0, u)
    
    % Compute the coordinates of a linear cascade
    x_1 = 0.0; y_1 = 0.0; c_ax = 1.0;
    [x, y, tan_beta] = compute_camberline_cartesian(camberline_type, x_1, y_1, beta_1, beta_2, c_ax, u);
    
    % Map the coordinates to a radial cascade
    [x, y] = apply_conformal_mapping(x, y, x_1, y_1, r_1, r_2, c_ax, theta_0);

    % Compute polar coordinates
    r = sqrt(x.^2 + y.^2);
    theta = atan2(y, x);

    % Compute slope angles
    beta = atan(tan_beta);
    phi = beta+theta;
    
    % Compute the stagger angle
    d_theta = theta(end)-theta(1);
    stagger = atan2(r_2*sin(d_theta), (r_2*cos(d_theta) - r_1));

end

function [x, y, r, theta, beta, phi, stagger] = create_camberline_circular_arc_conformal(r_1, r_2, beta_1, beta_2, theta_0, u)
    [x, y, r, theta, beta, phi, stagger] = create_camberline_conformal('circular_arc', r_1, r_2, beta_1, beta_2, theta_0, u);
end

function [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_angle_change_conformal(r_1, r_2, beta_1, beta_2, theta_0, u)
    [x, y, r, theta, beta, phi, stagger] = create_camberline_conformal('linear_angle_change', r_1, r_2, beta_1, beta_2, theta_0, u);
end

function [x, y, r, theta, beta, phi, stagger] = compute_camberline_linear_slope_change_conformal(r_1, r_2, beta_1, beta_2, theta_0, u)
    [x, y, r, theta, beta, phi, stagger] = create_camberline_conformal('linear_slope_change', r_1, r_2, beta_1, beta_2, theta_0, u);
end

