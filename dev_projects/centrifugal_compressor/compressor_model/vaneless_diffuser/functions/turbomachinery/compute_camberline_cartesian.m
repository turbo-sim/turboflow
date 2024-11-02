function [x, y, dydx, stagger, chord] = compute_camberline_cartesian(camberline_type, x_1, y_1, beta_1, beta_2, c_ax, u)

    % Compute camberline coordinates depending on input type
    if strcmp(camberline_type, "NACA")
        [x, y, stagger, dydx] = compute_camberline_NACA(x_1, y_1, beta_1, beta_2, c_ax, u);

    elseif strcmp(camberline_type, "circular_arc")
        [x, y, stagger, dydx] = compute_camberline_circular_arc(x_1, y_1, beta_1, beta_2, c_ax, u);

    elseif strcmp(camberline_type, "linear_angle_change")
        [x, y, stagger, dydx] = compute_camberline_linear_angle_change(x_1, y_1, beta_1, beta_2, c_ax, u);

    elseif strcmp(camberline_type, "linear_slope_change")
        [x, y, stagger, dydx] = compute_camberline_linear_slope_change(x_1, y_1, beta_1, beta_2, c_ax, u);

    else
        errID = 'myComponent:inputError';
        msgtext = ['The camberline input type is not valid\n' ...
                    'Valid types:\n' ...
                    '\t NACA \n' ...
                    '\t circular_arc \n' ...
                    '\t linear_angle_change\n' ...
                    '\t linear_slope_change\n'];
        fprintf("Input type: %s\n", string(camberline_type))
        ME = MException(errID,msgtext);
        throw(ME)
    end

    % Compute the blade chord
    chord = c_ax/cos(stagger);

end


function [x, y, stagger, dydx] = compute_camberline_circular_arc(x_1, y_1, beta_1, beta_2, c_ax, u)
    
    % Camberline described as a circular arc
    x_2 = x_1 + c_ax;
    stagger = (beta_1 + beta_2)/2;
    beta = beta_1 + u*(beta_2 - beta_1);
    if abs(beta_1-beta_2) > 1e-6
        x = x_1 + (x_2 - x_1)*(sin(beta) - sin(beta_1))/(sin(beta_2) - sin(beta_1));
        y = y_1 - (x_2 - x_1)*(cos(beta) - cos(beta_1))/(sin(beta_2) - sin(beta_1));
    else
        x_2 = x_1 + c_ax;
        x = x_1 + u*(x_2 - x_1);
        y = y_1 + (x-x_1)*tan(stagger);
    end

    dydx = tan(beta);

end


function [x, y, stagger, dydx] = compute_camberline_NACA(x_1, y_1, beta_1, beta_2, c_ax, u)

    % Compute the stagger angle as the arithmetic mean of the metal angles
    % (this choice is arbitratry, but it leads to a camberline with
    % no curvature discontinuity at the camber location)
    stagger = (beta_1 + beta_2)/2.0;

    % Compute the camber location [p=0.5 for stagger=(beta_1+beta_2)/2]
    p = tan(beta_2-stagger) / (tan(beta_2-stagger) - tan(beta_1-stagger));

    % Compute the camber magniture
    m = p/2*tan(beta_1 - stagger);

    % Compute the coordinates of the normalized blade
    x_c = u;
    if p > 0
        y_c = m/p^2*(2*p*x_c-x_c.^2).*(0 <= x_c & x_c <= p) + m/(1-p)^2*(1-2*p+2*p*x_c-x_c.^2).*(p < x_c & x_c <= 1);
        dy_c = 2*m/p^2*(p-x_c).*(0 <= x_c & x_c <= p) + 2*m/(1-p)^2*(p-x_c).*(p < x_c & x_c <= 1);
    else
        % Avoid division by zero for the case of symmetric airfoils
        y_c = 0*x_c;
        dy_c = 0*x_c;
    end
       
    % Rotate and scale the coordinates
    chord = c_ax/cos(stagger);
    R = [cos(stagger), -sin(stagger); +sin(stagger), cos(stagger)];
    coords = [x_1; y_1] + chord*R*[x_c; y_c];
    x = coords(1,:); y = coords(2,:);

    % Compute the slope of the scaled carmberline
    beta = atan(dy_c) + stagger;
    dydx = tan(beta);

end


function [x, y, stagger, dydx] = compute_camberline_linear_angle_change(x_1, y_1, beta_1, beta_2, c_ax, u)
    
    % I did not test the sign of arctan() extensively
    if abs(beta_1-beta_2) > 1e-6
        stagger = atan(-log(cos(beta_2)/cos(beta_1))/(beta_2 - beta_1 + 1e-6));
        beta = beta_1 + u*(beta_2 - beta_1);
        x_2 = x_1 + c_ax;
        x = x_1 + (beta - beta_1)/(beta_2 - beta_1)*(x_2 - x_1);
        y = y_1 - (x_2 - x_1)/(beta_2 - beta_1)*log(cos(beta)/cos(beta_1));
    else
        stagger = beta_1;
        x_2 = x_1 + c_ax;
        x = x_1 + u*(x_2 - x_1);
        y = y_1 + (x-x_1)*tan(stagger);
    end
        
    beta = beta_1 + (beta_2 - beta_1)*(x - x_1)/(x_2-x_1);
    dydx = tan(beta);


end


function [x, y, stagger, dydx] = compute_camberline_linear_slope_change(x_1, y_1, beta_1, beta_2, c_ax, u)

    % Linear change of slope (dy/dx=tan(beta)) with axial chord
    x_2 = x_1 + c_ax;
    x = x_1 + u*(x_2 - x_1);
    temp = 1/2*tan(beta_1)*(1 - ((x_2-x)/(x_2-x_1)).^2) + 1/2*tan(beta_2)*((x-x_1)/(x_2-x_1)).^2;
    y = y_1 + temp*(x_2 - x_1);
    stagger = atan(0.5*(tan(beta_1)+tan(beta_2)));
    dydx = tan(beta_1)*(x_2 - x)/(x_2-x_1) + tan(beta_2)*(x - x_1)/(x_2-x_1);

end


