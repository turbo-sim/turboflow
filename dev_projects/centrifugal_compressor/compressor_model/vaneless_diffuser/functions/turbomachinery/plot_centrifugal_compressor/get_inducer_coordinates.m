% Compute the coordinates of the inducer for 3D visualization
% Different types of inducer noses are supported
%   - Flat
%   - Circular arc
%   - Elliptical arc
%

function [z_inducer, r_inducer] = get_inducer_coordinates(geom)

    valid_options = {'flat', 'circular', 'elliptic'};
    if ~any(strcmp(geom.inducer_type, valid_options))
        error("Inducer type is '%s', but it must be: %s.", geom.inducer_type, strjoin(valid_options, ', '))
    end

    if strcmp(geom.inducer_type, "flat")
        z_inducer = [-geom.L_inducer, -geom.L_inducer, 0]';
        r_inducer = [0, geom.r_0_h, geom.r_1_h]';
    elseif strcmp(geom.inducer_type, "circular")
        % Straight line + tangent circle
        phi = atan2(geom.r_1_h - geom.r_0_h, geom.L_inducer);
        z_c = -geom.L_inducer + geom.r_0_h*tan(phi);
        r_c = 0.00;
        r_inducer = geom.r_0_h/cos(phi);
        theta = linspace(pi, pi/2+phi, 50);
        z_inducer = [z_c + r_inducer*cos(theta), 0]';
        r_inducer = [r_c + r_inducer*sin(theta), geom.r_1_h]'; 
    elseif strcmp(geom.inducer_type, "elliptic")
        % Elliptical arc
        phi = linspace(pi, pi/2, 50);
        z_inducer = [geom.L_inducer*cos(phi), 0]';
        r_inducer = [geom.r_1_h*sin(phi), geom.r_1_h]';
        
    end


end