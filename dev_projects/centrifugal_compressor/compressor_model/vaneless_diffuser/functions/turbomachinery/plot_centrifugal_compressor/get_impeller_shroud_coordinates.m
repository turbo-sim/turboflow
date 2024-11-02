function [z, r] = get_impeller_shroud_coordinates(geom, blade_fraction, N_points)

    arguments
        geom (1, 1) struct
        blade_fraction (1, 1) double = 1.00;
        N_points (1, 1) double = 100;
    end
    
    % Ruled surface between two elliptical segments
    u = linspace(3*pi/2+(1-blade_fraction)*pi/2, 2*pi, N_points)';
    [z, r] = get_impeller_channel_coordinates(u, 1.0, geom);

end