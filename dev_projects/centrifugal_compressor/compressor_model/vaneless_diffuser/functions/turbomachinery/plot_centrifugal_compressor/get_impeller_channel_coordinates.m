function [z, r] = get_impeller_channel_coordinates(u, v, geom)

    % Ruled surface between two elliptical segments
    z = (geom.L_z - v*geom.b_2).*cos(u);
    r = (geom.r_2 - (1-v)*geom.r_1_h - v*geom.r_1_s).*sin(u) + geom.r_2;

end