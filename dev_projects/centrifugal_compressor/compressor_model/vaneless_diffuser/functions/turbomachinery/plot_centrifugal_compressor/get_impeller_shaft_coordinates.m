function [x_shaft, y_shaft] = get_impeller_shaft_coordinates(geom)

    % Define the shaft x-y (r-z) coordinates
    x_shaft = [geom.L_z + geom.t_backplate_2
               geom.L_z + geom.t_backplate_2 + geom.L_shaft
               geom.L_z + geom.t_backplate_2 + geom.L_shaft];

    y_shaft = [geom.r_shaft
               geom.r_shaft
               0.00];

end
