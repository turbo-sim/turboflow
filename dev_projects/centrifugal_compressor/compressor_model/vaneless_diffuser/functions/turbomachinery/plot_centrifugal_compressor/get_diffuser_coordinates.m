function coords = get_diffuser_coordinates(geom, nameValueArgs)
    
    % Compute the coordinates of the diffuser for plotting:
    % - Vaneless diffuser
    % - Airfoil diffuser
    % - Wedge diffuser
    arguments
        geom (1, 1) struct
        nameValueArgs.N_points (1, 1) double = 200;
    end
    
    if geom.has_vaned
    
        if strcmp(geom.vane_type, 'airfoil')
            coords = get_diffuser_airfoil_coordinates(geom, nameValueArgs.N_points);
        elseif strcmp(geom.vane_type, 'wedge')
            coords = get_diffuser_wedge_coordinates(geom, nameValueArgs.N_points);
        end
    
    else % vaneless

        coords = [];

    end

end


