function [x_disk, y_disk] = get_impeller_disk_coordinates(geom, N_points)
    
    % Compute inducer coordinates
    [x_inducer, y_inducer] = get_inducer_coordinates(geom);

    % Compute hub coordinates
    [x_hub, y_hub] = get_impeller_hub_coordinates(geom, 1.00, N_points);

    % Compute backplate coordinates
    [x_backplate, y_backplate] = get_impeller_backplate_coordinates(geom, N_points);

    % Compute shaft coordinates
    [x_shaft, y_shaft] = get_impeller_shaft_coordinates(geom);

    % Merge coordinates to define the impeller disk
    x_disk = [x_inducer; x_hub; x_backplate; x_shaft];
    y_disk = [y_inducer; y_hub; y_backplate; y_shaft];

end