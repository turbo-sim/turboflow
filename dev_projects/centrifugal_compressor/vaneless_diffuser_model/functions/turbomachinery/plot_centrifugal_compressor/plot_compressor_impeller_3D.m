%% Plotting functions
% The plotting should be done in 3D and then projected to 2D because the
% blades may overlap and the rendered must be able to handle overlapping of
% 3D objects to display the blades corrently

% Observations:
% The lean angle at the leading edge can be controlled with the initial
% condition for wrap angle
% 
% The rake angle at the trailing edge depends on the angle distributions
% and the shape of the meridional channel (difficult to control)
% 
% One trick to achieve zero lean and rake angles is to set the same wrap
% angle distribution at the hub and the shroud (with the shortcoming that
% the shroud metal angle is not controlled)

function plot_compressor_impeller_3D(geom, nameValueArgs)

    arguments
        geom (1, 1) struct
        nameValueArgs.N_points (1, 1) double = 200;
        nameValueArgs.color_blades (1, 3) double = 0.75*[1, 1, 1];
        nameValueArgs.color_hub (1, 3) double = 0.90*[1, 1, 1];
        nameValueArgs.edge_width (1, 1) double = 0.75;
    end


    % Plot inducer
    plot_disk_3D(geom, nameValueArgs.color_hub, nameValueArgs.edge_width);
    
    % Plot full blade coordinates
    coords = get_impeller_blade_coordinates(geom, N_points=nameValueArgs.N_points);
    plot_blade_surface_3D(geom, coords, nameValueArgs.color_blades, nameValueArgs.edge_width)
%     plot_hub_surface_3D(geom, coords, nameValueArgs.color_hub, nameValueArgs.edge_width)
%     plot_shroud_surface_3D(geom, coords, nameValueArgs.color_hub, nameValueArgs.edge_width)
%     plot_camber_surface_3D(geom, coords, nameValueArgs.color_blades, nameValueArgs.edge_width)

    % Plot splitter blade coordinates
    if geom.Z_r_split > 0
        coords_splitter = get_impeller_splitter_coordinates(geom, coords);
        plot_blade_surface_3D(geom, coords_splitter, nameValueArgs.color_blades, nameValueArgs.edge_width)
%         plot_camber_surface_3D(geom, coords_splitter, nameValueArgs.color_blades, nameValueArgs.edge_width)
    end

end


function plot_disk_3D(geom, color, edge_width)

    % Plot disk surface
    [z_disk, r_disk] = get_impeller_disk_coordinates(geom, 50);
    theta = linspace(0, 2*pi, 200);
    surf(z_disk + 0*r_disk*theta, r_disk*cos(theta), r_disk*sin(theta), 'FaceColor', color, 'EdgeColor','none')
    
    % Plot inducer edge
    [z, r] = get_inducer_coordinates(geom);
    plot3(z(end-1)+0*theta, r(end-1)*cos(theta), r(end-1)*sin(theta), Color='black', LineWidth=edge_width)
    
    % Plot backplate edges
    [z, r] = get_impeller_backplate_coordinates(geom);
    plot3(z(1)+0*theta, r(1)*cos(theta), r(1)*sin(theta), Color='black', LineWidth=edge_width)
    plot3(z(2)+0*theta, r(2)*cos(theta), r(2)*sin(theta), Color='black', LineWidth=edge_width)
    plot3(z(end-1)+0*theta, r(end-1)*cos(theta), r(end-1)*sin(theta), Color='black', LineWidth=edge_width)
    
    % Plot shaft edges
    [z, r] = get_impeller_shaft_coordinates(geom);
    plot3(z(1)+0*theta, r(1)*cos(theta), r(1)*sin(theta), Color='black', LineWidth=edge_width)
    plot3(z(2)+0*theta, r(2)*cos(theta), r(2)*sin(theta), Color='black', LineWidth=edge_width)

end

function plot_hub_surface_3D(geom, coordinates, color, edge_width)
    theta = linspace(0, 2*pi, 100);
    z = coordinates.z_h';
    r = coordinates.r_h';
    surf(z + 0*r*theta, r*cos(theta), r*sin(theta), 'FaceColor', color, 'EdgeColor','none')
    fill3(z(1) + 0*theta, r(1)*cos(theta), r(1)*sin(theta), color, 'linewidth', edge_width)
    fill3(z(end) + 0*theta, r(end)*cos(theta), r(end)*sin(theta), color, 'linewidth', edge_width)
end


function plot_shroud_surface_3D(geom, coordinates, color, edge_width)
    theta = linspace(0, 2*pi, 100);
    z = coordinates.z_s';
    r = coordinates.r_s';
    surf(z + 0*r*theta, r*cos(theta), r*sin(theta), 'FaceColor', color, 'EdgeColor','none')
end


function plot_camber_surface_3D(geom, coords, color, edge_width)
    
    % Process coordinates
    N = size(coords.xyz_c_h, 2);
    XYZ = [coords.xyz_c_h, coords.xyz_c_s];
    XYZ_h = coords.xyz_c_h;
    XYZ_s = coords.xyz_c_s;
    
    % Plot all blades with z-rotation matrix
    theta_0 = 0.00;
    theta_blade = 2*pi/geom.Z_r_full;
    for i = 1:geom.Z_r_full
 
        % Construct the 3D rotation matrix around the Z-axis
        theta = theta_0 + (i-1)*theta_blade;
        R_z = [cos(theta) -sin(theta) 0;
               sin(theta) +cos(theta) 0;
               0 0 1];
        
        % Rotate the coordinates
        xyz = R_z*XYZ;
        xyz_c_h = R_z*XYZ_h;
        xyz_c_s = R_z*XYZ_s;
        
        % Reshape arrays
        x = reshape(xyz(1,:), N, []);
        y = reshape(xyz(2,:), N, []);
        z = reshape(xyz(3,:), N, []);
    
        % Plot surface
        surf(z, x, y, 'FaceColor', color, 'EdgeColor', 'none')
        plot3(xyz_c_h(3,:), xyz_c_h(1,:), xyz_c_h(2,:), ...
                  Color='black', LineWidth=edge_width)

        plot3(xyz_c_s(3,:), xyz_c_s(1,:), xyz_c_s(2,:), ...
                  Color='black', LineWidth=edge_width)

        plot3([xyz_c_h(3,1), xyz_c_s(3,1)], ...
              [xyz_c_h(1,1), xyz_c_s(1,1)], ...
              [xyz_c_h(2,1), xyz_c_s(2,1)], ...
                  Color='black', LineWidth=edge_width)

        plot3([xyz_c_h(3,end), xyz_c_s(3,end)], ...
              [xyz_c_h(1,end), xyz_c_s(1,end)], ...
              [xyz_c_h(2,end), xyz_c_s(2,end)], ...
                  Color='black', LineWidth=edge_width)


    end

end


function plot_blade_surface_3D(geom, coords, color, edge_width)

    % Process coordinates
    XYZ_h = [coords.xyz_ps_h, flip(coords.xyz_ss_h, 2)];
    XYZ_s = [coords.xyz_ps_s, flip(coords.xyz_ss_s, 2)];
    XYZ = [XYZ_h, XYZ_s];
    N = size(XYZ_h, 2);

    % Get the index of the leading and trailing edges
    idx_leading = 1;
    idx_trailing = size(coords.xyz_ps_h, 2);
    
    % Plot all blades with z-rotation matrix
    theta_0 = 0.00;
    theta_blade = 2*pi/geom.Z_r_full;
    for i = 1:geom.Z_r_full
 
        % Construct the 3D rotation matrix around the Z-axis
        theta = theta_0 + (i-1)*theta_blade;
        R_z = [cos(theta) -sin(theta) 0;
               sin(theta) +cos(theta) 0;
               0 0 1];
        
        % Rotate the coordinates
        xyz = R_z*XYZ;
        xyz_h = R_z*XYZ_h;
        xyz_s = R_z*XYZ_s;
        
        % Reshape arrays
        x = reshape(xyz(1,:), N, []);
        y = reshape(xyz(2,:), N, []);
        z = reshape(xyz(3,:), N, []);
    
        % Plot surface
        surf(z, x, y, 'FaceColor',  color, 'EdgeColor','none')
        fill3(xyz_h(3,:), xyz_h(1,:), xyz_h(2,:), color, LineWidth=edge_width)
        fill3(xyz_s(3,:), xyz_s(1,:), xyz_s(2,:), color, LineWidth=edge_width)

        plot3([xyz_h(3,idx_leading) xyz_s(3,idx_leading)], ...
              [xyz_h(1,idx_leading) xyz_s(1,idx_leading)], ...
              [xyz_h(2,idx_leading) xyz_s(2,idx_leading)], ...
              Color='black', LineWidth=edge_width)

        plot3([xyz_h(3,idx_trailing) xyz_s(3,idx_trailing)], ...
              [xyz_h(1,idx_trailing) xyz_s(1,idx_trailing)], ...
              [xyz_h(2,idx_trailing) xyz_s(2,idx_trailing)], ...
              Color='black', LineWidth=edge_width)

        plot3([xyz_h(3,idx_trailing+1) xyz_s(3,idx_trailing+1)], ...
              [xyz_h(1,idx_trailing+1) xyz_s(1,idx_trailing+1)], ...
              [xyz_h(2,idx_trailing+1) xyz_s(2,idx_trailing+1)], ...
              Color='black', LineWidth=edge_width)
    
    end
    
end
