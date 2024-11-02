function plot_compressor_diffuser_3D(geom, nameValueArgs)

    arguments
        geom (1, 1) struct
        nameValueArgs.color_blades (1, 3) double = 0.75*[1, 1, 1];
        nameValueArgs.color_hub (1, 3) double = 0.90*[1, 1, 1];
        nameValueArgs.plot_diffuser_channels (1, 1) logical = false
        nameValueArgs.plot_downstream_shroud (1, 1) logical = false
        nameValueArgs.edge_width (1, 1) double = 0.75;
        nameValueArgs.N_points_vanes (1, 1) double = 200;
        nameValueArgs.N_points_vaneless (1, 1) double = 200;
    end
    
    % Plot diffuser vanes
    if geom.has_vaned
        plot_vaned_diffuser(geom, ...
                            nameValueArgs.N_points_vanes, ...
                            nameValueArgs.color_blades, ...
                            nameValueArgs.edge_width, ...
                            nameValueArgs.plot_diffuser_channels, ...
                            nameValueArgs.plot_downstream_shroud)
    end

    % Plot vaneless diffuser hub
    plot_vaneless_diffuser_hub(geom, ...
                               nameValueArgs.N_points_vaneless, ...
                               nameValueArgs.color_hub, ...
                               nameValueArgs.edge_width)

end


function plot_vaneless_diffuser_hub(geom, N_points, color_hub, edge_width)

    % Plot the diffuser hub surface
    phi_arc = linspace(0, 2*pi, N_points);
    z = [geom.L_z, geom.L_z]';
    r = [geom.r_2, geom.r_6]';
    surf(z + 0*r*phi_arc, r*cos(phi_arc), r*sin(phi_arc), 'FaceColor', color_hub, 'EdgeColor','none')
    plot3(z(1) + 0*phi_arc, r(1)*cos(phi_arc), r(1)*sin(phi_arc), 'Color', 'k', 'linewidth', edge_width)
    plot3(z(2) + 0*phi_arc, r(2)*cos(phi_arc), r(2)*sin(phi_arc), 'Color', 'k', 'linewidth', edge_width)

end


function plot_vaned_diffuser(geom, N_points, color_vanes, edge_width, plot_diffuser_channels, plot_downstream_shroud)
    
    % Compute coordinates
    coords = get_diffuser_coordinates(geom, N_points=N_points);
    XYZ = coords.xyz;
    XYZ_h = coords.xyz_h;
    XYZ_s = coords.xyz_s;
    XYZ_channel = coords.xyz_channel;
    XYZ_inlet = coords.xyz_inlet;
    XYZ_outlet = coords.xyz_outlet;
    N = size(XYZ_h, 2);

    % Plot all blades with z-rotation matrix
    theta_0 = 0;
    d_theta = 2*pi/geom.Z_vd;
    for i = 1:geom.Z_vd
    
        % Construct the 3D rotation matrix around the Z-axis
        theta = theta_0 + (i-1)*d_theta;
        R_z = [cos(theta) -sin(theta) 0;
               sin(theta) +cos(theta) 0;
               0 0 1];
        
        % Rotate the coordinates
        xyz = R_z*XYZ;
        xyz_h = R_z*XYZ_h;
        xyz_s = R_z*XYZ_s;
        
        % Reshape array
        x = reshape(xyz(1,:), N, []);
        y = reshape(xyz(2,:), N, []);
        z = reshape(xyz(3,:), N, []);
    
        % Plot blade side surface
        surf(z, x, y, 'FaceColor', color_vanes, 'EdgeColor', 'none')
    
        % Hub and shroud faces
        fill3(xyz_h(3,:), xyz_h(1,:), xyz_h(2,:), color_vanes, 'edgecolor', 'k', 'linewidth', edge_width)
        fill3(xyz_s(3,:), xyz_s(1,:), xyz_s(2,:), color_vanes, 'edgecolor', 'k', 'linewidth', edge_width)
        
        % Draw blade edges
        if strcmp(geom.vane_type, 'airfoil')
            
            % Leading edge
            idx_leading = 1;
            plot3([xyz_h(3,idx_leading) xyz_s(3,idx_leading)], ...
                  [xyz_h(1,idx_leading) xyz_s(1,idx_leading)], ...
                  [xyz_h(2,idx_leading) xyz_s(2,idx_leading)], ...
                  Color='black', LineWidth=edge_width)
            
            % Trailing edge
            idx_trailing = ceil(1/2*N_points);
            plot3([xyz_h(3,idx_trailing) xyz_s(3,idx_trailing)], ...
                  [xyz_h(1,idx_trailing) xyz_s(1,idx_trailing)], ...
                  [xyz_h(2,idx_trailing) xyz_s(2,idx_trailing)], ...
                  Color='black', LineWidth=edge_width)

        elseif strcmp(geom.vane_type, 'wedge')
            
            % Leading edge
            idx_leading = 1;
            plot3([xyz_h(3,idx_leading) xyz_s(3,idx_leading)], ...
                  [xyz_h(1,idx_leading) xyz_s(1,idx_leading)], ...
                  [xyz_h(2,idx_leading) xyz_s(2,idx_leading)], ...
                  Color='black', LineWidth=edge_width)


            % Trailing edge (side 1)
            idx_trailing_1 = ceil(1/3*N_points);
            plot3([xyz_h(3,idx_trailing_1) xyz_s(3,idx_trailing_1)], ...
                  [xyz_h(1,idx_trailing_1) xyz_s(1,idx_trailing_1)], ...
                  [xyz_h(2,idx_trailing_1) xyz_s(2,idx_trailing_1)], ...
                  Color='black', LineWidth=edge_width)

            % Trailing edge (side 2)
            idx_trailing_2 = ceil(2/3*N_points);
            plot3([xyz_h(3,idx_trailing_2) xyz_s(3,idx_trailing_2)], ...
                  [xyz_h(1,idx_trailing_2) xyz_s(1,idx_trailing_2)], ...
                  [xyz_h(2,idx_trailing_2) xyz_s(2,idx_trailing_2)], ...
                  Color='black', LineWidth=edge_width)

        end

        if plot_diffuser_channels% && strcmp(geom.vane_type, 'wedge')

            % Rotate the coordinates
            xyz_channel = R_z*XYZ_channel;
            xyz_inlet = R_z*XYZ_inlet;
            xyz_outlet = R_z*XYZ_outlet;
    
            % Plot the planes
            fill3(xyz_channel(3,:), xyz_channel(1,:), xyz_channel(2,:), 'blue', EdgeColor='black', LineWidth=edge_width, FaceAlpha=0.3)
            fill3(xyz_inlet(3,:), xyz_inlet(1,:), xyz_inlet(2,:), 'blue', EdgeColor='black', LineWidth=edge_width, FaceAlpha=0.3)
            fill3(xyz_outlet(3,:), xyz_outlet(1,:), xyz_outlet(2,:), 'blue', EdgeColor='black', LineWidth=edge_width, FaceAlpha=0.3)

        end
    
    end
    
    % Plot the shroud surface downstream the diffuser
    if plot_downstream_shroud
        phi_arc = linspace(0, 2*pi, 100);
        z = [geom.L_z-geom.b_5, geom.L_z-geom.b_5]';
        r = [geom.r_5, geom.r_6]';
        surf(z + 0*r*phi_arc, r*cos(phi_arc), r*sin(phi_arc), FaceColor=color_vanes, EdgeColor='none')
        plot3(z(1) + 0*phi_arc, r(1)*cos(phi_arc), r(1)*sin(phi_arc), Color='black', LineWidth=edge_width)
        plot3(z(2) + 0*phi_arc, r(2)*cos(phi_arc), r(2)*sin(phi_arc), Color='black', LineWidth=edge_width)
    end

end

