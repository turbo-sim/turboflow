function plot_compressor_meridional_plane(geom, nameValueArgs)

    arguments
        geom (1, 1) struct
        nameValueArgs.axial_offset (1, 1) double = 0.00;
        nameValueArgs.edge_width (1, 1) double = 0.50;
        nameValueArgs.color_blades (1, 3) double = 0.75*[1, 1, 1];
        nameValueArgs.color_hub (1, 3) double = 0.90*[1, 1, 1];
        nameValueArgs.symmetric (1, 1) logical = false
    end

    % Load geometry
    has_vanes = geom.has_vaned;
    r_0_h = 0.5*geom.D_0_h;
    r_0_s = 0.5*geom.D_0_s;
    r_1_h = 0.5*geom.D_1_h;
    r_1_s = 0.5*geom.D_1_s;
    r_2 = 0.5*geom.D_2;
    r_3 = 0.5*geom.D_3;
    r_4 = 0.5*geom.D_4;
    r_5 = 0.5*geom.D_5;
    r_6 = 0.5*geom.D_6;
    b_1 = geom.b_1;
    b_2 = geom.b_2;
    b_3 = geom.b_3;
    b_5 = geom.b_5;
    b_6 = geom.b_6;
    L_z = geom.L_z;
    eps_a = geom.eps_a;
    eps_r = geom.eps_r;
    eps_b = geom.eps_b;
    LR_split = geom.LR_split;
    L_inducer = geom.L_inducer;
    L_shaft = geom.L_shaft; 

    % Define additional geometric parameters for plotting
    t_casing = 0.5*b_2;

    % Impeller
    [x_h_impeller, y_h_impeller] = get_impeller_hub_coordinates(geom, 1.0, 100);
    [x_s_impeller, y_s_impeller] = get_impeller_shroud_coordinates(geom, 1.0, 100);
    xy_impeller = [x_h_impeller, y_h_impeller;
                   flip(x_s_impeller), flip(y_s_impeller)];
    
    % Splitter blades
    [x_h_splitter, y_h_splitter] = get_impeller_hub_coordinates(geom, LR_split, 100);
    [x_s_splitter, y_s_splitter] = get_impeller_shroud_coordinates(geom, LR_split, 100);
    xy_splitter = [x_h_splitter, y_h_splitter;
                   flip(x_s_splitter), flip(y_s_splitter)];

    % Diffuser
    x_h_diffuser = [L_z, L_z]';
    y_h_diffuser = [r_3, r_5]';
    x_s_diffuser = [L_z - b_3, L_z - b_5]';
    y_s_diffuser = [r_3, r_5]';
    xy_diffuser = [x_h_diffuser, y_h_diffuser
                   flip(x_s_diffuser), flip(y_s_diffuser)];
    
    % Impeller disk
    [x_disk, y_disk] = get_impeller_disk_coordinates(geom, 100);
    xy_disk = [x_disk, y_disk];

    % Upper casing
    theta = linspace(3/2*pi, 2*pi, 50)';
    x_c1 = (L_z - b_2 - eps_a)*cos(theta);
    y_c1 = r_2 + (r_2 - r_1_s - eps_r)*sin(theta);
    x_c2 = (L_z - b_2 - eps_a - t_casing)*cos(theta);
    y_c2 = r_2 + (r_2 - r_1_s - eps_r - t_casing)*sin(theta);
    xy_casing = [-L_inducer, r_0_s+eps_r
                 x_c1, y_c1
                 L_z-b_3, r_3
                 L_z-b_5, r_5
                 L_z-b_5, r_6
                 L_z-b_5-t_casing, r_6
                 L_z-b_5-t_casing, r_5
                 L_z-b_3-t_casing, r_3
                 flip(x_c2), flip(y_c2)
                 -L_inducer, r_0_s+eps_r+t_casing];

    % Back casing
    [x_backplate, y_backplate] = get_impeller_backplate_coordinates(geom, 100);
    xy_back = [flip(x_backplate)+eps_b, flip(y_backplate)+eps_b
               L_z, r_2+eps_r
               L_z, y_h_diffuser(1)
               x_h_diffuser, y_h_diffuser
               x_h_diffuser(end), r_6
               L_z+geom.t_backplate_1+eps_b+t_casing, r_6
               x_backplate(2:end)+eps_b+t_casing, y_backplate(2:end)+eps_b];
%                L_z+geom.t_backplate_2+eps_b+t_casing, r_6
%                L_z+geom.t_backplate_2+eps_b+t_casing, geom.r_shaft+eps_b];
        
    % Plot the compressor
    side = +1;
    linewidth = nameValueArgs.edge_width;
    patch(nameValueArgs.axial_offset + xy_impeller(:,1), side*xy_impeller(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_splitter(:,1), side*xy_splitter(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_disk(:,1), side*xy_disk(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_casing(:,1), side*xy_casing(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_back(:,1), side*xy_back(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
    if has_vanes
        patch(nameValueArgs.axial_offset + xy_diffuser(:,1), side*xy_diffuser(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
    end
    
    if nameValueArgs.symmetric
        side = -1;
        patch(nameValueArgs.axial_offset + xy_impeller(:,1), side*xy_impeller(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_splitter(:,1), side*xy_splitter(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_disk(:,1), side*xy_disk(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_casing(:,1), side*xy_casing(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_back(:,1), side*xy_back(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
        if has_vanes
            patch(nameValueArgs.axial_offset + xy_diffuser(:,1), side*xy_diffuser(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
        end
    end
    
    % Avoid black line showing up at the axis
    plot(nameValueArgs.axial_offset + [-L_inducer, L_z+L_shaft], [0, 0], 'Color', nameValueArgs.color_hub)

    % Adjust axes
    z_min = nameValueArgs.axial_offset-max(2.0*geom.L_inducer, 0.75*geom.L_z);
    z_max = nameValueArgs.axial_offset+2.0*geom.L_z;
    a = 1.2*geom.r_6;
    if nameValueArgs.symmetric
        axis([z_min, z_max, -a, a])
    else
        axis([z_min, z_max, 0, a])
    end


end

