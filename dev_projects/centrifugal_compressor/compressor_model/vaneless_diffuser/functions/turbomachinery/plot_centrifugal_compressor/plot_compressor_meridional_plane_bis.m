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
    has_vanes = geom.has_vanes;
    r_1_h = 0.5*geom.D_1_h;
    r_1_s = 0.5*geom.D_1_s;
    r_2 = 0.5*geom.D_2;
    r_3 = 0.5*geom.D_3;
    r_4 = 0.5*geom.D_4;
    r_5 = 0.5*geom.D_5;
    r_6 = 0.5*geom.D_6;
    b_1 = geom.b_1;
    b_2 = geom.b_2;
    b_2 = geom.b_2;
    b_3 = geom.b_3;
    b_5 = geom.b_5;
    b_6 = geom.b_6;
    L_z = geom.L_z;
    eps_a = geom.eps_a;
    eps_r = geom.eps_r;
    eps_b = geom.eps_b;
    LR_split = geom.LR_split;
   
    % Define additional geometric parameters for plotting 
    z0 = 0;
    r_shaft = r_1_h;
    t_casing = 0.5*b_2;
    t_backplate = 0.40*b_2;
    L_inducer = geom.L_inducer;
    L_shaft = L_z/2;
    inducer_type = geom.inducer_type;
    
    % Impeller
    theta = linspace(3/2*pi, 2*pi, 50)';
    x_h_impeller = L_z*cos(theta);
    y_h_impeller = r_2 + (r_2 - r_1_h)*sin(theta);
    x_s_impeller = (L_z - b_2)*cos(theta);
    y_s_impeller = r_2 + (r_2 - r_1_s)*sin(theta);
    xy_impeller = [x_h_impeller, y_h_impeller;
                   flip(x_s_impeller), flip(y_s_impeller)];
    
    % Splitter blades
    theta = linspace(3/2*pi+(1-LR_split)*pi/2, 2*pi, 50)';
    x_h_splitter = L_z*cos(theta);
    y_h_splitter = r_2 + (r_2 - r_1_h)*sin(theta);
    x_s_splitter = (L_z - b_2)*cos(theta);
    y_s_splitter = r_2 + (r_2 - r_1_s)*sin(theta);
    xy_splitter = [x_h_splitter, y_h_splitter;
                   flip(x_s_splitter), flip(y_s_splitter)];
    
    % Diffuser
    x_h_diffuser = [L_z, L_z]';
    y_h_diffuser = [r_3, r_5]';
    x_s_diffuser = [L_z - b_3, L_z - b_5]';
    y_s_diffuser = [r_3, r_5]';
    xy_diffuser = [x_h_diffuser, y_h_diffuser
                   flip(x_s_diffuser), flip(y_s_diffuser)];
    
    % Hub and shaft
    if strcmp(inducer_type, "flat")
        x_nose = z0-[L_inducer, L_inducer]';
        y_nose = [0, r_1_h]';
    elseif strcmp(inducer_type, "round")
        phi = linspace(pi, 1/2*pi, 50)';
        x_nose = z0 + L_inducer*cos(phi);
        y_nose = r_1_h*sin(phi);
    end
    xy_shaft = [x_nose, y_nose
                x_h_impeller, y_h_impeller
                L_z+t_backplate, r_2
                L_z+t_backplate, r_shaft
                L_z+t_backplate+L_shaft, r_shaft
                L_z+t_backplate+L_shaft, 0];
    
    % Upper casing
    theta = linspace(3/2*pi, 2*pi, 50)';
    x_c1 = (L_z - b_2 - eps_a)*cos(theta);
    y_c1 = r_2 + (r_2 - r_1_s - eps_r)*sin(theta);
    x_c2 = (L_z - b_2 - eps_a - t_casing)*cos(theta);
    y_c2 = r_2 + (r_2 - r_1_s - eps_r - t_casing)*sin(theta);
    xy_casing = [z0-L_inducer, y_c1(1)
                 x_c1, y_c1
                 L_z-b_3, r_3
                 L_z-b_5, r_5
                 L_z-b_5, r_6
                 L_z-b_5-t_casing, r_6
                 L_z-b_5-t_casing, r_5
                 L_z-b_3-t_casing, r_3
                 flip(x_c2), flip(y_c2)
                 z0-L_inducer, y_c2(1)];

    % Back casing
    xy_back = [L_z+t_backplate+eps_b, r_shaft+eps_b
               L_z+t_backplate+eps_b, r_2+eps_r
               L_z, r_2+eps_b
               L_z, y_h_diffuser(1)
               x_h_diffuser, y_h_diffuser
               x_h_diffuser(end), r_6
               L_z+t_backplate+eps_b+t_casing, r_6
               L_z+t_backplate+eps_b+t_casing, r_shaft+eps_b];
    
    % Plot the compressor
    side = +1;
    linewidth = nameValueArgs.edge_width;
    patch(nameValueArgs.axial_offset + xy_impeller(:,1), side*xy_impeller(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_splitter(:,1), side*xy_splitter(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_shaft(:,1), side*xy_shaft(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_casing(:,1), side*xy_casing(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
    patch(nameValueArgs.axial_offset + xy_back(:,1), side*xy_back(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
    if has_vanes
        patch(nameValueArgs.axial_offset + xy_diffuser(:,1), side*xy_diffuser(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
    end
    
    if nameValueArgs.symmetric
        side = -1;
        patch(nameValueArgs.axial_offset + xy_impeller(:,1), side*xy_impeller(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_splitter(:,1), side*xy_splitter(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_shaft(:,1), side*xy_shaft(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_casing(:,1), side*xy_casing(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
        patch(nameValueArgs.axial_offset + xy_back(:,1), side*xy_back(:,2), nameValueArgs.color_hub, 'edgecolor', 'k', 'linewidth', linewidth)
        if has_vanes
            patch(nameValueArgs.axial_offset + xy_diffuser(:,1), side*xy_diffuser(:,2), nameValueArgs.color_blades, 'edgecolor', 'k', 'linewidth', linewidth)
        end
    end
    
    % Avoid black line showing up at the axis
    plot(nameValueArgs.axial_offset + [x_nose(1), L_z+t_backplate+L_shaft], [0, 0], 'Color', nameValueArgs.color_hub)

    % Adjust axes
    z_min = nameValueArgs.axial_offset-0.75*geom.L_z;
    z_max = nameValueArgs.axial_offset+2.0*geom.L_z;
    a = 1.2*geom.r_6;
    if nameValueArgs.symmetric
        axis([z_min, z_max, -a, a])
    else
        axis([z_min, z_max, 0, a])
    end

%     % Symmetry plane
%     dx = L_z/2;
%     plot(x_offset + [0-dx, L_z+dx], [0, 0], 'k--')


end

