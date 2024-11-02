function plot_compressor_blade_to_blade(geom, nameValueArgs)

    arguments
        geom (1, 1) struct
        nameValueArgs.color_blades (1, 3) double = 0.75*[1, 1, 1];
        nameValueArgs.edge_width (1, 1) double = 0.75;
        nameValueArgs.section (1, 1) string = "shroud";
    end

    if strcmp(nameValueArgs.section, "shroud")
        
        % Plot full impeller blades
        d_theta = 2*pi/geom.Z_r_full;
        coords = get_impeller_blade_coordinates(geom);
        m_prime = [coords.m_prime_s flip(coords.m_prime_s)];
        theta = [coords.theta_ps_s flip(coords.theta_ss_s)];
        for i = 1:2*geom.Z_r_full
            patch(m_prime, theta+(i-1)*d_theta, nameValueArgs.color_blades, LineWidth=nameValueArgs.edge_width)
        end
    
        % Plot splitter impeller blades
        if geom.Z_r_split > 0
            coords_splitter = get_impeller_splitter_coordinates(geom, coords);
            m_prime = [coords_splitter.m_prime_s flip(coords_splitter.m_prime_s)];
            theta = [coords_splitter.theta_ps_s flip(coords_splitter.theta_ss_s)];
            for i = 1:2*geom.Z_r_full
                patch(m_prime, theta+(i-1)*d_theta, nameValueArgs.color_blades, LineWidth=nameValueArgs.edge_width)
            end
        end
    
        % Plot diffuser blades
        % Apply analytic formula for conformal transformation with dm=dr
        if geom.has_vaned
            d_theta = 2*pi/geom.Z_vd;
            coords_diffuser = get_diffuser_coordinates(geom);
            xyz = coords_diffuser.xyz_h;
            x = xyz(1,:);
            y = xyz(2,:);
            radius = sqrt(x.^2 + y.^2);
            m_prime = coords.m_prime_s(end) + log(radius/geom.r_2);
            theta = atan2(y, x);
            for i = 1:2*geom.Z_vd
                patch(m_prime, theta+(i-1-5)*d_theta, nameValueArgs.color_blades, LineWidth=nameValueArgs.edge_width)
            end
        end
        
        % Adjust m' axis limits
        m_start = coords.m_prime_s(1)-geom.L_z/geom.r_1_s/10;
        m_end = coords.m_prime_s(end) + log(geom.r_6/geom.r_2);
        dm = (m_end - m_start)/3;
        xlim([m_start-dm, m_end+dm])
        plot(m_start*[1, 1], 2*pi*[-1, 1], 'k--', Linewidth=0.25);
        plot(m_end*[1, 1], 2*pi*[-1, 1], 'k--', Linewidth=0.25);
    
        % Adjust theta axis limits
        ax = gca;
        ax.YTick = 0:pi/4:2*pi;
        angle_degree = ax.YTick*180/pi;
        labels = string(angle_degree);
        for i = 1:numel(labels)
            labels(i) = strcat(labels(i), "$^\circ$");
        end
        ax.YTickLabel = labels;


    elseif strcmp(nameValueArgs.section, "hub")

        % Plot full impeller blades
        d_theta = 2*pi/geom.Z_r_full;
        coords = get_impeller_blade_coordinates(geom);
        m_prime = [coords.m_prime_h flip(coords.m_prime_h)];
        theta = [coords.theta_ps_h flip(coords.theta_ss_h)];
        for i = 1:2*geom.Z_r_full
            patch(m_prime, theta+(i-1)*d_theta, nameValueArgs.color_blades, LineWidth=nameValueArgs.edge_width)
        end
    
        % Plot splitter impeller blades
        if geom.Z_r_split > 0
            coords_splitter = get_impeller_splitter_coordinates(geom, coords);
            m_prime = [coords_splitter.m_prime_h flip(coords_splitter.m_prime_h)];
            theta = [coords_splitter.theta_ps_h flip(coords_splitter.theta_ss_h)];
            for i = 1:2*geom.Z_r_full
                patch(m_prime, theta+(i-1)*d_theta, nameValueArgs.color_blades, LineWidth=nameValueArgs.edge_width)
            end
        end
    
        % Plot diffuser blades
        % Apply analytic formula for conformal transformation with dm=dr
        if geom.has_vaned
            d_theta = 2*pi/geom.Z_vd;
            coords_diffuser = get_diffuser_coordinates(geom);
            xyz = coords_diffuser.xyz_h;
            x = xyz(1,:);
            y = xyz(2,:);
            radius = sqrt(x.^2 + y.^2);
            m_prime = coords.m_prime_h(end) + log(radius/geom.r_2);
            theta = atan2(y, x);
            for i = 1:2*geom.Z_vd
                patch(m_prime, theta+(i-1-5)*d_theta, nameValueArgs.color_blades, LineWidth=nameValueArgs.edge_width)
            end
        end
        
        % Adjust m' axis limits
        m_start = coords.m_prime_h(1)-geom.L_z/geom.r_1_h/10;
        m_end = coords.m_prime_h(end) + log(geom.r_6/geom.r_2);
        dm = (m_end - m_start)/3;
        xlim([m_start-dm, m_end+dm])
        plot(m_start*[1, 1], 2*pi*[-1, 1], 'k--', Linewidth=0.25);
        plot(m_end*[1, 1], 2*pi*[-1, 1], 'k--', Linewidth=0.25);
    
        % Adjust theta axis limits
        ax = gca;
        ax.YTick = 0:pi/4:2*pi;
        angle_degree = ax.YTick*180/pi;
        labels = string(angle_degree);
        for i = 1:numel(labels)
            labels(i) = strcat(labels(i), "$^\circ$");
        end
        ax.YTickLabel = labels;


    else
        error("Section value must be 'hub' or 'shroud'")

    end



end


