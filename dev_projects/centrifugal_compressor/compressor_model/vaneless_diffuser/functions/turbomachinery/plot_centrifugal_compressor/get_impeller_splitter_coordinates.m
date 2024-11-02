function coords_splitter = get_impeller_splitter_coordinates(geom, coords)

    % Get the location where the splittler blade starts
    delta_blade = 2*pi/geom.Z_r_full;
    [~,index] = min(abs(coords.m_h - (1-geom.LR_split)*coords.m_h(end)));
    
    % Get the r-z-theta coordinates of the splitter blade
    z_h = coords.z_h(index:end);
    r_h = coords.r_h(index:end);
    z_s = coords.z_s(index:end);
    r_s = coords.r_s(index:end);
    m_h = coords.m_h(index:end);
    m_s = coords.m_s(index:end);
    m_prime_h = coords.m_prime_h(index:end);
    m_prime_s = coords.m_prime_s(index:end);
    theta_h = coords.theta_h(index:end) + delta_blade/2;
    theta_s = coords.theta_s(index:end) + delta_blade/2;

    % Compute camber surface hub
    x = (r_h.*cos(theta_h))';
    y = (r_h.*sin(theta_h))';
    z = z_h';
    xyz_c_h = [x(:), y(:), z(:)]';
    
    % Compute camber surface shroud
    x = (r_s.*cos(theta_s))';
    y = (r_s.*sin(theta_s))';
    z = z_s';
    xyz_c_s = [x(:), y(:), z(:)]';
    
    % Compute suction side hub
    t = get_impeller_thickness(m_h, m_s, 0, geom);
    x = r_h.*cos(theta_h + t/2./r_h);
    y = r_h.*sin(theta_h + t/2./r_h);
    z = z_h;
    xyz_ss_h = [x(:), y(:), z(:)]';
    theta_ss_h = theta_h + t/2./r_h;
    
    % Compute pressure side hub
    x = r_h.*cos(theta_h - t/2./r_h);
    y = r_h.*sin(theta_h - t/2./r_h);
    z = z_h;
    xyz_ps_h = [x(:), y(:), z(:)]';
    theta_ps_h = theta_h - t/2./r_h;
    
    % Compute suction side shroud
    t = get_impeller_thickness(m_h, m_s, 1, geom);
    x = r_s.*cos(theta_s + t/2./r_s);
    y = r_s.*sin(theta_s + t/2./r_s);
    z = z_s;
    xyz_ss_s = [x(:), y(:), z(:)]';
    theta_ss_s = theta_s + t/2./r_s;
    
    % Compute pressure side shroud
    x = r_s.*cos(theta_s - t/2./r_s);
    y = r_s.*sin(theta_s - t/2./r_s);
    z = z_s;
    xyz_ps_s = [x(:), y(:), z(:)]';
    theta_ps_s = theta_s - t/2./r_s;
    
    % Store coordinates in structure
    coords_splitter.m_h = m_h;
    coords_splitter.m_s = m_s;
    coords_splitter.m_prime_h = m_prime_h;
    coords_splitter.m_prime_s = m_prime_s;
    coords_splitter.theta_h = theta_h;
    coords_splitter.theta_s = theta_s;
    coords_splitter.theta_ss_h = theta_ss_h;
    coords_splitter.theta_ps_h = theta_ps_h;
    coords_splitter.theta_ss_s = theta_ss_s;
    coords_splitter.theta_ps_s = theta_ps_s;
    coords_splitter.z_h = z_h;
    coords_splitter.r_h = r_h;
    coords_splitter.z_s = z_s;
    coords_splitter.r_s = r_s;
    coords_splitter.xyz_c_h = xyz_c_h;
    coords_splitter.xyz_c_s = xyz_c_s;
    coords_splitter.xyz_ss_h = xyz_ss_h;
    coords_splitter.xyz_ps_h = xyz_ps_h;
    coords_splitter.xyz_ss_s = xyz_ss_s;
    coords_splitter.xyz_ps_s = xyz_ps_s;

end

