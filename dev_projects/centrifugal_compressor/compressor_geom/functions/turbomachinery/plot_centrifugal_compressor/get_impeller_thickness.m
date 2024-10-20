function thickness = get_impeller_thickness(m_h, m_s, v, geom)
    
    % Scale meridional coordinate (required for splitter blades)
    m_h = (m_h - m_h(1)) / (m_h(end) - m_h(1)) * m_h(end);
    m_s = (m_s - m_s(1)) / (m_s(end) - m_s(1)) * m_s(end);

    % Compute thickness distribution
    thickness_h = thickness_clipped(m_h, geom.t_b_1);
    thickness_s = thickness_clipped(m_s, geom.t_b_1);
    thickness = (1-v)*thickness_h + v*thickness_s;

    % Define piecewise thickness distribution with a round nose at the
    % leading and trailing edges (circle function)
    function t = thickness_smooth(m, t)
        L = m(end);
        t = sqrt(eps + t^2 - (m - t).^2).*(m >= 0).*(m < t) + ...
                 t.*(m >= t).*(m <= L - t) + ...
                 sqrt(eps + t.^2 - (m - (L - t)).^2).*(m > L - t).*(m <= L);
    end

    % Define piecewise thickness distribution with a round nose at the
    % leading edge and a clipped trailing edge
    function t = thickness_clipped(m, t)
        t = sqrt(eps + t^2 - (m - t).^2).*(m >= 0).*(m < t) + ...
            t.*(m >= t);
    end

end