function [m, m_prime, theta, geom] = get_impeller_wrap_angle(u, v, geom, phi_start, exponent)

    % Precoumpute arc-length
    L = get_impeller_channel_arclength(u, v, geom);

    % Integrate ODE to find arclength and wrapping angle
    [~, Y] = ode45(@(u, Y) ode_handle(u, Y, v, geom), u, [0, 0, phi_start]);

    % Export solution
    m = Y(:, 1)';
    m_prime = Y(:, 2)';
    theta = Y(:, 3)';
    
    % Define ODE right hand side
    function dYdu = ode_handle(uu, Y, v, geom)

        % Arc-length differential
        du = 1e-3;
        [x2, r2] = get_impeller_channel_coordinates(uu+du, v, geom);
        [x1, r1] = get_impeller_channel_coordinates(uu-du, v, geom);
        dmdu = sqrt(((x2-x1)/(2*du))^2 + ((r2-r1)/(2*du))^2);

        % Normalize meridional length
        dmdu_prime = dmdu/r1;

        % Wrap angle differential
        beta = metal_angle_distribution(Y(1)/L, v, geom, exponent);     
        dTheta_du = tand(beta)/r1*dmdu;

        % Export results
        dYdu = [dmdu; dmdu_prime; dTheta_du];

    end

    function beta = metal_angle_distribution(u, v, geom, exponent)
            
        % The u-parameter should be the arc-length fraction
        u = u^exponent;

        % Linear distribution between inlet and exit
        beta_h = (1 - u)*geom.beta_1b_h + u*geom.beta_2b;
        beta_s = (1 - u)*geom.beta_1b_s + u*geom.beta_2b;

        % Linear distribution between hub and shroud
        beta = (1-v)*beta_h + v*beta_s;
    
    end

end
