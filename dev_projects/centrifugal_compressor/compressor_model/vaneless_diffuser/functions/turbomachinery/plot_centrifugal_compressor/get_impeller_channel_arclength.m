function L = get_impeller_channel_arclength(u, v, geom)
    
    % Integrate ODE to find arclength
    [~, m] = ode45(@(u, Y) ode_handle(u, Y, v, geom), u, 0);
    L = m(end);
    
    % Define ODE right hand side
    function dmdu = ode_handle(u, ~, v, geom)
                
        % Arclength differential
        du = 1e-3;
        [x2, r2] = get_impeller_channel_coordinates(u+du, v, geom);
        [x1, r1] = get_impeller_channel_coordinates(u-du, v, geom);
        dmdu = sqrt(((x2-x1)/(2*du))^2 + ((r2-r1)/(2*du))^2);

    end
        
end
