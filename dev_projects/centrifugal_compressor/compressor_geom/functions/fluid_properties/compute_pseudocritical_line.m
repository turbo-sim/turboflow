function pseudocritical_line = compute_pseudocritical_line(fluid, T)

    arguments
        fluid
        T (1, :) double = linspace(0.01+fluid.T_critical, 2.00*fluid.T_critical, 50)
    end


    % Compute the point of minimum temperature
    properties = compute_pseudocritical_point(fluid.T_critical+0.01, fluid, fluid.rhomass_critical);
    names = fieldnames(properties);
    
    % Loop over all temperatures (re-use previous point as initial guess)
    for i = 1:numel(T)
        
        % Compute properties along the pseudocritical line
        properties = compute_pseudocritical_point(T(i), fluid, properties.rhomass);

        % Store properties for export
        for j = 1:numel(names)
            pseudocritical_line.(names{j})(i) = properties.(names{j});
        end

    end


end
