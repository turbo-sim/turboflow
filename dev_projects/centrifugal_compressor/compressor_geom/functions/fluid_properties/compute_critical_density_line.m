function critical_density_line = compute_critical_density_line(fluid, T)

    arguments
        fluid
        T (1, :) double = linspace(0.01+fluid.T_critical, 2.00*fluid.T_critical, 50)
    end


    % Compute the point of minimum temperature
    properties = compute_properties_metastable_Td(fluid.T_critical, fluid.rhomass_critical, fluid);
    names = fieldnames(properties);
    
    % Loop over all temperatures (re-use previous point as initial guess)
    for i = 1:numel(T)
        
        % Compute properties along the pseudocritical line
        properties = compute_properties_metastable_Td(T(i), fluid.rhomass_critical, fluid);

        % Store properties for export
        for j = 1:numel(names)
            critical_density_line.(names{j})(i) = properties.(names{j});
        end

    end


end
