function [liquid_line, vapor_line] = compute_saturation_line(fluid, NameValueArgs)

    arguments
        fluid
        NameValueArgs.N_points (1, 1) = 100
    end

    % Define property names
    prop_names = ["T", "p", "rhomass", "smass", "hmass", "umass", "gibbsmass", "speed_sound", "cpmass", "cvmass", "viscosity", "conductivity", "isothermal_compressibility"];
%     prop_names = ["T", "p", "rhomass", "smass", "hmass", "umass", "speed_sound", "cpmass", "cvmass"];

    % Temperature array with refinement close to the critical point
%     ratio = 1 - 1.01*fluid.Ttriple/fluid.T_critical;
    ratio = 1 - fluid.Ttriple/fluid.T_critical;
    t1 = logspace(log10(1-0.9999), log10(ratio/10), ceil(NameValueArgs.N_points/2));
    t2 = logspace(log10(ratio/10), log10(ratio), floor(NameValueArgs.N_points/2));
    T_sat = (1-[t1 t2])*fluid.T_critical;

    % Loop over temperatures and property names in a efficient way
    for i = 1:numel(T_sat)

         % Compute liquid saturation line
        for j = 1:numel(prop_names)
            fluid.update(py.CoolProp.QT_INPUTS, 0, T_sat(i))
            liquid_line.(prop_names(j))(i) = fluid.(prop_names(j));
        end

        % Compute liquid saturation line
        for j = 1:numel(prop_names)
            fluid.update(py.CoolProp.QT_INPUTS, 1, T_sat(i))
            vapor_line.(prop_names(j))(i) = fluid.(prop_names(j));
        end

    end

    % Add critical point as part of the spinodal line
    props_crit = compute_properties_metastable_Td(fluid.T_critical, fluid.rhomass_critical, fluid);
    for j = 1:numel(prop_names)
        liquid_line.(prop_names{j}) = [props_crit.(prop_names{j}), liquid_line.(prop_names{j})];
        vapor_line.(prop_names{j}) = [props_crit.(prop_names{j}), vapor_line.(prop_names{j})];
    end
    
    % Re-format for easy concatenation
    for i = 1:numel(prop_names)
        liquid_line.(prop_names{i}) = flip(liquid_line.(prop_names{i}));
    end
    

end