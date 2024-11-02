function [spinodal_liq, spinodal_vap] = compute_spinodal_line(fluid, NameValueArgs)

    arguments
        fluid
        NameValueArgs.N_points (1, 1) = 100
        NameValueArgs.method (1, 1) string = 'standard'
    end

    % Temperature array with refinement close to the critical point
    ratio = 1 - 1.0*fluid.Ttriple/fluid.T_critical;
    t1 = logspace(log10(1-0.999), log10(ratio/10), ceil(NameValueArgs.N_points/2));
    t2 = logspace(log10(ratio/10), log10(ratio), floor(NameValueArgs.N_points/2));
    T = (1-[t1 t2])*fluid.T_critical;
    props_liq = compute_spinodal_point(T(1), fluid, "liquid", method=NameValueArgs.method);
    props_vap = compute_spinodal_point(T(1), fluid, "vapor", method=NameValueArgs.method);
    names = fieldnames(props_liq);
    for i = 1:numel(T)

        % Liquid spinodal line
        props_liq = compute_spinodal_point(T(i), fluid, props_liq.rhomass, method=NameValueArgs.method);

        % Vapor spinodal line
        props_vap = compute_spinodal_point(T(i), fluid, props_vap.rhomass, method=NameValueArgs.method);

        % Store properties for export
        for j = 1:numel(names)
            spinodal_liq.(names{j})(i) = props_liq.(names{j});
            spinodal_vap.(names{j})(i) = props_vap.(names{j});
        end

    end

    % Add critical point as part of the spinodal line
    props_crit = compute_properties_metastable_Td(fluid.T_critical, fluid.rhomass_critical, fluid);
    for j = 1:numel(names)
        spinodal_liq.(names{j}) = [props_crit.(names{j}), spinodal_liq.(names{j})];
        spinodal_vap.(names{j}) = [props_crit.(names{j}), spinodal_vap.(names{j})];
    end

    % Re-format for easy concatenation
    for j = 1:numel(names)
        spinodal_liq.(names{j}) = flip(spinodal_liq.(names{j}));
    end

end


% I learned that the best way to sweep the spinoda line is from the
% critical temperature to the triple temperature. Extending the spinodal
% line below the triple pressure will lead to strange wiggles and
% unphysical property values because the HEOS is outside of its range for
% which it was intended (and trained with data).
%
% When plotting the spinodal line from [T_crit, T_triple] in the p-s
% diagram the following features are observed:
%
%   1. The liquid line crosses the zero pressure limit and reaches very
%   high negative pressure values
%   2. The vapor line does not cross the zero pressure limit and reaches a
%   point of T=T_triple at a pressure higher than the triple pressure
%
% I am not sure if these metastable states are meaningful from the point of
% view of CFD, but it may be relevant to have a robust barotropic model
% that can include these states along the isentropes if required
% 