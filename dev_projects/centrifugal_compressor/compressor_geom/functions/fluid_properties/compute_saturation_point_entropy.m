function properties = compute_saturation_point_entropy(s_in, fluid)
    
    % Check that the input entropy is within limits
    fluid.update(py.CoolProp.QT_INPUTS, 0.0, fluid.Ttriple)
    if s_in < fluid.smass
        error('Input entropy is lower than the liquid saturation entropy at the triple point: s_in=%0.2f, s_min=%0.2f', s_in, fluid.smass)
    end
    fluid.update(py.CoolProp.QT_INPUTS, 1.0, fluid.Ttriple)
    if s_in > fluid.smass
        error('Input entropy is higher than the vaport saturation entropy at the triple point: s_in=%0.2f, s_max=%0.2f', s_in, fluid.smass)
    end

    % Compute critical entropy
    fluid.update(py.CoolProp.PT_INPUTS, fluid.p_critical, fluid.T_critical)
    s_critical = fluid.smass;

    % Compute the spinodal point
    % get_entropy_residual(fluid.Ttriple)
    % get_entropy_residual(fluid.T_critical)
    [T, ~, exitflag] = fzero(@(T) get_entropy_residual(T), [fluid.Ttriple, fluid.T_critical]);
    if exitflag ~= 1
        error('Spinodal point calculation did not converge')
    end

    % Compute properties at the saturation point (inside 2-phase region)
    fluid.update(py.CoolProp.SmassT_INPUTS, s_in, T-1e-4)
    properties = fluid;
    % properties = compute_properties_metastable_Ts(T, s_in, fluid);

    function res = get_entropy_residual(T)
    
        % Compute saturated liquid state
        fluid.update(py.CoolProp.QT_INPUTS, 0.0, T)
        s_liq = fluid.smass;

        % Compute saturated vapor state
        fluid.update(py.CoolProp.QT_INPUTS, 1.0, T)
        s_vap = fluid.smass;

        % Evaluate residual depending on entropy level
        % Using quality is not a good approach because (s_vap - s_liq) in
        % the denominator approaches infinity close to the critical point
        res = (s_in - s_liq)*(s_in <= s_critical) + (s_in - s_vap)*(s_in > s_critical);
    
    end

end

