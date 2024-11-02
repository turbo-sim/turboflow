function props = compute_properties_metastable_Ts(T_in, s_in, fluid, rho_guess)

    arguments
        T_in (1, 1) double
        s_in (1, 1) double
        fluid
        rho_guess (1, 1) = -1
    end

    % Compute thermodynamic properties using the Helmholtz energy equation
    % of state. This function uses temperature-entropy as independent
    % variables and solves a non-linear equation to determine the correct
    % density for the fundamental temperature-density function call

    % Estimate density bounds if initial guess is not provided
    if rho_guess == -1
        T_triple = fluid.Ttriple;
        fluid.update(py.CoolProp.QT_INPUTS, 1, T_triple); rho_min = 0.5*fluid.rhomass;
        fluid.update(py.CoolProp.QT_INPUTS, 0, T_triple); rho_max = 1.5*fluid.rhomass;
        rho_guess = [rho_min, rho_max];
    end

    % Solve residual equation for density   
    [rhomass, ~, exitflag, ~]  = fzero(@(d)get_state_residual(d, T_in, s_in, fluid), rho_guess);
    if exitflag ~= 1
        error('T-s function call did not converge')
    end

    % Compute thermodynamic state using 
    props = compute_properties_metastable_Td(T_in, rhomass, fluid);
   
end


function res = get_state_residual(d, T_in, s_in, fluid)

    % Update fluid state
    props = compute_properties_metastable_Td(T_in, d, fluid);

    % Compute state residual
    res = (props.smass - s_in)/s_in;

end
