function props = compute_properties_metastable_ps(p_in, s_in, fluid, T_guess, rho_guess)

    arguments
        p_in (1, 1) double
        s_in (1, 1) double
        fluid
        T_guess (1, 1) = -1
        rho_guess (1, 1) = -1
    end

    % Compute thermodynamic properties using the Helmholtz energy equation
    % of state. This function uses pressure-entropy as independent
    % variables and solves a non-linear system to determine the correct
    % density for the fundamental temperature-density function call

    % Estimate density bounds if initial guess is not provided
    if rho_guess == -1
        rho_guess = 0.5*fluid.rhomass_critical;
    end
    if T_guess == -1
        T_guess = fluid.T_critical+20;
    end
    rhoT_guess = [rho_guess/fluid.rhomass_critical, T_guess/fluid.T_critical];

    % Solve residual equation for pressure and entropy
    options = optimoptions('fsolve', ...
                           'Algorithm','trust-region', ...
                           'MaxIterations', 2000, ...
                           'MaxFunctionEvaluations', 10000, ...
                           'FunctionTolerance', 1e-12, ...
                           'OptimalityTolerance', 1e-12, ...
                           'StepTolerance', 1e-12, ...
                           'FiniteDifferenceType','forward', ...
                           'Display', 'none');

    [x, ~, exitflag, output] = fsolve(@(rhoT_pair)get_state_residual(rhoT_pair, p_in, s_in, fluid), rhoT_guess, options);
    if exitflag <= 0
        fprintf('Exitflag: %s\n', exitflag)
        fprintf('%s\n', output.message)
        error('p-s function call did not converge')
    end

    % Compute thermodynamic state
    T = x(2)*fluid.T_critical;
    rho = x(1)*fluid.rhomass_critical;
    props = compute_properties_metastable_Td(T, rho, fluid);

   
end


function res = get_state_residual(rhoT_pair, p_in, s_in, fluid)

    % Update fluid state
    rho = rhoT_pair(1)*fluid.rhomass_critical;
    T = rhoT_pair(2)*fluid.T_critical;
    props = compute_properties_metastable_Td(T, rho, fluid);

    % Compute state residual
    p_res = (props.p - p_in)/p_in;
    s_res = (props.smass - s_in)/s_in;
    res = [p_res; s_res];

end
