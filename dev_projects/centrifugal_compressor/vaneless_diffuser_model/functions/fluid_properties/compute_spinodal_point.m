function properties = compute_spinodal_point(T, fluid, rho_guess, NameValueArgs)
    
    arguments
        T (1, 1) double
        fluid
        rho_guess
        NameValueArgs.method (1, 1) string = 'standard'
        NameValueArgs.N_trial (1, 1) double = 100;
    end

    % Check that the inlet temperature is lower than the critical value
    if T >= fluid.T_critical
        error("T_in=%0.3fK must be less than T_critical=%0.3fK", T, fluid.T_critical)
    end

    % Define the initial guess for density
    if strcmp(rho_guess, 'liquid')

        % Liquid branch of the spinodal line
        rho_guess = get_spinodal_liquid_density_guess(T, fluid, NameValueArgs.N_trial);

    elseif strcmp(rho_guess, 'vapor')
       
        % Vapor branch of the spinodal line
        rho_guess = get_spinodal_vapor_density_guess(T, fluid, NameValueArgs.N_trial);

    elseif ~isnumeric(rho_guess) || ~isscalar(rho_guess)

        % Invalid input value for the density initial guess
        error(['The initial guess for density was: %s\n' ...
               'The initial guess for density must be a "liquid", "vapor" or a numerical value'], num2str(rho_guess))

    end


    % Compute the spinodal point with the selected method
    if strcmp(NameValueArgs.method, 'standard')

        % Standard method based on finding the zero of the residual
        rho = fzero(@(rho) get_residual(T, rho, fluid), rho_guess);

    elseif strcmp(NameValueArgs.method, 'robust')

        % Robust method based on minimizing the absolute value of residual
        % This method can handle ill-posed equations of state that do not
        % have a well-defined spinodal point (like nitrogen)
        options = optimoptions('fminunc', ...
                               'MaxFunctionEvaluations', 2000, ...
                               'MaxIterations', 500, ...
                               'OptimalityTolerance', 1e-16, ...
                               'StepTolerance', 1e-16, ...
                               'Display', 'none');
        [rho, ~, exitflag, output] = fminunc(@(rho)abs(get_residual(T, rho, fluid)), rho_guess, options);
        if exitflag <= 0
            fprintf('Exitflag: %s\n', exitflag)
            fprintf('%s\n', output.message)
            error('Spinodal point calculation did not converge')
        end
    
    else
        error('Computation method must be "standard" or "robust"')
    end

    % Evaluate thermodynamic properties at the spinodal point
    properties = compute_properties_metastable_Td(T, rho, fluid);


end


function rho_guess = get_spinodal_liquid_density_guess(T, fluid, N)

    % Compute saturated liquid density
    fluid.update(py.CoolProp.CoolProp.QT_INPUTS, 0.00, T)
    rho_liq = fluid.rhomass;

    % Test array of points
    rho_guess = linspace(rho_liq, fluid.rhomass_critical, N);

    % Evaluate the residual for each test-point
    residual = abs(arrayfun(@(rho) get_residual(T, rho, fluid), rho_guess));

    % Get the first local minimum (closest to saturation)
    index = find(islocalmin(residual), 1, 'first');
    if ~isempty(index)
        rho_guess = rho_guess(find(islocalmin(residual), 1, 'first'));
    else
        [~, index] = min(residual);
        rho_guess = rho_guess(index);
    end

end


function rho_guess = get_spinodal_vapor_density_guess(T, fluid, N)

    % Calculate saturated vapor density
    fluid.update(py.CoolProp.CoolProp.QT_INPUTS, 1.00, T)
    rho_vap = fluid.rhomass;

    % Test array of points
    rho_guess = linspace(rho_vap, fluid.rhomass_critical, N);

    % Evaluate the residual for each test-point
    residual = abs(arrayfun(@(rho) get_residual(T, rho, fluid), rho_guess));

    % Get the first local minimum (closest to saturation)
    index = find(islocalmin(residual), 1, 'first');
    if ~isempty(index)
        rho_guess = rho_guess(find(islocalmin(residual), 1, 'first'));
    else
        [~, index] = min(residual);
        rho_guess = rho_guess(index);
    end
    
end


function res = get_residual(T, rho, fluid)

    % Residual of the the equation K=(dp/drho)_T=0
    props = compute_properties_metastable_Td(T, rho, fluid);
    res = props.isothermal_bulk_modulus;

end