function properties = compute_pseudocritical_point(T, fluid, d_guess)

    arguments
        T (1, 1) double
        fluid
        d_guess (1, 1) double = fluid.rhomass_critical
    end


    % Evaluate the pseudocritical point at temperature T
    % The pseudocritical point is defined as the point where the isobaric
    % heat capacity of the fluid at a certain temperature is a maximum.
    % Along this line the fluid has a similar behavior as that close to the
    % critical point (high heat capacity and compressibility)
    options = optimoptions('fminunc', ...
                           'MaxFunctionEvaluations', 2000, ...
                           'MaxIterations', 500, ...
                           'OptimalityTolerance', 1e-16, ...
                           'StepTolerance', 1e-16, ...
                           'Display', 'none');
    [rho, ~, exitflag, output] = fminunc(@(rho)-get_isobaric_heat_capacity(rho, T, fluid), d_guess, options);
    if exitflag <= 0
        fprintf('Exitflag: %s\n', exitflag)
        fprintf('%s\n', output.message)
        error('Pseudocritical point calculation did not converge')
    end

    % Evaluate thermodynamic properties at the pseudocritical point
    properties = compute_properties_metastable_Td(T, rho, fluid);

end

function cp = get_isobaric_heat_capacity(rho, T, fluid)
    props = compute_properties_metastable_Td(T, rho, fluid);
    cp = props.cpmass;
end
