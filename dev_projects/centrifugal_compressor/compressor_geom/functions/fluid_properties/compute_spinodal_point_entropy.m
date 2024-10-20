function properties = compute_spinodal_point_entropy(s_in, fluid, rhoT_guess, NameValueArgs)

    % This function calculates the spinodal point corresponding to the
    % input entropy specified as argument.
    % 
    % The function verifies if the input entropy is within the limits of 
    % the spinodal line and throws an error if it is outside the range
    %
    % The standard calculation method solves the nonlinear system:
    %
    %   1. s(T,rho) - s_in = 0
    %   2. isothermal_bulk_modulus(T,rho) / p(T,rho) = 0
    %
    % Whereas the robust calculation method solves the nonlinear equation:
    %
    %   s_spinodal(T) - s_in = 0
    % 
    % where s_spinodal is the solution of the optimization problem:
    %
    %   min f(rho)=abs(isothermal_bulk_modulus(T, rho)) [at fixed T]
    %
    % The standard method is suitable for fluids with a well-posed equation
    % of state that satisfies the condition isothermal_bulk_modulus=0 at
    % the spinodal point
    %
    % The robust method is suitable for fluids with a ill-posed equation of
    % state that do not satisfy the condition isothermal_bulk_modulus=0 at
    % the spinodal point. In these cases the "pseudo" spinodal point is
    % defined as the point in which the absolute value of the isothermal
    % bulk modulus is a minimum (inflection point of the isotherm)
    %
    % The properties are evaluated using temperature-entropy function
    % calls to the Helmholtz energy equation of state.
    % 
    % The initial guess to solve the nonlinear system is obtained by 
    % precomputing the spinodal line and using the point with closest 
    % entropy as initial guess.
    %

    arguments
        s_in
        fluid
        rhoT_guess = -1
        NameValueArgs.T_margin (1, 1) double = 0.00
        NameValueArgs.method (1, 1) string = 'none'
        NameValueArgs.keep_initial_guess (1, 1) logical = false
    end

    % Check that the input entropy is within limits
    [spinodal_liq, spinodal_vap] = compute_spinodal_line(fluid, N_points=150, method=NameValueArgs.method);
    if s_in < spinodal_liq.smass(1)
        error('Input entropy is lower than the liquid spinodal entropy at the triple point: s_in=%0.2f, s_min=%0.2f', s_in, spinodal_liq.smass(1))
    end    
    if s_in > spinodal_vap.smass(end)
        error('Input entropy is higher than the vapor spinodal entropy at the triple point: s_in=%0.2f, s_max=%0.2f', s_in, spinodal_vap.smass(end))
    end

    % Compute critical entropy
    fluid.update(py.CoolProp.DmassT_INPUTS, fluid.rhomass_critical, fluid.T_critical)
    s_critical = fluid.smass;

    % Estimate the initial guess
    if rhoT_guess == -1
        T_spinodal = [spinodal_liq.T, spinodal_vap.T];
        s_spinodal = [spinodal_liq.smass, spinodal_vap.smass];
        rho_spinodal = [spinodal_liq.rhomass, spinodal_vap.rhomass];
        [~, index] = min(abs(s_in-s_spinodal));
        rhoT_guess = [rho_spinodal(index)*1.0, T_spinodal(index)+0.0];
    end

    % Solve the spinodal point problem using the specified method
    %   1. Standard solver a system of nonlinear equations
    %   2. Robust solves a constrained optimization problem where the
    %      absolute value of the isothermal bulk modulus is minimized 
    %      subject to the constraint that the input entropy is satisfied
    % The robust method is suitable for fluid with an ill-posed equation of
    % state such as nitrogen. For well posed fluids the standard and robust
    % methods should give the same results (up to the solver tolerance)
    if  NameValueArgs.keep_initial_guess
        x = rhoT_guess;
        properties = compute_properties_metastable_Td(x(2)+NameValueArgs.T_margin, x(1), fluid);

    elseif strcmp(NameValueArgs.method, 'standard')

        % Solve residual equation for isothermal bulk modulus and entropy  
        problem.x0 = rhoT_guess;
        problem.objective = @(rhoT_pair)get_spinodal_residual(rhoT_pair, s_in, fluid);
        problem.solver = 'fsolve';
        problem.options = optimoptions('fsolve', ...
                               'Algorithm','trust-region-dogleg', ...
                               'MaxIterations', 2000, ...
                               'MaxFunctionEvaluations', 10000, ...
                               'FunctionTolerance', 1e-16, ...
                               'OptimalityTolerance', 1e-16, ...
                               'StepTolerance', 1e-16, ...
                               'FiniteDifferenceType','forward', ...
                               'Display', 'none');
        [x, ~, exitflag, ~]  = fsolve(problem);
        if exitflag <= 0
            error('Spinodal point calculation did not converge')
        end
        properties = compute_properties_metastable_Td(x(2)+NameValueArgs.T_margin, x(1), fluid);

    
    elseif strcmp(NameValueArgs.method, 'robust')

        % Solve nested nonlinear equations
%         problem.x0 = [max(T_spinodal(index)-2.5, fluid.Ttriple), min(T_spinodal(index)+2.5, fluid.T_critical-1e5)];
        problem.x0 = T_spinodal(index);
        problem.objective = @(T)get_spinodal_entropy_residual(T, s_in, s_critical, fluid);
        problem.solver = 'fzero';
        problem.options = optimset('Display','notify');
        [T, ~, exitflag] = fzero(problem);
        if exitflag <= 0
            error('Spinodal point calculation did not converge')
        end
        properties = compute_spinodal_point(T, fluid, rho_spinodal(index), method='robust');

    else
        error("Invalid spinodal point computation method")
    end

end

function res = get_spinodal_residual(rhoT_pair, s_in, fluid)

    % The residual (i.e., isothermal bulk modulus) is normalized by
    % pressure to scale the problem. This is necessary to have a 
    % well-conditioned system of equations and achieve tigh convergence
    props = compute_properties_metastable_Td(rhoT_pair(2), rhoT_pair(1), fluid);
    B_res = props.isothermal_bulk_modulus/props.p;
    s_res = (props.smass - s_in)/s_in;
    res = [B_res; s_res];

end

function res = get_spinodal_entropy_residual(T, s_in, s_crit, fluid)
    
    % Evaluate spinidal point at the input temperature
    T = min(T, fluid.T_critical-1e-6);
    if s_in > s_crit
        rho_guess = "vapor";
    else
        rho_guess = "liquid";
    end
    spinodal_point = compute_spinodal_point(T, fluid, rho_guess, method='robust');
    res = (spinodal_point.smass - s_in)/s_in;

end



% Whereas the robust calculation method solves the constrained
% optimization problem:
%
%   min f(T,rho) = abs(isothermal_bulk_modulus(T,rho))
%   s.t. s(T,rho) - s_in = 0
% 
%         % Solve constrained optimization problem
%         problem.x0 = rhoT_guess;
%         problem.objective = @(rhoT_pair) get_spinodal_fval(rhoT_pair, fluid, p_spinodal(index));
%         problem.nonlcon = @(rhoT_pair) get_spinodal_constraint(rhoT_pair, s_in, fluid);
%         problem.solver = 'fmincon';
%         problem.options = optimoptions('fmincon', ...
%                                        'Algorithm','interior-point', ...
%                                        'ConstraintTolerance', 1e-12, ...
%                                        'FunctionTolerance', 1e-16, ...
%                                        'OptimalityTolerance', 1e-16, ...
%                                        'StepTolerance', 1e-14, ...
%                                        'MaxIterations', 1e3, ...
%                                        'MaxFunctionEvaluations', 1e4, ...
%                                        'Display','iter-detailed');
%         [x, ~, exitflag, ~]  = fmincon(problem);
% 
% function fval = get_spinodal_fval(rhoT_pair, fluid, p_ref)
% 
%     % Evaluate spinodal point normalized bulk modulus
%     props = compute_properties_metastable_Td(rhoT_pair(2), rhoT_pair(1), fluid);
%     fval = abs(props.isothermal_bulk_modulus/p_ref);  
% 
% end
% 
% function [c, c_eq] = get_spinodal_constraint(rhoT_pair, s_in, fluid)
% 
%     % Evaluate spinodal point entropy constraint
%     props = compute_properties_metastable_Td(rhoT_pair(2), rhoT_pair(1), fluid);
%     c_eq = (props.smass - s_in)/s_in;
%     c = [];
% 
% end


