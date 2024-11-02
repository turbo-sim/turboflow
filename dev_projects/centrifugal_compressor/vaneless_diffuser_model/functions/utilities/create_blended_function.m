function blended_function = create_blended_function(f1, f2, x0, alpha)

    % Smooth blending of functions f1 and f2 at x0
    sigma = @(x) (1 + tanh((x-x0)/alpha))/2;
    blended_function = @(x) (1-sigma(x)).*f1(x) + sigma(x).*f2(x);

end