function half_thickness = compute_thickness_distribution_NACA_default(x, t_max, t_te)

    % Compute the half-thickness distribution of an airfoil using the
    % NACA series 4 thickness distribution from:
    % "The characteristics of 78 related airfoil sections from tests in the
    % variable-density wind tunnel"
    
    % The coefficient E is computed such that the input trailing edge 
    % thickness is imposed exactly. Note that the condition A+B+C+D+E=0 
    % defines an airfoil with a sharp trailing edge
    A = +0.2969;
    B = -0.1260;
    C = -0.3516;
    D = +0.2843;
    % E = -0.1036;  % Sharp trailing edge
    E = (t_te/2)/(t_max/0.2) - (A + B + C + D);  % Finite thickness trailing edge
    half_thickness = (t_max/0.20)*(A*x.^0.5 + B*x + C*x.^2 + D*x.^3 + E.*x.^4);

end
