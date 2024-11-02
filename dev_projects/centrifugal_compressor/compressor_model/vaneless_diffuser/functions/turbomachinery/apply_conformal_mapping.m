function [x, y] = apply_conformal_mapping(x, y, x1, y1, r1, r2, c_ax, theta_0)
    
    % Convert [x,y] to [r,theta] coordinates preserving the wrap angle
    % dy/dx = tan(beta) = r*d(theta)/dr
    r = r1*exp(log(r2/r1)*(x-x1)/c_ax);
    theta = theta_0 + log(r2/r1)/c_ax*(y-y1);
    x = r.*cos(theta);
    y = r.*sin(theta);

end
