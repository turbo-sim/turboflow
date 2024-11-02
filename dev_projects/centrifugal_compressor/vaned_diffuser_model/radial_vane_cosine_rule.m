%% Initialize script
clear all
close all
clc

% Create folder to save results
results_path = 'results';
if not(isfolder(results_path))
    mkdir(results_path)
end

% Define plot settings
addpath(genpath("../common"))
set_plot_options()
defaultColors = get(groot, 'factoryAxesColorOrder');
N = 500;
u = linspace(0, 1, N);


%% Define the geometry of the cascade
% Cascade parameters
r_3 = 1.0;
r_5 = 1.5;
solidity = 1.5;
Z_vd = ceil(2*pi*solidity/(r_5/r_3-1));
d_theta = 2*pi/Z_vd;

% Wedge diffuser parameters
phi = 75*pi/180;        % Mean channel angle at inlet
eps = 0.0*pi/180;       % Wedge total angle
div = d_theta - eps;    % Divergence total angle    

% Airfoil diffuser camberline parameters
camberline_type = 'straight';
% camberline_type = 'circular_arc_conformal';
beta_1 = phi;
beta_2 = 50*pi/180;
% beta_2 = atan(sin(phi)./(sqrt((r_5/r_3).^2-sin(phi).^2)));

% Airfoil diffuser thickness distribution parameters
chord = r_5 - r_3;
loc_max = 0.150;
thickness_max = 0.050*chord;
thickness_trailing = 0.01*chord;
radius_leading = 0.01*chord;
wedge_trailing = 2.00*pi/180;


%% Plot the case geometry
% Plot the wedges and the channels
fig = figure(); hold on; box on; grid off;
axis image
% axis(1.25*r_5*[0, 1, 0, 1])
xlabel({''; '$x$ coordinate'})
ylabel({'$y$ coordinate'; ''})
xtickformat('%0.1f')
ytickformat('%0.1f')

% Plot the radii at the inlet and outlet of the diffuser
angle = linspace(0, 2*pi, 100);
plot(r_3*cos(angle), r_3*sin(angle), 'k', HandleVisibility='off')
plot(r_5*cos(angle), r_5*sin(angle), 'k', HandleVisibility='off')


% Plot the wedges and the channels
for i = 1:Z_vd
    
    % Compute blade coordinates
    theta_0 = 2*pi*(i-1)/Z_vd;

    % Plot wedge diffuser
    [x_wedge_1, y_wedge_1] = compute_camberline_radial('straight',r_3, r_5, phi+eps/2, [], theta_0, u);
    [x_wedge_2, y_wedge_2] = compute_camberline_radial('straight',r_3, r_5, phi-eps/2, [], theta_0, u);
%     plot(x_wedge_1, y_wedge_1, Color='k', LineStyle="-", LineWidth=1.25)
%     plot(x_wedge_2, y_wedge_2, Color='k', LineStyle="-", LineWidth=1.25)

    % Plot channel edges
    lambda = (sqrt((r_5/r_3)^2 - sin(phi+eps/2)^2) - cos(phi+eps/2));
    w_in = r_3*(sin(phi+eps/2) - sin(phi + eps/2 - d_theta))/cos(d_theta/2 - eps/2);
    w_out = r_3*(sin(d_theta+phi-eps/2) + lambda*sin(d_theta-eps) - sin(phi-eps/2))/cos(d_theta/2-eps/2);
    AR = w_out/w_in;
    length = (w_out - w_in)/(2*tan(div/2));
    X1_in = r_3*cos(theta_0+d_theta);
    Y1_in = r_3*sin(theta_0+d_theta);
    X2_in = r_3*cos(theta_0+d_theta) + w_in*sin(theta_0+d_theta/2+phi);  
    Y2_in = r_3*sin(theta_0+d_theta) - w_in*cos(theta_0+d_theta/2+phi);
    X1_out = r_3*cos(theta_0) + r_3*lambda*cos(theta_0+phi+eps/2);
    Y1_out = r_3*sin(theta_0) + r_3*lambda*sin(theta_0+phi+eps/2);
    X2_out = r_3*cos(theta_0) + r_3*lambda*cos(theta_0+phi+eps/2) - w_out*sin(theta_0+d_theta/2+phi);  
    Y2_out = r_3*sin(theta_0) + r_3*lambda*sin(theta_0+phi+eps/2) + w_out*cos(theta_0+d_theta/2+phi);
%     plot([X1_in, X2_in], [Y1_in, Y2_in], 'r+--')
%     plot([X1_out, X2_out], [Y1_out, Y2_out], 'r+--')
    
    % Plot airfoil diffuser camberline
    [x_airfoil, y_airfoil] = compute_camberline_radial(camberline_type, r_3, r_5, beta_1, beta_2, theta_0, u);
    plot(x_airfoil, y_airfoil, Color=defaultColors(1,:), LineStyle="-")
    
    % Plot the airfoil diffuser coordinates
    [x_airfoil, y_airfoil] = compute_blade_coordinates_radial(camberline_type, r_3, r_5, beta_1, beta_2, theta_0, loc_max, thickness_max, thickness_trailing, wedge_trailing, radius_leading, N);
    plot(x_airfoil, y_airfoil, Color=defaultColors(1,:), LineStyle="-", LineWidth=0.25)
    
    % Compute the inlet and outlet throats
    calculation_method = "intersection"; % "projection";
    if i == 1

        % Compute the coordinates of the throat
        sol_in = compute_leading_edge_throat(camberline_type, r_3, r_5, beta_1, beta_2, d_theta, theta_0, calculation_method);
        sol_out = compute_trailing_edge_throat(camberline_type, r_3, r_5, beta_1, beta_2, d_theta, theta_0, calculation_method);
        x_inlet = [sol_in.x_leading, sol_in.x_throat];
        y_inlet = [sol_in.y_leading, sol_in.y_throat];
        x_outlet = [sol_out.x_trailing, sol_out.x_throat];
        y_outlet = [sol_out.y_trailing, sol_out.y_throat];
        
        % Compute the divergence angle of the channel
        w_in = sol_in.w_throat;
        w_out = sol_out.w_throat;
        length_airfoil_channel = sqrt((mean(x_outlet)-mean(x_inlet))^2 + (mean(y_outlet)-mean(y_inlet))^2);
        div_airfoil = 2*atan(0.5*(w_out - w_in)/length_airfoil_channel);
        AR_airfoil = w_out/w_in;

    else

        % Rotate coordinates to reduce computational time
        [x_inlet, y_inlet] = rotate_counterclockwise_2D(x_inlet, y_inlet, d_theta);
        [x_outlet, y_outlet] = rotate_counterclockwise_2D(x_outlet, y_outlet, d_theta);

    end

    % Plot the inlet and outlet throats
    plot(x_inlet, y_inlet, 'b+:')
    plot(x_outlet, y_outlet, 'b+:')


end


%% Check the accuracy of the calculations
% Verify the diffuser channel parameters
fprintf("\nDiffuser channel parameter comparison\n")
fprintf("%-45s: %0.8f\n", "Area ratio of the straight channel", AR)
fprintf("%-45s: %0.8f\n", "Area ratio of the airfoil channel", AR_airfoil)
fprintf("%-45s: %0.8f\n", "Length of the straight channel", length)
fprintf("%-45s: %0.8f\n", "Length of the airfoil channel", length_airfoil_channel)
fprintf("%-45s: %0.8f\n", "Divergence angle of the straight channel", div*180/pi)
fprintf("%-45s: %0.8f\n", "Divergence angle of the airfoil channel", div_airfoil*180/pi)

% Verify the cosine rule approximation at the inlet
theta_0 = pi/2; u = [0, 1];
[~, ~, ~,  theta_airfoil, beta_airfoil, phi_airfoil] = compute_camberline_radial(camberline_type, r_3, r_5, beta_1, beta_2, theta_0, u);
phi = phi_airfoil(1) - theta_0;
w_in_straight_polar = r_3*(sin(phi+eps/2) - sin(phi + eps/2 - d_theta))/cos(d_theta/2 - eps/2);
w_in_cosine_rule = d_theta*cos(beta_1)*r_3;
fprintf("\nInlet throat width calculation comparison\n")
fprintf("%-25s \t %-25s\n", "Computation method:", "Numerical value:")
fprintf("%-25s \t %-25.6f\n", "Intersection", sol_in.w_throat)
fprintf("%-25s \t %-25.6f\n", "Straight polar", w_in_straight_polar)
fprintf("%-25s \t %-25.6f\n", "Cosine rule", w_in_cosine_rule)

% Verify the cosine rule approximation at the outlet
phi = phi_airfoil(end)- theta_0;
beta_2 = beta_airfoil(end);
lambda = (sqrt((r_5/r_3)^2 - sin(phi+eps/2)^2) - cos(phi+eps/2));
w_out_straight_polar = r_3*(sin(d_theta+phi-eps/2) + lambda*sin(d_theta-eps) - sin(phi-eps/2))/cos(d_theta/2-eps/2);
w_out_cosine_rule = d_theta*cos(beta_2)*r_5;
fprintf("\nOutlet throat width calculation comparison\n")
fprintf("%-25s \t %-25s\n", "Computation method:", "Numerical value:")
fprintf("%-25s \t %-25.6f\n", "Intersection", sol_out.w_throat)
fprintf("%-25s \t %-25.6f\n", "Straight polar", w_out_straight_polar)
fprintf("%-25s \t %-25.6f\n", "Cosine rule", w_out_cosine_rule)



% Comments
% 
% The cascade throat was calculated exactly for the case of straight blades
% using an analytic result. In addition, the cascade throat is also
% calculated numerically solving the intersection between the camberline of
% the adjacent blade with the throat line (non-linear system of equations)
% The results of both approaches match exactly when the blades are
% straight. The analytic formula for straight blades is a reasonable
% approximation of the exact throat when the blades have small curvature,
% but it overestimates the throat area when the blades have curvature
%
% The cosine rule approximates very well the analytic calculation for the
% throat area of straight vanes (wedge diffuser with eps=0). The
% approximation is better as the number of blades increases. The
% approximation is reasonably accurate for Z_vd>10
% Note that the angle for the cosine rule is not simply the blade metal
% angle, but it also includes the half pitch angle d_theta/2 = pi/Z_vd
%
% The cosine rule approximation is very accurate when the curvature of the
% channel is small. In reality, the blades will have some curvature because
% otherwise they would be straight vanes with a very high diverence angle
% that could easily stall. If the blades have some curvature, the cosine
% rule approximation accuracy decreases when the blades are curved
% 
% In conclusion, the modified cosine rule seems to be very simple and
% accurate way to estimate the throat area at the inlet and outlet of the
% cascade when the blades are straight. However, the accuracy of the
% approximation decreases when the blades are curved. When the blades have
% significant curvature, it becomes necessary to estimate the throat area
% numerically as the intersection of the camberline with the throat line.
% This requires the solution of a system of 2 nonlinear equations
% (1 for the intersection of the x coordinate, and another for y)
%
