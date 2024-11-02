%% Initialize script
clear all
close all
clc

% ----------------------------------------------------------------------- %
% 
% Coolprop installation instructions
% 
% Type in terminal:
%   conda create -n matlab_env python=3.10
%   conda activate matlab_env
%   pip install coolprop
% 
% Type in Matlab command window
%   pyenv('Version', 'C:\Users\roagr\AppData\Local\miniconda3\envs\matlab_env\python.exe')
%   (remember to replace with the correct path to your python executable
% 
% ----------------------------------------------------------------------- %

% Create folder to save results
results_path = 'results';
if not(isfolder(results_path))
    mkdir(results_path)
end

% Define plot settings
addpath(genpath("./functions"))
set_plot_options()
save_figures = true;
defaultColors = get(groot, 'factoryAxesColorOrder');


%% Compute vaneless diffuser performance
% Define input parameters
parameters = struct();
parameters.fluid = 'air';
parameters.p0_in = 101325;
parameters.T0_in = 273.15+20;
parameters.Ma_in = 0.75;
parameters.alpha_in = 65*pi/180;
parameters.Cf = 0.020;
parameters.phi = 90*(pi/180);
parameters.div = 0*(pi/180);
parameters.b_in = 0.25;
parameters.r_in = 1.00;
parameters.r_out = 3.00*parameters.r_in;

% Evaluate diffuser model
solution = vaneless_diffuser_model(parameters);


%% Plot the pressure recovery coefficient of an ideal diffuser
fig_1 = figure(); ax = gca; hold on; box on; grid on;
axis square;
ax.Layer = "top";
xlabel({''; 'Radius ratio'})
ylabel({'Area ratio'; ''})
title({"Pressure recovery of an ideal diffuser"; ''})
xticks(1:1:5)
yticks(1:1:5)
xtickformat('%0.1f')
ytickformat('%0.1f')
RR = linspace(1, 5, 100);
AR = linspace(1, 5, 100);
[RR, AR] = meshgrid(RR, AR);
Cp_ideal = 1 - (sin(parameters.alpha_in)./RR).^2 - (cos(parameters.alpha_in)./AR).^2;
contourf(RR, AR, Cp_ideal, linspace(0.0, 1, 11), DisplayName='Pressure recovery', HandleVisibility='off', LineStyle='-', LineWidth=0.1)
[C,h] = contour(RR, AR, AR./RR, [0.8, 0.9, 1.0, 1.1, 1.2], color='black', DisplayName='Width ratio');
colormap(parula(12));
cb = colorbar(Location="eastoutside", Limits=[0, 1], Ticks=linspace(0.0, 1, 11));
cb.Ruler.TickLabelFormat = '%.1f';
cb.Label.String = {''; '$C_p$ -- Pressure recovery factor'};
cb.Label.Interpreter = 'Latex';
clabel(C,h, Interpreter="Latex", LabelSpacing=500)
legend(Location="southeast")


%% Plot the pressure recovery coefficient distribution
fig_2 = figure(); hold on; box on; grid on;
axis square;
xlabel({''; 'Radius ratio'})
ylabel({'Pressure recovery coefficient'; ''})
xtickformat('%0.1f')
ytickformat('%0.1f')
plot(solution.RR, solution.Cp, DisplayName='Real compressible', Color=defaultColors(1,:))
plot(solution.RR, solution.Cp_ideal, DisplayName='Ideal incompressible', Color=defaultColors(2,:))
legend(Location="southeast")


% % Plot Mach number distribution
% fig_3 = figure(); hold on; box on; grid on;
% axis square;
% xlabel({''; 'Radius ratio'})
% ylabel({'Mach number'; ''})
% xtickformat('%0.1f')
% ytickformat('%0.1f')
% plot(solution.RR, solution.Ma, DisplayName='Modulus')
% plot(solution.RR, solution.Ma_t, DisplayName='Tangential')
% plot(solution.RR, solution.Ma_m, DisplayName='Meridional')
% legend(Location="northeast")


%% Plot streamlines
fig_3 = figure(); hold on; box on; grid off;
axis image
xlabel({''; '$x$ coordinate'})
ylabel({'$y$ coordinate'; ''})
title({'Diffuser streamlines'; ''})
xtickformat('%0.1f')
ytickformat('%0.1f')
axis(1.1*parameters.r_out*[-1, 1, -1, 1])

theta = linspace(0, 2*pi, 100);
x_in = parameters.r_in*cos(theta);
y_in = parameters.r_in*sin(theta);
x_out = parameters.r_out*cos(theta);
y_out = parameters.r_out*sin(theta);
plot(x_in, y_in, 'k', HandleVisibility='off')
plot(x_out, y_out, 'k', HandleVisibility='off')

% Evaluate diffuser model
parameters.Cf = 0.000;
solution = vaneless_diffuser_model(parameters);

theta = linspace(0, 2*pi, 8);
for i = 1:numel(theta)
    x = solution.r.*cos(solution.theta + theta(i));
    y = solution.r.*sin(solution.theta + theta(i));
    if i == 1
        plot(x, y, DisplayName="Ideal compressible", Color=defaultColors(1,:))
    else
        plot(x, y, HandleVisibility='off', DisplayName='', Color=defaultColors(1,:))
    end

    x = solution.r.*cos(solution.theta_ideal + theta(i));
    y = solution.r.*sin(solution.theta_ideal + theta(i));
    if i == 1
        plot(x, y, DisplayName="Ideal incompressible", Color=defaultColors(2,:))
    else
        plot(x, y, HandleVisibility='off', Color=defaultColors(2,:))
    end

end

legend(Location='southeast')

% Save the figures
if save_figures
    exportgraphics(fig_1, fullfile(results_path, 'vaneless_diffuser_ideal_pressure_recovery.png'), Resolution=500)
    exportgraphics(fig_2, fullfile(results_path, 'vaneless_diffuser_real_vs_ideal_recovery.png'), Resolution=500)
    exportgraphics(fig_3, fullfile(results_path, 'vaneless_diffuser_compressible_vs_incompressible_streamlines.png'), Resolution=500)
end

